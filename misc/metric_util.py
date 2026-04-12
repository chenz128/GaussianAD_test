
import numpy as np
from mmengine import MMLogger
logger = MMLogger.get_instance('selfocc')
import torch.distributed as dist
import torch
from model.utils import common_utils
import pickle
import os


class MeanIoU:

    def __init__(self,
                 class_indices,
                #  ignore_label: int,
                 empty_label,
                 label_str,
                 use_mask=False,
                 dataset_empty_label=17,
                 filter_minmax=True,
                 name = 'none'):
        self.class_indices = class_indices
        self.num_classes = len(class_indices)
        # self.ignore_label = ignore_label
        self.empty_label = empty_label
        self.dataset_empty_label = dataset_empty_label
        self.label_str = label_str
        self.use_mask = use_mask
        self.filter_minmax = filter_minmax
        self.name = name

    def reset(self) -> None:
        self.total_seen = torch.zeros(self.num_classes+1).cuda()
        self.total_correct = torch.zeros(self.num_classes+1).cuda()
        self.total_positive = torch.zeros(self.num_classes+1).cuda()

    def _after_step(self, outputs, targets, mask=None):
        if not isinstance(targets, (torch.Tensor, np.ndarray)):
            assert mask is None
            labels = torch.from_numpy(targets['semantics']).cuda()
            masks = torch.from_numpy(targets['mask_camera']).bool().cuda()
            targets = labels
            targets[targets == self.dataset_empty_label] = self.empty_label
            if self.filter_minmax:
                max_z = (targets != self.empty_label).nonzero()[:, 2].max()
                min_z = (targets != self.empty_label).nonzero()[:, 2].min()
                outputs[..., (max_z + 1):] = self.empty_label
                outputs[..., :min_z] = self.empty_label
            if self.use_mask:
                outputs = outputs[masks]
                targets = targets[masks]
        else:
            if mask is not None:
                outputs = outputs[mask]
                targets = targets[mask]

        for i, c in enumerate(self.class_indices):
            self.total_seen[i] += torch.sum(targets == c).item()
            self.total_correct[i] += torch.sum((targets == c)
                                               & (outputs == c)).item()
            self.total_positive[i] += torch.sum(outputs == c).item()

        self.total_seen[-1] += torch.sum(targets != self.empty_label).item()
        self.total_correct[-1] += torch.sum((targets != self.empty_label)
                                            & (outputs != self.empty_label)).item()
        self.total_positive[-1] += torch.sum(outputs != self.empty_label).item()

    def _after_epoch(self):
        if dist.is_initialized():
            dist.all_reduce(self.total_seen)
            dist.all_reduce(self.total_correct)
            dist.all_reduce(self.total_positive)
            dist.barrier()

        ious = []
        precs = []
        recas = []

        for i in range(self.num_classes):
            if self.total_positive[i] == 0:
                precs.append(0.)
            else:
                cur_prec = self.total_correct[i] / self.total_positive[i]
                precs.append(cur_prec.item())
            if self.total_seen[i] == 0:
                ious.append(1)
                recas.append(1)
            else:
                cur_iou = self.total_correct[i] / (self.total_seen[i]
                                                   + self.total_positive[i]
                                                   - self.total_correct[i])
                cur_reca = self.total_correct[i] / self.total_seen[i]
                ious.append(cur_iou.item())
                recas.append(cur_reca)

        miou = np.mean(ious)
        # logger = get_root_logger()
        logger.info(f'Validation per class iou {self.name}:')
        for iou, prec, reca, label_str in zip(ious, precs, recas, self.label_str):
            logger.info('%s : %.2f%%, %.2f, %.2f' % (label_str, iou * 100, prec, reca))

        logger.info(self.total_seen.int())
        logger.info(self.total_correct.int())
        logger.info(self.total_positive.int())

        occ_iou = self.total_correct[-1] / (self.total_seen[-1]
                                            + self.total_positive[-1]
                                            - self.total_correct[-1])
        # logger.info(f'iou: {occ_iou}')

        return miou * 100, occ_iou * 100


class DetMetric:
    def __init__(self,
                 cfg,
                 eval_output_dir,
                 ):
        self.recall_thresh_list = cfg.model.decoder.model_cfg.post_processing.recall_thresh_list
        self.class_names = cfg.det_config.class_names
        self.eval_output_dir = eval_output_dir

        self.cfg = cfg

    def generate_prediction_dicts(self, batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7 or 9), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """

        def get_template_prediction(num_samples):
            box_dim = 9
            ret_dict = {
                'name': np.zeros(num_samples), 'score': np.zeros(num_samples),
                'boxes_lidar': np.zeros([num_samples, box_dim]), 'pred_labels': np.zeros(num_samples)
            }
            return ret_dict

        def generate_single_sample_dict(box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes
            pred_dict['pred_labels'] = pred_labels

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            single_pred_dict = generate_single_sample_dict(box_dict)
            single_pred_dict['frame_id'] = batch_dict['frame_id'][index]
            if 'metadata' in batch_dict:
                single_pred_dict['metadata'] = batch_dict['metadata'][index]
            annos.append(single_pred_dict)

        return annos

    def _after_step(self, pred_dicts, metas,  metric):
        ret_dict = pred_dicts[0]['recall_dicts']
        batch_dict = {'frame_id': metas['frame_id'],'metadata': metas['metadata']}
        disp_dict = {}
        # for cur_thresh in self.recall_thresh_list:
        #     metric['recall_roi_%s' % str(cur_thresh)] = 0
        #     metric['recall_rcnn_%s' % str(cur_thresh)] = 0

        annos = self.generate_prediction_dicts(batch_dict, pred_dicts, self.class_names)
        return annos, metric

    def _after_epoch(self, local_rank, distributed, dataset, det_annos, metric):
        ret_dict = {}
        if distributed:
            rank, world_size = common_utils.get_dist_info()
            det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=os.path.join(self.eval_output_dir, 'tmpdir'), eval_output_dir=self.eval_output_dir)

        if local_rank != 0:
            return {}

        return det_annos
