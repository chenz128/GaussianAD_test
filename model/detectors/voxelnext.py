from .detector3d_template import Detector3DTemplate
from mmengine.registry import MODELS
from mmengine import build_from_cfg
import torch
# from mmcv.runner import force_fp32
from torch.nn import functional as F

@MODELS.register_module()
class VoxelNeXt(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, det_config):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=det_config)
        self.module_list = self.build_networks()

    def prepare_gaussian_args(self, batch_dict):
        gaussians = batch_dict['representation_temp']['gaussian']
        means = gaussians.means # b, g, 3
        scales = gaussians.scales # b, g, 3
        rotations = gaussians.rotations # b, g, 4
        opacities = gaussians.opacities # b, g, 1
        semantics = gaussians.semantics # b, g, c
        gs_output = torch.cat([means,scales,rotations,opacities,semantics],dim=-1)
        return gs_output

    def forward(self, batch_dict):
        gaussians = self.prepare_gaussian_args(batch_dict)
        batch_dict['gaussian_output'] = gaussians
        batch_dict['batch_size'] = gaussians.shape[0]
        for i, cur_module in enumerate(self.module_list):
            if i == 0:
                voxels, num_points, coors = self.voxelize(gaussians, cur_module)
                batch_dict.update({
                    'voxels': voxels,
                    'voxel_num_points': num_points,
                    'voxel_coords': coors,})
            else:
                batch_dict = cur_module(batch_dict)

        if self.training:
            return batch_dict
        else:
            pred_dicts = self.post_processing(batch_dict)
            return {"det_res":pred_dicts}

    @torch.no_grad()
    # @force_fp32()
    def voxelize(self, points, cur_module):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """
        voxels, coors, num_points = [], [], []
        for i, res in enumerate(points):
            res_voxels, res_coors, res_num_points = cur_module(res)
            # v, f, p = res_voxels.shape
            # if v < 1:
            #     self.logger.warning(f"points range exceed! {i}")
            #     res_voxels = res_voxels.new_zeros(8000, f, p)
            #     res_coors = res_coors.new_zeros(8000, 3)
            #     res_num_points = res_num_points.new_zeros(8000)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode="constant", value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def get_training_loss(self):

        disp_dict = {}
        loss, tb_dict = self.dense_head.get_loss()

        return loss, tb_dict, disp_dict

    def post_processing(self, batch_dict):
        post_process_cfg = self.model_cfg.post_processing
        batch_size = batch_dict['batch_size']
        final_pred_dict = batch_dict['final_box_dicts']
        recall_dict = {}
        for index in range(batch_size):
            pred_boxes = final_pred_dict[index]['pred_boxes']

            recall_dict = self.generate_recall_record(
                box_preds=pred_boxes,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.recall_thresh_list
            )

            final_pred_dict[index].update({"recall_dicts":recall_dict})
        return final_pred_dict
