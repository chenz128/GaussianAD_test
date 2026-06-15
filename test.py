# try:
#     from vis import save_occ
# except:
#     print('Load Occupancy Visualization Tools Failed.')
import time, argparse, os.path as osp, os
import torch, numpy as np
import torch.distributed as dist

from mmengine import Config
from mmengine.runner import set_random_seed
from mmengine.logging import MMLogger
from mmseg.models import build_segmentor
from dataset.metric_stp3 import compute_planner_metric_stp3
from dataset.dataset import output_to_vecs
import mmengine
from tqdm import tqdm
import gc

import warnings
warnings.filterwarnings("ignore")


def pass_print(*args, **kwargs):
    pass

def main(local_rank, args):
    # global settings
    set_random_seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # load config
    cfg = Config.fromfile(args.py_config)
    cfg.work_dir = args.work_dir

    # init DDP
    if args.gpus > 1:
        distributed = True
        ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
        port = os.environ.get("MASTER_PORT", "20507")
        hosts = int(os.environ.get("WORLD_SIZE", 1))  # number of nodes
        rank = int(os.environ.get("RANK", 0))  # node id
        gpus = torch.cuda.device_count()  # gpus per node
        print(f"tcp://{ip}:{port}")
        dist.init_process_group(
            backend="nccl", init_method=f"tcp://{ip}:{port}",
            world_size=hosts * gpus, rank=rank * gpus + local_rank)
        world_size = dist.get_world_size()
        cfg.gpu_ids = range(world_size)
        torch.cuda.set_device(local_rank)

        if local_rank != 0:
            import builtins
            builtins.print = pass_print
    else:
        distributed = False
        world_size = 1

    writer = None
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_name = args.log_name if args.log_name else f'test_{timestamp}'
    log_file = osp.join(args.work_dir, f'{log_name}.log')
    logger = MMLogger('selfocc', log_file=log_file)
    MMLogger._instance_dict['selfocc'] = logger
    logger.info(f'Config:\n{cfg.pretty_text}')

    # build model
    import model
    from dataset import get_dataloader

    my_model = build_segmentor(cfg.model)
    my_model.init_weights()
    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    logger.info(f'Number of params: {n_parameters}')
    if distributed:
        if cfg.get('syncBN', True):
            my_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(my_model)
            logger.info('converted sync bn.')

        find_unused_parameters = cfg.get('find_unused_parameters', False)
        ddp_model_module = torch.nn.parallel.DistributedDataParallel
        my_model = ddp_model_module(
            my_model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
        raw_model = my_model.module
    else:
        my_model = my_model.cuda()
        raw_model = my_model
    logger.info('done ddp model')

    train_dataset_loader, val_dataset_loader = get_dataloader(
        cfg.train_dataset_config,
        cfg.val_dataset_config,
        cfg.train_loader,
        cfg.val_loader,
        dist=distributed,
        val_only=True)

    # resume and load
    cfg.resume_from = ''
    if osp.exists(osp.join(args.work_dir, 'latest.pth')):
        cfg.resume_from = osp.join(args.work_dir, 'latest.pth')
    if args.resume_from:
        cfg.resume_from = args.resume_from

    logger.info('resume from: ' + cfg.resume_from)
    logger.info('work dir: ' + args.work_dir)

    if cfg.resume_from and osp.exists(cfg.resume_from):
        map_location = 'cpu'
        ckpt = torch.load(cfg.resume_from, map_location=map_location)
        raw_model.load_state_dict(ckpt['state_dict'], strict=True)
        print(f'successfully resumed.')
    elif cfg.load_from:
        ckpt = torch.load(cfg.load_from, map_location='cpu')
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        try:
            print(raw_model.load_state_dict(state_dict, strict=False))
        except RuntimeError:
            # Skip keys with incompatible tensor shapes (e.g. neck channels).
            model_state = raw_model.state_dict()
            filtered_state_dict = {}
            skipped = []
            for k, v in state_dict.items():
                if k in model_state and hasattr(v, 'shape') and model_state[k].shape == v.shape:
                    filtered_state_dict[k] = v
                else:
                    skipped.append(k)

            print(f'filtered {len(skipped)} incompatible keys from load_from checkpoint')
            print(raw_model.load_state_dict(filtered_state_dict, strict=False))

    print_freq = cfg.print_freq
    from misc.metric_util import MeanIoU, DetMetric
    occ_label_str = ['barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
         'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
         'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
         'vegetation']
    miou_metric = MeanIoU(
        list(range(1, 17)),
        17, #17,
        occ_label_str,
         True, 17, filter_minmax=False)
    miou_metric.reset()

    # ---- future occ (occ_flow) evaluation: 6 frames x 0.5s = 3s ----
    num_future = 6
    flow_miou_metrics = []
    for fi in range(num_future):
        m = MeanIoU(
            list(range(1, 17)), 17, occ_label_str,
            True, 17, filter_minmax=False,
            name='future_%.1fs' % ((fi + 1) * 0.5))
        m.reset()
        flow_miou_metrics.append(m)

    det_metric = DetMetric(
        cfg,
        args.work_dir
    )
    metric = {
        'gt_num': 0,
    }
    det_result = []
    planner_result = []
    map_result = []

    my_model.eval()
    os.environ['eval'] = 'true'

    with torch.no_grad():
        for i_iter_val, data in enumerate(val_dataset_loader):
            # if i_iter_val > 8:
            #     break

            for k in list(data.keys()):
                if isinstance(data[k], torch.Tensor):
                    data[k] = data[k].cuda()
            input_imgs = data.pop('img')
            result_dict = my_model(imgs=input_imgs, metas=data)

            sample_token = data['img_metas'][0].data['sample_idx']

            os.makedirs('eval', exist_ok=True)

            planner_metric_results = compute_planner_metric_stp3(
                pred_ego_fut_trajs = result_dict['ego_fut_preds'][result_dict['metas']['ego_fut_cmd']==1],
                gt_ego_fut_trajs = result_dict['metas']['ego_fut_trajs'],
                gt_agent_boxes = result_dict['metas']['gt_boxes_planner'][0],
                gt_agent_feats = result_dict['metas']['attr_labels_planner'],
                fut_valid_flag = result_dict['metas']['fut_valid_flag']
            )
            planner_result += [planner_metric_results]

            # annos, metric = det_metric._after_step(result_dict['det_res'],result_dict['metas'], metric)
            # det_result += annos

            # map_pred_anno = {}
            # vecs = output_to_vecs(result_dict)
            # map_pred_anno['sample_token'] = sample_token
            # pred_vec_list=[]
            # map_mapped_class_names = ['divider', 'ped_crossing','boundary']
            # for i, vec in enumerate(vecs):
            #     name = map_mapped_class_names[vec['label']]
            #     anno = dict(
            #         pts=vec['pts'],
            #         pts_num=len(vec['pts']),
            #         cls_name=name,
            #         type=vec['label'],
            #         confidence_level=vec['score'])
            #     pred_vec_list.append(anno)
            # map_pred_anno['vectors'] = pred_vec_list
            # map_result += [map_pred_anno]

            for idx, pred in enumerate(result_dict['pred_occ'][-1]):
                pred_occ = pred.argmax(0)
                gt_occ = result_dict['sampled_label'][idx]
                miou_metric._after_step(pred_occ, gt_occ)

            # future occ (occ_flow): evaluate each of the 6 future frames (0.5~3.0s)
            occ_flow = result_dict.get('occ_flow', None)
            if occ_flow is not None:
                for fi in range(min(num_future, len(occ_flow))):
                    fdict = occ_flow[fi][0]
                    if not fdict['flow_valid_flag']:
                        continue
                    pred_f = fdict['pred_flow'][0].argmax(0)
                    gt_f = fdict['sampled_label']
                    if isinstance(gt_f, np.ndarray):
                        gt_f = torch.from_numpy(gt_f)
                    gt_f = gt_f.to(pred_f.device).reshape(-1)
                    flow_miou_metrics[fi]._after_step(pred_f, gt_f)

            if i_iter_val % print_freq == 0 and local_rank == 0:
                logger.info('[EVAL] Iter %5d'%(i_iter_val))

    miou, iou2 = miou_metric._after_epoch()
    logger.info(f'mIoU: {miou}, iou2: {iou2}')
    miou_metric.reset()

    # ---- future occ mIoU summary (0.5~3.0s) ----
    logger.info('-------------- Future Occupancy (occ_flow) --------------')
    future_mious = []
    for fi in range(num_future):
        f_miou, f_iou2 = flow_miou_metrics[fi]._after_epoch()
        logger.info('[Future %.1fs] mIoU: %.2f, iou(geo): %.2f' % (
            (fi + 1) * 0.5, f_miou, f_iou2))
        future_mious.append(f_miou)
        flow_miou_metrics[fi].reset()
    if len(future_mious) > 0:
        logger.info('[Future avg 0.5~3.0s] mIoU: %.2f' % np.mean(future_mious))

    results_total_single  = {
            # "det_results": det_result,
            "planner_results": planner_result,
            # "map_results": map_result
        }

    results_total = det_metric._after_epoch(local_rank, distributed, val_dataset_loader.dataset, results_total_single, metric)

    if local_rank == 0:
        eval_kwargs = dict(metric='bbox', map_metric='chamfer', pipeline=None)
        print(val_dataset_loader.dataset.evaluate(results_total, **eval_kwargs))

    if writer is not None:
        writer.close()


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/tpv_lidarseg.py')
    parser.add_argument('--work-dir', type=str, default='./out/tpv_lidarseg')
    parser.add_argument('--resume-from', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--vis-occ', action='store_true', default=False)
    parser.add_argument('--log-name', type=str, default='', help='custom log file name (without .log)')
    args = parser.parse_args()

    ngpus = torch.cuda.device_count()
    args.gpus = ngpus
    print(args)

    if ngpus > 1:
        torch.multiprocessing.spawn(main, args=(args,), nprocs=args.gpus)
    else:
        main(0, args)
