"""Current and six-frame future OCC evaluation for the no-Planner model."""

import argparse
import os
import os.path as osp
import time
import warnings

import numpy as np
import torch
import torch.distributed as dist
from mmengine import Config
from mmengine.logging import MMLogger
from mmengine.runner import set_random_seed
from mmseg.models import build_segmentor


warnings.filterwarnings('ignore')


def _quiet_print(*args, **kwargs):
    del args, kwargs


def _load_checkpoint(model, path):
    if not path or not osp.isfile(path):
        raise FileNotFoundError(f'checkpoint not found: {path}')
    checkpoint = torch.load(path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=True)


def main(local_rank, args):
    set_random_seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    cfg = Config.fromfile(args.py_config)
    cfg.work_dir = args.work_dir
    os.makedirs(args.work_dir, exist_ok=True)

    distributed = args.gpus > 1
    if distributed:
        address = os.environ.get('MASTER_ADDR', '127.0.0.1')
        port = os.environ.get('MASTER_PORT', '21874')
        dist.init_process_group(
            backend='nccl',
            init_method=f'tcp://{address}:{port}',
            world_size=args.gpus,
            rank=local_rank,
        )
        torch.cuda.set_device(local_rank)
        cfg.gpu_ids = range(args.gpus)
        if local_rank != 0:
            import builtins
            builtins.print = _quiet_print

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_name = args.log_name or f'test_occ_{timestamp}'
    logger = MMLogger(
        'selfocc', log_file=osp.join(args.work_dir, f'{log_name}.log'))
    MMLogger._instance_dict['selfocc'] = logger

    # Import registrations only after the logger exists, matching test.py.
    import model  # noqa: F401
    from dataset import get_dataloader
    from misc.metric_util import MeanIoU

    network = build_segmentor(cfg.model)
    network.init_weights()
    if distributed and cfg.get('syncBN', True):
        network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(network)
    network = network.cuda()
    _load_checkpoint(network, args.resume_from)

    if distributed:
        network = torch.nn.parallel.DistributedDataParallel(
            network,
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=cfg.get('find_unused_parameters', False),
        )

    _, val_loader = get_dataloader(
        cfg.train_dataset_config,
        cfg.val_dataset_config,
        cfg.train_loader,
        cfg.val_loader,
        dist=distributed,
        val_only=True,
    )

    labels = [
        'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
        'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
        'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
        'vegetation',
    ]
    current_metric = MeanIoU(
        list(range(1, 17)), 17, labels,
        True, 17, filter_minmax=False, name='current')
    current_metric.reset()
    future_metrics = []
    for step in range(6):
        metric = MeanIoU(
            list(range(1, 17)), 17, labels,
            True, 17, filter_minmax=False,
            name=f'future_{0.5 * (step + 1):.1f}s')
        metric.reset()
        future_metrics.append(metric)

    network.eval()
    os.environ['eval'] = 'true'
    with torch.no_grad():
        for iteration, data in enumerate(val_loader):
            for key in list(data):
                if isinstance(data[key], torch.Tensor):
                    data[key] = data[key].cuda()
            images = data.pop('img')
            outputs = network(imgs=images, metas=data)

            for index, prediction in enumerate(outputs['pred_occ'][-1]):
                current_metric._after_step(
                    prediction.argmax(0), outputs['sampled_label'][index])

            occ_flow = outputs.get('occ_flow')
            if occ_flow is not None:
                for step in range(min(6, len(occ_flow))):
                    frame = occ_flow[step][0]
                    if not frame['flow_valid_flag']:
                        continue
                    prediction = frame['pred_flow'][0].argmax(0)
                    target = frame['sampled_label']
                    if isinstance(target, np.ndarray):
                        target = torch.from_numpy(target)
                    future_metrics[step]._after_step(
                        prediction, target.to(prediction.device).reshape(-1))

            if iteration % cfg.print_freq == 0 and local_rank == 0:
                logger.info(f'[EVAL OCC] Iter {iteration:5d}')

    current_miou, current_geo = current_metric._after_epoch()
    if local_rank == 0:
        logger.info(
            '[Current] mIoU: %.2f, iou(geo): %.2f',
            current_miou, current_geo)

    future_mious = []
    future_geos = []
    for step, metric in enumerate(future_metrics):
        future_miou, future_geo = metric._after_epoch()
        future_mious.append(future_miou)
        future_geos.append(future_geo)
        if local_rank == 0:
            logger.info(
                '[Future %.1fs] mIoU: %.2f, iou(geo): %.2f',
                0.5 * (step + 1), future_miou, future_geo)

    if local_rank == 0:
        logger.info(
            '[Future avg 0.5~3.0s] mIoU: %.2f, iou(geo): %.2f',
            float(np.mean(future_mious)), float(np.mean(future_geos)))

    if distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--py-config', required=True)
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--resume-from', required=True)
    parser.add_argument('--log-name', default='')
    parser.add_argument('--seed', type=int, default=42)
    parsed = parser.parse_args()
    parsed.gpus = torch.cuda.device_count()
    if parsed.gpus > 1:
        torch.multiprocessing.spawn(
            main, args=(parsed,), nprocs=parsed.gpus)
    else:
        main(0, parsed)
