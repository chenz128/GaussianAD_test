"""
Export occupancy voxels (pred + gt) for a single val frame to an .npz file.

Runs the model forward on the selected frame and saves:
  xyz   : (N, 3) float32  — voxel center coords (ego frame)
  pred  : (N,)   int16    — predicted class per voxel (0..17, 17=empty)
  gt    : (N,)   int16    — ground-truth class per voxel (0..17, 17=empty)

The GT is identical across checkpoints (same sampled_label), so any of the
exported npz files can be used as the GT source for plotting.

Usage (single GPU):
  python tools/viz/export_occ_npz.py \
    --py-config out/<run>/<cfg>.py --work-dir out/<run> \
    --resume-from out/<run>/latest.pth --vis-index 0 --out out/<run>/vis/val_0_occ.npz
"""
import argparse
import os
import os.path as osp
import time
import warnings

import numpy as np
import torch

from mmengine import Config
from mmengine.runner import set_random_seed
from mmengine.logging import MMLogger
from mmseg.models import build_segmentor

warnings.filterwarnings("ignore")


def main(args):
    set_random_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    cfg = Config.fromfile(args.py_config)
    cfg.work_dir = args.work_dir

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(args.work_dir, f'export_occ_{timestamp}.log')
    logger = MMLogger('selfocc', log_file=log_file)
    MMLogger._instance_dict['selfocc'] = logger

    import model  # noqa: F401  (registers modules + resolves GaussianPrediction)
    from dataset import get_dataloader

    my_model = build_segmentor(cfg.model)
    my_model.init_weights()
    my_model = my_model.cuda()
    raw_model = my_model

    # limit val dataset to the requested frame(s)
    cfg.val_dataset_config.update({
        "vis_indices": args.vis_index,
        "num_samples": 0})

    _, val_dataset_loader = get_dataloader(
        cfg.train_dataset_config,
        cfg.val_dataset_config,
        cfg.train_loader,
        cfg.val_loader,
        dist=False,
        val_only=True)

    assert args.resume_from and osp.exists(args.resume_from), \
        f'checkpoint not found: {args.resume_from}'
    ckpt = torch.load(args.resume_from, map_location='cpu')
    raw_model.load_state_dict(ckpt['state_dict'], strict=True)
    logger.info(f'resumed from {args.resume_from}')

    my_model.eval()
    os.environ['eval'] = 'true'

    with torch.no_grad():
        for i_iter_val, data in enumerate(val_dataset_loader):
            for k in list(data.keys()):
                if isinstance(data[k], torch.Tensor):
                    data[k] = data[k].cuda()
            input_imgs = data.pop('img')
            result_dict = my_model(imgs=input_imgs, metas=data)

            pred = result_dict['pred_occ'][-1][0]          # (C, N)
            pred_occ = pred.argmax(0).cpu().numpy().astype(np.int16)   # (N,)
            gt_occ = result_dict['sampled_label'][0].cpu().numpy().astype(np.int16)  # (N,)
            xyz = result_dict['sampled_xyz'][0].cpu().numpy().astype(np.float32)     # (N, 3)

            os.makedirs(osp.dirname(args.out), exist_ok=True)
            np.savez_compressed(args.out, xyz=xyz, pred=pred_occ, gt=gt_occ)
            logger.info(f'saved {args.out}  N={xyz.shape[0]}  '
                        f'pred_nonempty={(pred_occ != 17).sum()}  '
                        f'gt_nonempty={(gt_occ != 17).sum()}')
            print(f'saved {args.out} N={xyz.shape[0]} '
                  f'pred_nonempty={(pred_occ != 17).sum()} '
                  f'gt_nonempty={(gt_occ != 17).sum()}')
            break  # only the first (and only) frame


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--py-config', required=True)
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--resume-from', required=True)
    parser.add_argument('--vis-index', type=int, nargs='+', default=[0])
    parser.add_argument('--out', required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(args)
