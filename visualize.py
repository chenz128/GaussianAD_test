try:
    from vis import save_occ, save_gaussian
except:
    print('Load Occupancy Visualization Tools Failed.')
import time, argparse, os.path as osp, os
import torch, numpy as np
import torch.distributed as dist

from mmengine import Config
from mmengine.runner import set_random_seed
from mmengine.logging import MMLogger
from mmseg.models import build_segmentor

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
    log_file = osp.join(args.work_dir, f'{timestamp}.log')
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

    cfg.val_dataset_config.update({
        "vis_indices": args.vis_index,
        "num_samples": args.num_samples})

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
        print(raw_model.load_state_dict(state_dict, strict=False))
        
    print_freq = cfg.print_freq
    from misc.metric_util import MeanIoU
    miou_metric = MeanIoU(
        list(range(1, 17)),
        17, #17,
        ['barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
         'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
         'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
         'vegetation'],
         True, 17, filter_minmax=False)
    miou_metric.reset()

    my_model.eval()
    os.environ['eval'] = 'true'
    if args.vis_occ or args.vis_gaussian:
        os.makedirs(os.path.join(args.work_dir, 'vis'), exist_ok=True)

    # Exact dynamic/static probe: reuse the same GT-box membership and optional
    # GT semantic gate as DynamicLoss, then collect raw logits for threshold-
    # independent PR/F1 analysis. This is deliberately evaluation-only.
    dyn_probe = None
    dyn_logits_all, dyn_target_all = [], []
    if args.eval_dynamic:
        from loss.dynamic_loss import DynamicLoss
        dyn_cfg = next(
            (item.copy() for item in cfg.loss.loss_cfgs
             if item.get('type') == 'DynamicLoss'),
            None,
        )
        if dyn_cfg is None or not dyn_cfg.get('use_gt_box', False):
            raise ValueError('--eval-dynamic requires use_gt_box=True DynamicLoss')
        dyn_cfg.pop('type')
        dyn_probe = DynamicLoss(**dyn_cfg).cuda().eval()

    with torch.no_grad():
        for i_iter_val, data in enumerate(val_dataset_loader):
            
            for k in list(data.keys()):
                if isinstance(data[k], torch.Tensor):
                    data[k] = data[k].cuda()
            input_imgs = data.pop('img')
            result_dict = my_model(imgs=input_imgs, metas=data)

            if dyn_probe is not None:
                gaussian = result_dict.get('gaussian')
                logits = getattr(gaussian, 'dynamic_logits', None)
                if logits is not None:
                    sampled_xyz = result_dict.get('sampled_xyz', data.get('sampled_xyz'))
                    sampled_label = result_dict.get('sampled_label', data.get('sampled_label'))
                    target = dyn_probe._gt_box_membership(
                        gaussian.means, data['gt_boxes'], sampled_xyz, sampled_label)
                    dyn_logits_all.append(logits[..., 0].detach().flatten().cpu())
                    dyn_target_all.append(target.detach().flatten().cpu())

            for idx, pred in enumerate(result_dict['pred_occ'][-1]):
                pred_occ = pred.argmax(0)
                gt_occ = result_dict['sampled_label'][idx]
                if args.vis_occ:
                    save_occ(
                        os.path.join(args.work_dir, 'vis'),
                        pred_occ.reshape(1, 200, 200, 16),
                        f'val_{i_iter_val}_pred',
                        True, 0)
                    save_occ(
                        os.path.join(args.work_dir, 'vis'),
                        gt_occ.reshape(1, 200, 200, 16),
                        f'val_{i_iter_val}_gt',
                        True, 0)
                if args.vis_gaussian and local_rank == 0:
                    save_gaussian(
                        os.path.join(args.work_dir, 'vis'),
                        result_dict['gaussian'],
                        f'val_{i_iter_val}_gaussian')
                    # dump future-frame offset (+ego motion) for animation
                    try:
                        off_t = result_dict.get('offset')
                        if off_t is not None:
                            off = off_t.reshape(1, -1, 6, 2)[0].detach().cpu().numpy()  # (A,6,2)
                            planner = np.zeros((6, 2), dtype=np.float32)
                            if 'ego_fut_preds' in result_dict and 'ego_fut_cmd' in data:
                                cmd = data['ego_fut_cmd'].argmax(dim=-1)
                                pr = result_dict['ego_fut_preds'].cumsum(dim=1)[0, cmd, ...]
                                planner = pr.detach().cpu().numpy().reshape(-1, 2)[:6]  # (6,2)
                            # GT ego trajectory (cumulative per-step displacement, LIDAR frame)
                            gt_ego = np.zeros((6, 2), dtype=np.float32)
                            if 'ego_fut_trajs' in data:
                                egt = data['ego_fut_trajs']
                                if torch.is_tensor(egt):
                                    egt = egt[0].float().cpu().numpy()  # (6,2)
                                else:
                                    egt = np.asarray(egt)[0]
                                egt = np.nan_to_num(egt, nan=0.0).astype(np.float32)[:6, :2]
                                gt_ego = np.cumsum(egt, axis=0)  # cumulative (6,2)
                            g = result_dict['gaussian']
                            pred_cls = g.semantics[0].detach().cpu().numpy().argmax(-1).astype(np.int16)
                            np.savez(
                                os.path.join(args.work_dir, 'vis', f'val_{i_iter_val}_future.npz'),
                                offset=off.astype(np.float32),
                                planner=planner.astype(np.float32),
                                gt_ego=gt_ego.astype(np.float32),
                                pred_cls=pred_cls)
                    except Exception as e:
                        logger.info(f'[future dump] failed on val_{i_iter_val}: {e}')
                    # dump occ_flow 6 future frames (the model's real future occ) for animation
                    try:
                        occ_flow = result_dict.get('occ_flow')
                        if occ_flow is not None:
                            sxyz = result_dict['sampled_xyz'][0].detach().cpu().numpy().astype(np.float32)  # (N,3)
                            occ_now = result_dict['pred_occ'][-1][0].argmax(0).detach().cpu().numpy().astype(np.int16)  # (N,)
                            N = occ_now.shape[0]
                            nfut = len(occ_flow)
                            occ_fut = np.zeros((nfut, N), dtype=np.int16)
                            occ_fut_gt = np.full((nfut, N), -1, dtype=np.int16)
                            valid = np.zeros((nfut,), dtype=np.int8)
                            for fi in range(nfut):
                                fdict = occ_flow[fi][0]
                                valid[fi] = int(bool(fdict['flow_valid_flag']))
                                pf = fdict['pred_flow'][0].argmax(0).detach().cpu().numpy().astype(np.int16)
                                occ_fut[fi, :pf.shape[0]] = pf
                                gtf = fdict['sampled_label']
                                if isinstance(gtf, torch.Tensor):
                                    gtf = gtf.detach().cpu().numpy()
                                gtf = np.asarray(gtf).reshape(-1).astype(np.int16)
                                occ_fut_gt[fi, :gtf.shape[0]] = gtf
                            np.savez(
                                os.path.join(args.work_dir, 'vis', f'val_{i_iter_val}_occflow.npz'),
                                xyz=sxyz, occ_now=occ_now,
                                occ_fut=occ_fut, occ_fut_gt=occ_fut_gt, valid=valid)
                    except Exception as e:
                        logger.info(f'[occflow dump] failed on val_{i_iter_val}: {e}')
                miou_metric._after_step(pred_occ, gt_occ)
            
            if i_iter_val % print_freq == 0 and local_rank == 0:
                logger.info('[EVAL] Iter %5d'%(i_iter_val))
                    
    miou, iou2 = miou_metric._after_epoch()
    logger.info(f'mIoU: {miou}, iou2: {iou2}')
    if dyn_logits_all:
        logits = torch.cat(dyn_logits_all).float()
        target = torch.cat(dyn_target_all).bool()
        logger.info(
            f'[DynamicEval] gaussians={target.numel()} gt_dynamic={target.sum().item()} '
            f'({target.float().mean().item():.3%})')
        candidates = torch.unique(torch.cat([
            torch.tensor([-4., -2., -1., -.5, 0., .5, 1., 2., 4.]), logits
        ])).sort().values
        # A dense percentile grid finds calibration optimum without retaining any
        # gradients or changing the model.
        quantiles = torch.linspace(0., 1., 401)
        candidates = torch.unique(torch.cat([candidates, torch.quantile(logits, quantiles)])).sort().values
        best = None
        for threshold in candidates.tolist():
            pred = logits > threshold
            tp = int((pred & target).sum().item())
            fp = int((pred & ~target).sum().item())
            fn = int((~pred & target).sum().item())
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-12)
            if best is None or f1 > best[-1]:
                best = (threshold, tp, fp, fn, precision, recall, f1)
        for threshold in (0.,):
            pred = logits > threshold
            tp = int((pred & target).sum().item())
            fp = int((pred & ~target).sum().item())
            fn = int((~pred & target).sum().item())
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-12)
            logger.info(
                f'[DynamicEval @logit>0] TP={tp} FP={fp} FN={fn} '
                f'P={precision:.3%} R={recall:.3%} F1={f1:.3%} '
                f'pred_dynamic={pred.float().mean().item():.3%}')
        threshold, tp, fp, fn, precision, recall, f1 = best
        logger.info(
            f'[DynamicEval best-F1] threshold={threshold:.4f} TP={tp} FP={fp} FN={fn} '
            f'P={precision:.3%} R={recall:.3%} F1={f1:.3%}')
    miou_metric.reset()
    
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
    parser.add_argument('--vis-gaussian', action='store_true', default=False)
    parser.add_argument('--vis-index', type=int, nargs='+', default=[])
    parser.add_argument('--num-samples', type=int, default=1)
    parser.add_argument('--eval-dynamic', action='store_true',
                        help='report dynamic_logits precision/recall/F1 against the '
                        'same GT-box target used by DynamicLoss')
    args = parser.parse_args()
    
    ngpus = torch.cuda.device_count()
    args.gpus = ngpus
    print(args)

    if ngpus > 1:
        torch.multiprocessing.spawn(main, args=(args,), nprocs=args.gpus)
    else:
        main(0, args)
