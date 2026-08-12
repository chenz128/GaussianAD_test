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
                    sampled_xyz = result_dict['sampled_xyz'][idx]
                    occ_shape = tuple(
                        int(torch.unique(sampled_xyz[:, axis]).numel())
                        for axis in range(3))
                    if np.prod(occ_shape) != pred_occ.numel():
                        raise ValueError(
                            f'cannot reshape {pred_occ.numel()} occupancy values '
                            f'to inferred grid {occ_shape}')
                    save_occ(
                        os.path.join(args.work_dir, 'vis'),
                        pred_occ.reshape(1, *occ_shape),
                        f'val_{i_iter_val}_pred',
                        True, 0)
                    save_occ(
                        os.path.join(args.work_dir, 'vis'),
                        gt_occ.reshape(1, *occ_shape),
                        f'val_{i_iter_val}_gt',
                        True, 0)
                if args.vis_gaussian and local_rank == 0:
                    save_gaussian(
                        os.path.join(args.work_dir, 'vis'),
                        result_dict['gaussian'],
                        f'val_{i_iter_val}_gaussian')
                    # GaussianHeadFrontier recycles only slots that leave the
                    # occupancy window. Save the actual per-step replacement
                    # result so its visualization does not approximate future
                    # frames as merely current means plus the offset.
                    try:
                        head = raw_model.head
                        if hasattr(head, 'future_generator'):
                            from model.utils.utils import get_rotation_matrix

                            gaussian = result_dict['gaussian']
                            offset = result_dict['offset'].reshape(
                                1, -1, 6, 2)
                            offset = torch.cat(
                                [offset, offset.new_zeros(*offset.shape[:-1], 1)],
                                dim=-1)
                            provided_transforms = result_dict.get(
                                'future_lidar_transforms')
                            if (getattr(head, 'future_pose_mode', 'translation')
                                    == 'se3' or provided_transforms is not None):
                                future_transforms = head.get_future_lidar_transforms(
                                    data, gaussian.means,
                                    provided_transforms=provided_transforms)
                            else:
                                ego = data['ego_fut_trajs']
                                if not torch.is_tensor(ego):
                                    ego = torch.as_tensor(
                                        ego, device=offset.device)
                                ego = torch.nan_to_num(
                                    ego.to(offset).float()).cumsum(dim=1)
                                future_transforms = torch.eye(
                                    4, device=offset.device, dtype=offset.dtype
                                ).reshape(1, 1, 4, 4).repeat(
                                    gaussian.means.shape[0], 6, 1, 1)
                                future_transforms[..., :2, 3] = -ego
                            future_to_current = torch.linalg.inv(
                                future_transforms)
                            future_origins = future_to_current[..., :3, 3]

                            generated = getattr(
                                head.future_generator, 'last_generated', None)
                            if generated is None:
                                generated = head.future_generator(
                                    ego_cumulative=future_origins,
                                    temporal_features=result_dict[
                                        'temporal_context_features'],
                                    temporal_indices=result_dict[
                                        'temporal_context_indices'],
                                    ms_img_feats=result_dict['ms_img_feats'],
                                    metas=data,
                                    batch_size=gaussian.means.shape[0],
                                    future_to_current_rotations=(
                                        future_to_current[..., :3, :3]))
                            generated_semantics = generated['semantics']
                            if generated_semantics.shape[-1] < gaussian.semantics.shape[-1]:
                                generated_semantics = torch.nn.functional.pad(
                                    generated_semantics,
                                    (0, gaussian.semantics.shape[-1]
                                     - generated_semantics.shape[-1]))

                            future = [dict(
                                means=gaussian.means[0],
                                scales=gaussian.scales[0],
                                rotations=gaussian.rotations[0],
                                rotation_matrices=get_rotation_matrix(
                                    gaussian.rotations)[0].transpose(-1, -2),
                                opacities=gaussian.opacities[0, :, 0],
                                semantics=gaussian.semantics[0],
                                generated=torch.zeros(
                                    gaussian.means.shape[1], dtype=torch.bool,
                                    device=offset.device))]
                            means_future = gaussian.means[..., None, :] + offset
                            for step in range(6):
                                transform = future_transforms[:, step]
                                rotation = transform[:, None, :3, :3]
                                warped_old = head.transform_points(
                                    means_future[..., step, :], transform)
                                old_inside = head.get_in_range_mask(warped_old)[0]
                                new_means = head.transform_points(
                                    generated['means'], transform)
                                new_active = (
                                    head.get_in_range_mask(new_means)[0]
                                    & (generated['enter_time'][0]
                                       <= ((step + 1) / 6.0)))
                                old_count = int(old_inside.sum())
                                new_count = int(new_active.sum())
                                future.append(dict(
                                    means=torch.cat([
                                        warped_old[0, old_inside],
                                        new_means[0, new_active]], 0),
                                    scales=torch.cat([
                                        gaussian.scales[0, old_inside],
                                        generated['scales'][0, new_active]], 0),
                                    rotations=torch.cat([
                                        gaussian.rotations[0, old_inside],
                                        generated['rotations'][0, new_active]], 0),
                                    rotation_matrices=torch.cat([
                                        (rotation @ get_rotation_matrix(
                                            gaussian.rotations).transpose(
                                                -1, -2))[0, old_inside],
                                        (rotation @ get_rotation_matrix(
                                            generated['rotations']).transpose(
                                                -1, -2))[0, new_active]], 0),
                                    opacities=torch.cat([
                                        gaussian.opacities[0, old_inside, 0],
                                        generated['opacities'][0, new_active, 0]], 0),
                                    semantics=torch.cat([
                                        gaussian.semantics[0, old_inside],
                                        generated_semantics[0, new_active]], 0),
                                    generated=torch.cat([
                                        torch.zeros(
                                            old_count, dtype=torch.bool,
                                            device=offset.device),
                                        torch.ones(
                                            new_count, dtype=torch.bool,
                                            device=offset.device)], 0)))

                            max_count = max(item['means'].shape[0] for item in future)
                            valid = []
                            for item in future:
                                count = item['means'].shape[0]
                                padding = max_count - count
                                valid.append(torch.cat([
                                    torch.ones(count, dtype=torch.bool, device=offset.device),
                                    torch.zeros(padding, dtype=torch.bool, device=offset.device)]))
                                for key in ('means', 'scales', 'rotations',
                                            'rotation_matrices', 'opacities',
                                            'semantics', 'generated'):
                                    value = item[key]
                                    pad_shape = (padding, *value.shape[1:])
                                    item[key] = torch.cat([
                                        value, value.new_zeros(pad_shape)], 0)
                            np.savez_compressed(
                                os.path.join(
                                    args.work_dir, 'vis',
                                    f'val_{i_iter_val}_frontier_future.npz'),
                                means=np.stack([
                                    item['means'].detach().cpu().numpy()
                                    for item in future]),
                                scales=np.stack([
                                    item['scales'].detach().cpu().numpy()
                                    for item in future]),
                                rotations=np.stack([
                                    item['rotations'].detach().cpu().numpy()
                                    for item in future]),
                                rotation_matrices=np.stack([
                                    item['rotation_matrices'].detach().cpu().numpy()
                                    for item in future]),
                                opacities=np.stack([
                                    item['opacities'].detach().cpu().numpy()
                                    for item in future]),
                                semantics=np.stack([
                                    item['semantics'].detach().cpu().numpy()
                                    for item in future]),
                                generated=np.stack([
                                    item['generated'].detach().cpu().numpy()
                                    for item in future]),
                                valid=np.stack([
                                    item.detach().cpu().numpy()
                                    for item in valid]))
                        elif hasattr(head, 'frontier_generator'):
                            gaussian = result_dict['gaussian']
                            offset = result_dict['offset'].reshape(
                                1, -1, 6, 2)
                            offset = torch.cat(
                                [offset, offset.new_zeros(*offset.shape[:-1], 1)],
                                dim=-1)
                            ego = data['ego_fut_trajs']
                            if not torch.is_tensor(ego):
                                ego = torch.as_tensor(ego, device=offset.device)
                            ego = ego.to(offset.device).float()
                            if ego.dim() == 2:
                                ego = ego[None]
                            ego = torch.nan_to_num(ego).cumsum(dim=1)
                            ego = torch.cat(
                                [ego, ego.new_zeros(*ego.shape[:-1], 1)], dim=-1)

                            means_fut = gaussian.means[..., None, :] + offset
                            num_real = gaussian.means.shape[1]
                            if hasattr(head, 'target_num_gaussians'):
                                target_num = head.target_num_gaussians
                                missing = target_num - num_real
                                if missing < 0:
                                    raise AssertionError(
                                        f'current Gaussian count {num_real} exceeds '
                                        f'visualization target {target_num}')
                                # The current frame has fewer real Gaussians than
                                # v2's fixed future slots. Invisible zero-opacity
                                # padding keeps the animation tensor rectangular.
                                future = [dict(
                                    means=torch.cat([
                                        gaussian.means[0],
                                        gaussian.means.new_zeros(missing, 3)], 0),
                                    scales=torch.cat([
                                        gaussian.scales[0],
                                        gaussian.scales.new_zeros(missing, 3)], 0),
                                    rotations=torch.cat([
                                        gaussian.rotations[0],
                                        gaussian.rotations.new_zeros(missing, 4)], 0),
                                    opacities=torch.cat([
                                        gaussian.opacities[0, :, 0],
                                        gaussian.opacities.new_zeros(missing)], 0),
                                    semantics=torch.cat([
                                        gaussian.semantics[0],
                                        gaussian.semantics.new_zeros(
                                            missing, gaussian.semantics.shape[-1])], 0),
                                    recycled=torch.zeros(
                                        target_num, dtype=torch.bool,
                                        device=offset.device))]
                                image_map, projection, image_wh = (
                                    head._current_camera_inputs(
                                        result_dict['ms_img_feats'], data,
                                        gaussian.means.shape[0]))
                                image_features = (
                                    head.frontier_generator.prepare_image_features(
                                        image_map))
                                context = dict(
                                    means=gaussian.means,
                                    scales=gaussian.scales,
                                    rotations=gaussian.rotations,
                                    opacities=gaussian.opacities,
                                    semantics=gaussian.semantics)
                                for step in range(6):
                                    means = (
                                        means_fut[..., step, :]
                                        - ego[:, step:step + 1])
                                    inside = head.get_in_range_mask(means)
                                    generated = head.frontier_generator(
                                        ego_disp=ego[:, step],
                                        num_gaussians=target_num,
                                        time_index=step,
                                        context_gaussian=context,
                                        context_valid=inside,
                                        image_features=image_features,
                                        projection_mat=projection,
                                        image_wh=image_wh)
                                    keep = inside[..., None]
                                    future.append(dict(
                                        means=torch.cat([
                                            torch.where(
                                                keep, means,
                                                generated['means'][:, :num_real]),
                                            generated['means'][:, num_real:]], 1)[0],
                                        scales=torch.cat([
                                            torch.where(
                                                keep, gaussian.scales,
                                                generated['scales'][:, :num_real]),
                                            generated['scales'][:, num_real:]], 1)[0],
                                        rotations=torch.cat([
                                            torch.where(
                                                keep, gaussian.rotations,
                                                generated['rotations'][:, :num_real]),
                                            generated['rotations'][:, num_real:]], 1)[0],
                                        opacities=torch.cat([
                                            torch.where(
                                                keep, gaussian.opacities,
                                                generated['opacities'][:, :num_real]),
                                            generated['opacities'][:, num_real:]], 1)[0, :, 0],
                                        semantics=torch.cat([
                                            torch.where(
                                                keep, gaussian.semantics,
                                                generated['semantics'][:, :num_real]),
                                            generated['semantics'][:, num_real:]], 1)[0],
                                        recycled=torch.cat([
                                            (~inside)[0],
                                            torch.ones(
                                                missing, dtype=torch.bool,
                                                device=offset.device)], 0)))
                            else:
                                future = [dict(
                                    means=gaussian.means[0],
                                    scales=gaussian.scales[0],
                                    rotations=gaussian.rotations[0],
                                    opacities=gaussian.opacities[0, :, 0],
                                    semantics=gaussian.semantics[0],
                                    recycled=torch.zeros(
                                        gaussian.means.shape[1], dtype=torch.bool,
                                        device=offset.device))]
                                for step in range(6):
                                    means = means_fut[..., step, :] - ego[:, step:step + 1]
                                    inside = head.get_in_range_mask(means)
                                    generated = head.frontier_generator(
                                        ego_disp=ego[:, step],
                                        num_gaussians=num_real,
                                        time_index=step)
                                    keep = inside[..., None]
                                    future.append(dict(
                                        means=torch.where(keep, means, generated['means'])[0],
                                        scales=torch.where(
                                            keep, gaussian.scales, generated['scales'])[0],
                                        rotations=torch.where(
                                            keep, gaussian.rotations, generated['rotations'])[0],
                                        opacities=torch.where(
                                            keep, gaussian.opacities, generated['opacities'])[0, :, 0],
                                        semantics=torch.where(
                                            keep, gaussian.semantics, generated['semantics'])[0],
                                        recycled=(~inside)[0]))
                            np.savez_compressed(
                                os.path.join(
                                    args.work_dir, 'vis',
                                    f'val_{i_iter_val}_frontier_future.npz'),
                                means=np.stack([
                                    item['means'].detach().cpu().numpy()
                                    for item in future]),
                                scales=np.stack([
                                    item['scales'].detach().cpu().numpy()
                                    for item in future]),
                                rotations=np.stack([
                                    item['rotations'].detach().cpu().numpy()
                                    for item in future]),
                                opacities=np.stack([
                                    item['opacities'].detach().cpu().numpy()
                                    for item in future]),
                                semantics=np.stack([
                                    item['semantics'].detach().cpu().numpy()
                                    for item in future]),
                                recycled=np.stack([
                                    item['recycled'].detach().cpu().numpy()
                                    for item in future]))
                    except Exception as e:
                        print(f'[frontier future dump] failed on val_{i_iter_val}: {e}')
                        logger.info(
                            f'[frontier future dump] failed on val_{i_iter_val}: {e}')
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
