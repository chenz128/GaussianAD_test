import datetime
import time, argparse, os.path as osp, os
import torch, numpy as np
import torch.distributed as dist
from copy import deepcopy

import mmcv
from mmengine import Config
from mmengine.runner import set_random_seed
from mmengine.optim import build_optim_wrapper
from mmengine.logging import MMLogger
from mmengine.utils import symlink
from mmseg.models import build_segmentor
from timm.scheduler import CosineLRScheduler, MultiStepLRScheduler

import warnings
warnings.filterwarnings("ignore")

try:
    import gpu_affinity
except ImportError as e:
    raise ImportError(
        "An error occurred while trying to import : gpu_affinity, "
        + "install gpu_affinity by 'pip install git+https://github.com/NVIDIA/gpu_affinity' please"
    )

from vis import vis_map_train

def pass_print(*args, **kwargs):
    pass

def initialize(args):#多卡训练时的初始化，返回rank和local_rank
    rank = int(os.environ.get("RANK", 0))#这里获取全局的rank，默认为0，如果是多卡训练，每个进程的rank会不同
    local_rank = int(os.environ.get("LOCAL_RANK", 0))#这里获取当前节点上的GPU编号，默认为0，如果是多卡训练，每个进程的local_rank会不同
    node_num = os.environ.get("NODE_NUM", None)#这里获取节点数量，默认为None，如果是多卡训练，节点数量会大于1
    gpu_count = os.environ.get("GPU_COUNT", None)#这里获取每个节点上的GPU数量，默认为None，如果是多卡训练，每个节点上的GPU数量会大于1
    world_size = int(os.environ.get("WORLD_SIZE", None))#这里获取全局的进程数量，默认为None，如果是多卡训练，全局的进程数量会大于1
    if node_num is not None and gpu_count is not None:
        assert int(node_num) * int(gpu_count) == world_size
        print(f"node_num={node_num}, gpu_count={gpu_count}", flush=True)
    print(f"rank={rank}, local_rank={local_rank}, world_size={world_size}", flush=True)

    device_count = torch.cuda.device_count()
    if dist.is_available():
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            if local_rank == 0:
                print(
                    "lpai torch distributed is already initialized, "
                    "skipping initialization ...",
                    flush=True,
                )
        else:
            # Manually set the device ids.
            if device_count > 0:
                device = rank % device_count
                assert device == local_rank
                # if args.local_rank is not None:
                #    assert args.local_rank == device, \
                #        'expected local-rank to be the same as rank % device-count.'
                # else:
                #    args.local_rank = device
                torch.cuda.set_device(device)
            dist.init_process_group(
                backend=args.backend,
                world_size=world_size,
                rank=rank,
                timeout=datetime.timedelta(hours=1),
            )
    affinity = gpu_affinity.set_affinity(local_rank, device_count)
    print(f"rank={rank}, local_rank={local_rank}, world_size={world_size}", flush=True)

    return rank, local_rank #这个函数返回的是rank和local_rank，rank是全局的进程编号，local_rank是当前节点上的GPU编号

def main(args):
    # global settings
    set_random_seed(args.seed) #设置随机种子，保证实验的可复现性
    torch.backends.cudnn.deterministic = False 
    torch.backends.cudnn.benchmark = True

    # load config#从配置文件中加载训练配置，配置文件中包含了模型结构、优化器设置、数据加载等信息
    cfg = Config.fromfile(args.py_config)
    cfg.work_dir = args.work_dir

    # init DDP
    if args.gpus > 1:
        distributed = True
        rank, local_rank = initialize(args)
    else:
        distributed = False
        world_size = 1
        local_rank = 0

    if local_rank == 0:
        os.makedirs(args.work_dir, exist_ok=True)
        cfg.dump(osp.join(args.work_dir, osp.basename(args.py_config)))
        from misc.tb_wrapper import WrappedTBWriter
        writer = WrappedTBWriter('selfocc', log_dir=osp.join(args.work_dir, 'tf'))
        WrappedTBWriter._instance_dict['selfocc'] = writer
    else:
        writer = None
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(args.work_dir, f'{timestamp}.log')
    logger = MMLogger('selfocc', log_file=log_file)
    MMLogger._instance_dict['selfocc'] = logger
    # logger.info(f'Config:\n{cfg.pretty_text}')

    # build model
    import model
    from dataset import get_dataloader
    from loss import OPENOCC_LOSS

    my_model = build_segmentor(cfg.model)
    my_model.init_weights()#初始化模型权重，这个函数会根据模型的定义来初始化权重，通常会使用一些常见的初始化方法，比如Xavier初始化或者Kaiming初始化等
    # Freeze modules not contributing to the active losses (e.g. map-only stage)
    for mod_name in cfg.get('frozen_modules', []):#冻结模型中不参与当前损失计算的模块，这些模块在训练过程中不会更新权重，通常用于多阶段训练或者迁移学习等场景
        mod = getattr(my_model, mod_name, None)
        if mod is not None:
            for param in mod.parameters():
                param.requires_grad = False
            logger.info(f'Frozen module: {mod_name}')
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
        iter_resume=args.iter_resume)

    # get optimizer, loss, scheduler
    optimizer = build_optim_wrapper(my_model, cfg.optimizer)
    loss_func = OPENOCC_LOSS.build(cfg.loss).cuda()
    max_num_epochs = cfg.max_epochs
    if cfg.get('multisteplr', False):
        scheduler = MultiStepLRScheduler(
            optimizer,
            **cfg.multisteplr_config
        )
    else:
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=len(train_dataset_loader) * max_num_epochs,
            lr_min=cfg.optimizer["optimizer"]["lr"] * 0.1, #1e-6,
            warmup_t=cfg.get('warmup_iters', 500),
            warmup_lr_init=1e-6,
            t_in_epochs=False)
    amp = cfg.get('amp', False)
    backbone_fp16 = cfg.get('backbone_fp16', False)
    # bf16 doesn't need GradScaler (same dynamic range as fp32)
    use_scaler = amp  # only full-model fp16 needs scaler
    if use_scaler:
        scaler = torch.cuda.amp.GradScaler()
    os.environ['amp'] = 'true' if amp else 'false'

    # resume and load
    epoch = 0
    global_iter = 0
    last_iter = 0

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
        print(raw_model.load_state_dict(ckpt['state_dict'], strict=False))
        try:
            optimizer.load_state_dict(ckpt['optimizer'])
        except (ValueError, KeyError) as e:
            logger.info(f'Optimizer state mismatch (frozen modules changed?), skipping optimizer resume: {e}')
        try:
            scheduler.load_state_dict(ckpt['scheduler'])
        except (ValueError, KeyError) as e:
            logger.info(f'Scheduler state mismatch, skipping scheduler resume: {e}')
        epoch = ckpt['epoch']
        global_iter = ckpt['global_iter']
        last_iter = ckpt['last_iter'] if 'last_iter' in ckpt else 0
        if hasattr(train_dataset_loader.sampler, 'set_last_iter'):
            train_dataset_loader.sampler.set_last_iter(last_iter)
        print(f'successfully resumed from epoch {epoch}')
    elif cfg.load_from:
        ckpt = torch.load(cfg.load_from, map_location='cpu')
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        try:
            print(raw_model.load_state_dict(state_dict, strict=False))
        except:
            from misc.checkpoint_util import refine_load_from_sd
            print(raw_model.load_state_dict(
                refine_load_from_sd(state_dict), strict=False))

    # training
    print_freq = cfg.print_freq
    first_run = True
    grad_accumulation = args.gradient_accumulation
    grad_norm = 0
    from misc.metric_util import MeanIoU
    miou_metric = MeanIoU(
        list(range(1, 17)),
        17, #17,
        ['barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
         'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
         'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
         'vegetation'],
         True, 17, filter_minmax=False)#这里的MeanIoU类是一个计算平均交并比（mIoU）的工具类，用于评估模型在语义分割任务中的性能。它接受以下参数：
    miou_metric.reset()

    while epoch < max_num_epochs:
        my_model.train()
        os.environ['eval'] = 'false'#这里设置了一个环境变量eval为false，表示当前处于训练阶段。在训练过程中，模型会根据输入数据进行前向传播和反向传播来更新权重，而在评估阶段，模型会根据输入数据进行前向传播来计算损失和评估指标。通过设置这个环境变量，可以在模型的前向传播过程中区分训练和评估阶段，从而执行不同的操作，比如是否计算损失、是否更新权重等。
        if hasattr(train_dataset_loader.sampler, 'set_epoch'):#这里检查了train_dataset_loader的采样器是否具有set_epoch方法，如果有的话，就调用这个方法来设置当前的epoch。这通常用于分布式训练中的数据采样器，以确保每个epoch的数据顺序不同，从而提高模型的泛化能力。
            train_dataset_loader.sampler.set_epoch(epoch)
        loss_list = []
        time.sleep(10)
        data_time_s = time.time()
        time_s = time.time()
        for i_iter, data in enumerate(train_dataset_loader):
            if first_run:
                i_iter = i_iter + last_iter

            for k in list(data.keys()):#这里遍历了数据字典中的所有键，并检查对应的值是否是一个PyTorch张量。如果是的话，就将这个张量移动到GPU上进行计算。这是为了确保在训练过程中，所有的输入数据都在GPU上，以加快计算速度。
                if isinstance(data[k], torch.Tensor):
                    data[k] = data[k].cuda()
            input_imgs = data.pop('img')
            data_time_e = time.time()

            with torch.cuda.amp.autocast(amp):
                # forward + backward + optimize
                result_dict = my_model(imgs=input_imgs, metas=data, global_iter=global_iter)#前向传播

                loss_input = {
                    'metas': data
                }#这个字典用来存储计算损失所需要的输入数据，初始时包含了一个键'metas'，对应的值是从数据字典中弹出的数据。这个数据通常包含了与输入图像相关的元信息，比如标签、坐标等。在后续的代码中，会根据配置文件中的loss_input_convertion来更新这个字典，添加更多的键值对，以满足损失函数的输入需求。
                for loss_input_key, loss_input_val in cfg.loss_input_convertion.items():
                    if loss_input_key not in result_dict:
                        loss_input.update({
                            loss_input_key: result_dict['metas'].get(loss_input_val)})
                    else:
                        loss_input.update({
                            loss_input_key: result_dict[loss_input_val]})
                loss, loss_dict = loss_func(loss_input)

                loss = loss / grad_accumulation

                if args.vis_map:
                    vis_map_train(result_dict, data)
            if not use_scaler:
                loss.backward()
                if (global_iter + 1) % grad_accumulation == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(my_model.parameters(), cfg.grad_max_norm)
                    optimizer.step()
                    optimizer.zero_grad()
            else:
                scaler.scale(loss).backward()
                if (global_iter + 1) % grad_accumulation == 0:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(my_model.parameters(), cfg.grad_max_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

            loss_list.append(loss.detach().cpu().item())
            scheduler.step_update(global_iter)
            time_e = time.time()

            global_iter += 1
            if i_iter % print_freq == 0 and local_rank == 0:
                lr = optimizer.param_groups[0]['lr']
                logger.info('[TRAIN] Epoch %d Iter %5d/%d: Loss: %.3f (%.3f), grad_norm: %.3f, lr: %.7f, time: %.3f (%.3f)'%(
                    epoch, i_iter, len(train_dataset_loader),
                    loss.item(), np.mean(loss_list), grad_norm, lr,
                    time_e - time_s, data_time_e - data_time_s))
                detailed_loss = []
                for loss_name, loss_value in loss_dict.items():
                    detailed_loss.append(f'{loss_name}: {loss_value:.5f}')
                detailed_loss = ', '.join(detailed_loss)
                logger.info(detailed_loss)
                loss_list = []
            data_time_s = time.time()
            time_s = time.time()

            if args.iter_resume:
                if (i_iter + 1) % 50 == 0 and local_rank == 0:
                    dict_to_save = {
                        'state_dict': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch,
                        'global_iter': global_iter,
                        'last_iter': i_iter + 1,
                    }
                    save_file_name = os.path.join(os.path.abspath(args.work_dir), 'iter.pth')
                    torch.save(dict_to_save, save_file_name)
                    dst_file = osp.join(args.work_dir, 'latest.pth')
                    symlink(save_file_name, dst_file)
                    logger.info(f'iter ckpt {i_iter + 1} saved!')

        # save checkpoint
        if local_rank == 0:
            dict_to_save = {
                'state_dict': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch + 1,
                'global_iter': global_iter,
            }
            save_file_name = os.path.join(os.path.abspath(args.work_dir), f'epoch_{epoch+1}.pth')
            torch.save(dict_to_save, save_file_name)
            dst_file = osp.join(args.work_dir, 'latest.pth')
            symlink(save_file_name, dst_file)

        epoch += 1
        first_run = False

        # eval
        if epoch % cfg.get('eval_every_epochs', 1) != 0:
            continue
        my_model.eval()
        os.environ['eval'] = 'true'
        val_loss_list = []

        with torch.no_grad():
            for i_iter_val, data in enumerate(val_dataset_loader):
                for k in list(data.keys()):
                    if isinstance(data[k], torch.Tensor):
                        data[k] = data[k].cuda()
                input_imgs = data.pop('img')

                with torch.cuda.amp.autocast(amp):
                    result_dict = my_model(imgs=input_imgs, metas=data)

                    loss_input = {
                        'metas': data,
                    }
                    for loss_input_key, loss_input_val in cfg.loss_input_convertion.items():
                        if loss_input_key not in result_dict:
                            loss_input.update({
                                loss_input_key: result_dict['metas'].get(loss_input_val)})
                        else:
                            loss_input.update({
                                loss_input_key: result_dict[loss_input_val]})
                    loss, loss_dict = loss_func(loss_input)

                for idx, pred in enumerate(result_dict['pred_occ'][-1]):
                    pred_occ = pred.argmax(0)
                    gt_occ = result_dict['sampled_label'][idx]
                    miou_metric._after_step(pred_occ, gt_occ)

                val_loss_list.append(loss.detach().cpu().numpy())
                if i_iter_val % print_freq == 0 and local_rank == 0:
                    logger.info('[EVAL] Epoch %d Iter %5d: Loss: %.3f (%.3f)'%(
                        epoch, i_iter_val, loss.item(), np.mean(val_loss_list)))
                    detailed_loss = []
                    for loss_name, loss_value in loss_dict.items():
                        detailed_loss.append(f'{loss_name}: {loss_value:.5f}')
                    detailed_loss = ', '.join(detailed_loss)
                    logger.info(detailed_loss)

        miou, iou2 = miou_metric._after_epoch()
        logger.info(f'mIoU: {miou}, iou2: {iou2}')
        logger.info('Current val loss is %.3f' % (np.mean(val_loss_list)))
        miou_metric.reset()

    if writer is not None:
        writer.close()


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/tpv_lidarseg.py')
    parser.add_argument('--work-dir', type=str, default='./out/tpv_lidarseg')
    parser.add_argument('--resume-from', type=str, default='')
    parser.add_argument('--iter-resume', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gradient-accumulation', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='nuscenes')
    parser.add_argument(
        "--backend",
        type=str,
        help="Distributed backend",
        choices=[dist.Backend.GLOO, dist.Backend.NCCL, dist.Backend.MPI],
        default=dist.Backend.NCCL,
    )
    parser.add_argument('--vis-map', action='store_true', default=False)

    args = parser.parse_args()
    ngpus = torch.cuda.device_count()
    args.gpus = ngpus
    print(args)

    main(args)
