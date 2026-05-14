"""
Memory & Time Profiling Script for GaussianAD (splatting branch).
Runs 3 training iterations on a single GPU and reports per-stage memory/time.

Usage:
    CUDA_VISIBLE_DEVICES=4 /data/chenz/conda_env/splatting/bin/python profile_memory.py \
        --py-config config/nuscenes_gs25600.py
"""
import time
import argparse
import os
import os.path as osp
import torch
import numpy as np

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True


def mem_mb():
    """Current GPU memory allocated in MB."""
    return torch.cuda.memory_allocated() / 1024 / 1024


def peak_mb():
    """Peak GPU memory allocated in MB."""
    return torch.cuda.max_memory_allocated() / 1024 / 1024


def fmt(mb):
    return f"{mb:.0f} MB"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--py-config", required=True)
    parser.add_argument("--num-iters", type=int, default=3)
    args = parser.parse_args()

    from mmengine import Config
    from mmengine.logging import MMLogger
    from mmseg.models import build_segmentor
    from mmengine.optim import build_optim_wrapper

    cfg = Config.fromfile(args.py_config)
    cfg.work_dir = '/tmp/profile_run'
    os.makedirs(cfg.work_dir, exist_ok=True)

    logger = MMLogger('selfocc', log_file=osp.join(cfg.work_dir, 'profile.log'))
    MMLogger._instance_dict['selfocc'] = logger

    import model
    from dataset import get_dataloader
    from loss import OPENOCC_LOSS

    print("=" * 70)
    print("GaussianAD Memory Profiler (splatting branch)")
    print("=" * 70)

    # ── Build model ──
    torch.cuda.reset_peak_memory_stats()
    mem_start = mem_mb()
    print(f"\n[0] Before model build: {fmt(mem_start)}")

    my_model = build_segmentor(cfg.model)
    my_model.init_weights()
    for mod_name in cfg.get('frozen_modules', []):
        mod = getattr(my_model, mod_name, None)
        if mod is not None:
            for param in mod.parameters():
                param.requires_grad = False
    my_model = my_model.cuda()
    torch.cuda.synchronize()

    mem_after_model = mem_mb()
    n_params = sum(p.numel() for p in my_model.parameters())
    n_trainable = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    print(f"[1] After model.cuda(): {fmt(mem_after_model)} (+{fmt(mem_after_model - mem_start)})")
    print(f"    Total params: {n_params/1e6:.1f}M, Trainable: {n_trainable/1e6:.1f}M")

    # ── Build dataloader ──
    train_loader, _ = get_dataloader(
        cfg.train_dataset_config,
        cfg.val_dataset_config,
        cfg.train_loader,
        cfg.val_loader,
        dist=False,
        iter_resume=False)

    # ── Build optimizer & loss ──
    optimizer = build_optim_wrapper(my_model, cfg.optimizer)
    loss_func = OPENOCC_LOSS.build(cfg.loss).cuda()
    amp = cfg.get('amp', False)
    print(f"\n[INFO] AMP enabled: {amp}")
    print(f"[INFO] Batch size: {cfg.train_loader.get('batch_size', 1)}")
    print(f"[INFO] Num workers: {cfg.train_loader.get('num_workers', 0)}")

    mem_after_optim = mem_mb()
    print(f"[2] After optimizer+loss build: {fmt(mem_after_optim)} (+{fmt(mem_after_optim - mem_after_model)})")

    # ── Training iterations ──
    my_model.train()
    os.environ['eval'] = 'false'
    data_iter = iter(train_loader)

    for it in range(args.num_iters):
        print(f"\n{'─' * 70}")
        print(f"  ITERATION {it}")
        print(f"{'─' * 70}")

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        optimizer.zero_grad()

        # ── Data loading ──
        t0 = time.time()
        data = next(data_iter)
        for k in list(data.keys()):
            if isinstance(data[k], torch.Tensor):
                data[k] = data[k].cuda()
        input_imgs = data.pop('img')
        torch.cuda.synchronize()
        t_data = time.time() - t0
        mem_data = mem_mb()
        print(f"  [data]     mem={fmt(mem_data)} time={t_data:.3f}s  "
              f"img_shape={list(input_imgs.shape)}")

        # ── Forward ──
        t1 = time.time()
        with torch.cuda.amp.autocast(amp):
            result_dict = my_model(imgs=input_imgs, metas=data, global_iter=0)
        torch.cuda.synchronize()
        t_forward = time.time() - t1
        mem_forward = mem_mb()
        peak_forward = peak_mb()
        print(f"  [forward]  mem={fmt(mem_forward)} (+{fmt(mem_forward - mem_data)}) "
              f"peak={fmt(peak_forward)} time={t_forward:.3f}s")

        # ── Loss computation ──
        t2 = time.time()
        with torch.cuda.amp.autocast(amp):
            loss_input = {'metas': data}
            for loss_input_key, loss_input_val in cfg.loss_input_convertion.items():
                if loss_input_key not in result_dict:
                    loss_input[loss_input_key] = result_dict['metas'].get(loss_input_val)
                else:
                    loss_input[loss_input_key] = result_dict[loss_input_val]
            loss, loss_dict = loss_func(loss_input)
        torch.cuda.synchronize()
        t_loss = time.time() - t2
        mem_loss = mem_mb()
        print(f"  [loss]     mem={fmt(mem_loss)} (+{fmt(mem_loss - mem_forward)}) time={t_loss:.3f}s")
        loss_details = ", ".join(f"{k}: {v:.3f}" for k, v in loss_dict.items())
        print(f"             {loss_details}")

        # ── Backward ──
        t3 = time.time()
        loss.backward()
        torch.cuda.synchronize()
        t_backward = time.time() - t3
        mem_backward = mem_mb()
        peak_backward = peak_mb()
        print(f"  [backward] mem={fmt(mem_backward)} (+{fmt(mem_backward - mem_loss)}) "
              f"peak={fmt(peak_backward)} time={t_backward:.3f}s")

        # ── Optimizer step ──
        t4 = time.time()
        torch.nn.utils.clip_grad_norm_(my_model.parameters(), cfg.grad_max_norm)
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()
        t_optim = time.time() - t4
        mem_optim = mem_mb()
        print(f"  [optim]    mem={fmt(mem_optim)} time={t_optim:.3f}s")

        # ── Summary ──
        total_time = t_data + t_forward + t_loss + t_backward + t_optim
        print(f"\n  ── Iter {it} Summary ──")
        print(f"  Total time: {total_time:.3f}s")
        print(f"    data:     {t_data:.3f}s ({t_data/total_time*100:.1f}%)")
        print(f"    forward:  {t_forward:.3f}s ({t_forward/total_time*100:.1f}%)")
        print(f"    loss:     {t_loss:.3f}s ({t_loss/total_time*100:.1f}%)")
        print(f"    backward: {t_backward:.3f}s ({t_backward/total_time*100:.1f}%)")
        print(f"    optim:    {t_optim:.3f}s ({t_optim/total_time*100:.1f}%)")
        print(f"  Peak memory: {fmt(peak_backward)}")
        print(f"  Memory breakdown:")
        print(f"    model weights:     {fmt(mem_after_model - mem_start)}")
        print(f"    optimizer states:  {fmt(mem_after_optim - mem_after_model)}")
        print(f"    data on GPU:       {fmt(mem_data - mem_after_optim)}")
        print(f"    forward activations: {fmt(mem_forward - mem_data)}")
        print(f"    loss computation:  {fmt(mem_loss - mem_forward)}")
        print(f"    gradients:         {fmt(mem_backward - mem_loss)}")

    # ── Final summary ──
    print(f"\n{'=' * 70}")
    print(f"FINAL: Peak GPU memory = {fmt(peak_mb())}")
    print(f"H20 GPU total = 97871 MB")
    print(f"Headroom for batch=2: need peak < {97871//2} MB = {fmt(97871/2)}")
    print(f"Current peak uses {peak_mb()/97871*100:.1f}% of GPU")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
