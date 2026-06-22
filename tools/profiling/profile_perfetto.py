"""
Generate a Perfetto-compatible Chrome Trace JSON for GaussianAD training.

Captures CPU + CUDA kernel-level timing with high-level stage annotations.
Output can be loaded directly at https://ui.perfetto.dev/

Usage:
    CUDA_VISIBLE_DEVICES=0 /data/chenz/conda_env/splatting/bin/python tools/profiling/profile_perfetto.py \
        --py-config config/nuscenes_gs25600.py --num-iters 5 --output trace.json

    # Then open https://ui.perfetto.dev/ and drag-drop trace.json
"""
import argparse
import os
import os.path as osp
import sys
import time
import torch
import numpy as np

# file now lives in tools/profiling/, go up 3 levels to reach repo root
sys.path.insert(0, osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True


def patch_forward_with_markers(model):
    """
    Replace model.forward with a version that inserts torch.profiler.record_function
    markers for each sub-module stage, giving clear high-level spans in Perfetto.
    """
    original_forward = model.forward

    def traced_forward(imgs=None, metas=None, points=None, **kwargs):
        results = {
            'imgs': imgs,
            'metas': metas,
            'points': points,
            'gt_boxes': metas['gt_boxes'],
        }
        results.update(kwargs)

        with torch.profiler.record_function("## 1. img_backbone+fpn"):
            outs = model.extract_img_feat(**results)
        results.update(outs)

        with torch.profiler.record_function("## 2. lifter"):
            outs = model.lifter(**results)
        results.update(outs)

        with torch.profiler.record_function("## 3. encoder"):
            outs = model.encoder(**results)
        results.update(outs)

        if hasattr(model, 'temporal_encoder'):
            with torch.profiler.record_function("## 4. temporal_encoder"):
                outs = model.temporal_encoder(**results)
            results.update(outs)

        with torch.profiler.record_function("## 5. decoder"):
            outs = model.decoder(results)
        results.update(outs)

        if hasattr(model, 'map_decoder'):
            with torch.profiler.record_function("## 6. map_decoder"):
                outs = model.map_decoder(results)
            results.update(outs)

        with torch.profiler.record_function("## 7. planner_head"):
            outs = model.planner_head(results)
        results.update(outs)

        with torch.profiler.record_function("## 8. head"):
            outs = model.head(**results)
        results.update(outs)

        return results

    model.forward = traced_forward


def main():
    parser = argparse.ArgumentParser(description="Generate Perfetto trace for GaussianAD")
    parser.add_argument("--py-config", required=True)
    parser.add_argument("--num-iters", type=int, default=5,
                        help="Total iterations (first 2 are warmup, rest are profiled)")
    parser.add_argument("--output", type=str, default="trace.json",
                        help="Output trace file path")
    args = parser.parse_args()

    from mmengine import Config
    from mmengine.logging import MMLogger
    from mmengine.optim import build_optim_wrapper
    from mmseg.models import build_segmentor

    cfg = Config.fromfile(args.py_config)
    cfg.work_dir = '/tmp/profile_perfetto'
    os.makedirs(cfg.work_dir, exist_ok=True)

    logger = MMLogger('selfocc', log_file=osp.join(cfg.work_dir, 'profile.log'))
    MMLogger._instance_dict['selfocc'] = logger

    import model as _  # noqa: register modules
    from dataset import get_dataloader
    from loss import OPENOCC_LOSS

    print("=" * 70)
    print("  GaussianAD Perfetto Trace Generator")
    print("=" * 70)

    # ── Build model ──
    my_model = build_segmentor(cfg.model)
    my_model.init_weights()
    for mod_name in cfg.get('frozen_modules', []):
        mod = getattr(my_model, mod_name, None)
        if mod is not None:
            for param in mod.parameters():
                param.requires_grad = False
    my_model = my_model.cuda()
    my_model.train()
    os.environ['eval'] = 'false'

    n_params = sum(p.numel() for p in my_model.parameters())
    n_trainable = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    print(f"  Params: {n_params/1e6:.1f}M | Trainable: {n_trainable/1e6:.1f}M")

    # ── Build dataloader, optimizer, loss ──
    train_loader, _ = get_dataloader(
        cfg.train_dataset_config, cfg.val_dataset_config,
        cfg.train_loader, cfg.val_loader, dist=False, iter_resume=False)
    optimizer = build_optim_wrapper(my_model, cfg.optimizer)
    loss_func = OPENOCC_LOSS.build(cfg.loss).cuda()
    amp = cfg.get('amp', False)

    # ── Patch forward with record_function markers ──
    patch_forward_with_markers(my_model)

    # ── Dataloader iterator ──
    data_iter = iter(train_loader)

    warmup = min(2, args.num_iters - 1)
    active = args.num_iters - warmup
    print(f"  Warmup: {warmup} iters | Profiled: {active} iters")
    print(f"  Output: {args.output}")
    print()

    # ── Warmup (no profiling, let CUDA JIT / cuDNN autotune settle) ──
    for i in range(warmup):
        print(f"  Warmup iter {i}...", flush=True)
        data = next(data_iter)
        for k in list(data.keys()):
            if isinstance(data[k], torch.Tensor):
                data[k] = data[k].cuda()
        input_imgs = data.pop('img')
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(amp):
            result_dict = my_model(imgs=input_imgs, metas=data, global_iter=0)
            loss_input = {'metas': data}
            for lk, lv in cfg.loss_input_convertion.items():
                if lk not in result_dict:
                    loss_input[lk] = result_dict['metas'].get(lv)
                else:
                    loss_input[lk] = result_dict[lv]
            # Optional render-loss inputs (visualization only)
            loss_input.setdefault('input_imgs', None)
            loss_input.setdefault('aug_flip', None)
            loss, _ = loss_func(loss_input)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(my_model.parameters(), cfg.grad_max_norm)
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()

    # ── Profiled iterations ──
    print(f"  Starting profiled iterations...", flush=True)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,  # stack traces make the file huge, skip
        with_flops=True,
    ) as prof:
        for i in range(active):
            with torch.profiler.record_function(f"=== ITER {warmup + i} ==="):
                # Data loading
                with torch.profiler.record_function("## 0. data_loading"):
                    data = next(data_iter)
                    for k in list(data.keys()):
                        if isinstance(data[k], torch.Tensor):
                            data[k] = data[k].cuda()
                    input_imgs = data.pop('img')

                optimizer.zero_grad()

                # Forward
                with torch.profiler.record_function("## FORWARD"):
                    with torch.cuda.amp.autocast(amp):
                        result_dict = my_model(imgs=input_imgs, metas=data, global_iter=0)

                # Loss
                with torch.profiler.record_function("## 9. loss_compute"):
                    with torch.cuda.amp.autocast(amp):
                        loss_input = {'metas': data}
                        for lk, lv in cfg.loss_input_convertion.items():
                            if lk not in result_dict:
                                loss_input[lk] = result_dict['metas'].get(lv)
                            else:
                                loss_input[lk] = result_dict[lv]
                        # Optional render-loss inputs (visualization only)
                        loss_input.setdefault('input_imgs', None)
                        loss_input.setdefault('aug_flip', None)
                        loss, loss_dict = loss_func(loss_input)

                # Backward
                with torch.profiler.record_function("## 10. backward"):
                    loss.backward()

                # Optimizer
                with torch.profiler.record_function("## 11. optimizer_step"):
                    torch.nn.utils.clip_grad_norm_(my_model.parameters(), cfg.grad_max_norm)
                    optimizer.step()
                    optimizer.zero_grad()

                torch.cuda.synchronize()
            print(f"  Profiled iter {warmup + i} done.", flush=True)

    # ── Export ──
    output_path = osp.abspath(args.output)
    prof.export_chrome_trace(output_path)
    file_size = os.path.getsize(output_path) / 1024 / 1024
    print(f"\n{'=' * 70}")
    print(f"  Trace exported: {output_path} ({file_size:.1f} MB)")
    print(f"  Open https://ui.perfetto.dev/ and drag-drop the file to visualize.")
    print(f"{'=' * 70}")

    # ── Also print a quick text summary ──
    print("\n  Top 30 CUDA kernels by total time:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))


if __name__ == "__main__":
    main()
