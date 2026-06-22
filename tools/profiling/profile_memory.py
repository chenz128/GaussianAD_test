"""
Sub-module level Memory & Time Profiling for GaussianAD (splatting branch).
Instruments each stage of the forward pass and reports per-module memory/time.

Usage:
    CUDA_VISIBLE_DEVICES=4 /data/chenz/conda_env/splatting/bin/python tools/profiling/profile_memory.py \
        --py-config config/nuscenes_gs25600.py --num-iters 3
"""
import time
import argparse
import os
import os.path as osp
import sys
import torch
import numpy as np
from functools import wraps

# Add project root to path (file now lives in tools/profiling/, go up 3 levels)
sys.path.insert(0, osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True


# ═══════════════════════ Utilities ═══════════════════════

def mem_mb():
    return torch.cuda.memory_allocated() / 1024 / 1024

def peak_mb():
    return torch.cuda.max_memory_allocated() / 1024 / 1024

def fmt(mb):
    if mb >= 1024:
        return f"{mb/1024:.2f} GB"
    return f"{mb:.0f} MB"


class Profiler:
    """Context manager for timing + memory delta."""
    def __init__(self, name):
        self.name = name
        self.mem_before = 0
        self.mem_after = 0
        self.peak = 0
        self.elapsed = 0

    def __enter__(self):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.mem_before = mem_mb()
        self._t0 = time.time()
        return self

    def __exit__(self, *args):
        torch.cuda.synchronize()
        self.elapsed = time.time() - self._t0
        self.mem_after = mem_mb()
        self.peak = peak_mb()

    @property
    def delta(self):
        return self.mem_after - self.mem_before

    def report(self):
        return (f"  [{self.name:20s}] "
                f"delta={fmt(self.delta):>10s}  "
                f"peak={fmt(self.peak):>10s}  "
                f"time={self.elapsed:.3f}s")


# ═══════════════════════ Monkey-patch model forward ═══════════════════════

def make_profiled_forward(model):
    """
    Replace model.forward with a version that records per-sub-module timing/memory.
    Returns a dict of Profiler objects after each call.
    """
    original_forward = model.forward
    profilers = {}

    def profiled_forward(imgs=None, metas=None, points=None, **kwargs):
        results = {
            'imgs': imgs,
            'metas': metas,
            'points': points,
            'gt_boxes': metas['gt_boxes'],
        }
        results.update(kwargs)

        # ── Stage 1: Image Backbone + FPN ──
        with Profiler("img_backbone+fpn") as p:
            outs = model.extract_img_feat(**results)
        profilers["img_backbone+fpn"] = p
        results.update(outs)

        # ── Stage 2: Lifter ──
        with Profiler("lifter") as p:
            outs = model.lifter(**results)
        profilers["lifter"] = p
        results.update(outs)

        # ── Stage 3: Encoder (6-layer deformable + spconv + refine) ──
        with Profiler("encoder") as p:
            outs = model.encoder(**results)
        profilers["encoder"] = p
        results.update(outs)

        # ── Stage 4: Temporal Encoder ──
        if hasattr(model, 'temporal_encoder'):
            with Profiler("temporal_encoder") as p:
                outs = model.temporal_encoder(**results)
            profilers["temporal_encoder"] = p
            results.update(outs)

        # ── Stage 5: Decoder (detection head) ──
        with Profiler("decoder") as p:
            outs = model.decoder(results)
        profilers["decoder"] = p
        results.update(outs)

        # ── Stage 6: Map Decoder (frozen in splatting) ──
        if hasattr(model, 'map_decoder'):
            with Profiler("map_decoder") as p:
                outs = model.map_decoder(results)
            profilers["map_decoder"] = p
            results.update(outs)

        # ── Stage 7: Planner Head (frozen) ──
        with Profiler("planner_head") as p:
            outs = model.planner_head(results)
        profilers["planner_head"] = p
        results.update(outs)

        # ── Stage 8: Gaussian Head (LocalAggregator + 2D Rasterizer) ──
        # We further split this into sub-stages
        head = model.head

        # 8a: prepare gaussian args + LocalAggregator (3D render)
        with Profiler("head_3d_render") as p:
            occ_xyz = metas['occ_xyz'].to(head.zero_tensor.device)
            occ_label = metas['occ_label'].to(head.zero_tensor.device)
            occ_cam_mask = metas['occ_cam_mask'].to(head.zero_tensor.device)
            sampled_xyz, sampled_label = head._sampling(occ_xyz, occ_label, None)
            gaussians = results['representation_temp']['gaussian']
            means, origi_opa, opacities, scales, CovInv = head.prepare_gaussian_args(gaussians)
            bs, g = means.shape[:2]
            semantics_3d = head.aggregator(
                sampled_xyz.clone().float(), means,
                origi_opa.reshape(bs, g), opacities, scales, CovInv
            )[None].transpose(1, 2)
        profilers["head_3d_render"] = p

        # 8b: forward_flow (occ flow prediction)
        with Profiler("head_flow") as p:
            occ_flow = head.forward_flow(
                sampled_xyz, results['representation_temp'],
                metas=metas, gs=(origi_opa, opacities, scales, CovInv),
                **{k: results[k] for k in ['offset', 'ego_fut_preds'] if k in results})
        profilers["head_flow"] = p

        # 8c: 2D Gaussian splatting (gsplat)
        rendered_sem, rendered_depth = None, None
        if head.rasterizer_2d is not None and head.training:
            with Profiler("head_gsplat_2d") as p:
                gs_extrins = metas['gs_extrins'].to(head.zero_tensor.device)
                gs_intrins = metas['gs_intrins'].to(head.zero_tensor.device)
                rendered_sem, rendered_depth = head.rasterizer_2d(gaussians, gs_extrins, gs_intrins)
            profilers["head_gsplat_2d"] = p

        # Assemble output dict
        output = {
            'pred_occ': [semantics_3d],
            'sampled_label': sampled_label,
            'sampled_xyz': sampled_xyz,
            'occ_mask': occ_cam_mask,
            'gaussian': gaussians,
            'occ_flow': occ_flow,
        }
        if head.rasterizer_2d is not None:
            output['rendered_sem'] = rendered_sem
            output['rendered_depth'] = rendered_depth
        results.update(output)
        return results

    model.forward = profiled_forward
    return profilers


# ═══════════════════════ Main ═══════════════════════

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

    print("=" * 75)
    print("  GaussianAD Sub-Module Memory & Time Profiler (splatting branch)")
    print("=" * 75)

    # ── Build model ──
    torch.cuda.reset_peak_memory_stats()
    mem_start = mem_mb()

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
    torch.cuda.synchronize()

    mem_model = mem_mb()
    n_params = sum(p.numel() for p in my_model.parameters())
    n_trainable = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    print(f"\n  Model weights on GPU: {fmt(mem_model - mem_start)}")
    print(f"  Total params: {n_params/1e6:.1f}M | Trainable: {n_trainable/1e6:.1f}M")
    print(f"  AMP: {cfg.get('amp', False)} | Batch: {cfg.train_loader.get('batch_size', 1)}")

    # ── Build dataloader ──
    train_loader, _ = get_dataloader(
        cfg.train_dataset_config, cfg.val_dataset_config,
        cfg.train_loader, cfg.val_loader, dist=False, iter_resume=False)

    # ── Build optimizer & loss ──
    optimizer = build_optim_wrapper(my_model, cfg.optimizer)
    loss_func = OPENOCC_LOSS.build(cfg.loss).cuda()
    amp = cfg.get('amp', False)

    # ── Install profiled forward ──
    profilers = make_profiled_forward(my_model)

    # ── Run iterations ──
    data_iter = iter(train_loader)

    for it in range(args.num_iters):
        print(f"\n{'━' * 75}")
        print(f"  ITERATION {it}")
        print(f"{'━' * 75}")

        profilers.clear()
        torch.cuda.reset_peak_memory_stats()
        optimizer.zero_grad()

        # ── Data ──
        with Profiler("data_loading") as p_data:
            data = next(data_iter)
            for k in list(data.keys()):
                if isinstance(data[k], torch.Tensor):
                    data[k] = data[k].cuda()
            input_imgs = data.pop('img')
        print(f"\n{p_data.report()}  shape={list(input_imgs.shape)}")

        # ── Forward (instrumented) ──
        mem_pre_fwd = mem_mb()
        torch.cuda.reset_peak_memory_stats()
        t_fwd_start = time.time()
        with torch.cuda.amp.autocast(amp):
            result_dict = my_model(imgs=input_imgs, metas=data, global_iter=0)
        torch.cuda.synchronize()
        t_fwd_total = time.time() - t_fwd_start
        peak_fwd = peak_mb()

        print(f"\n  ┌{'─'*73}┐")
        print(f"  │ {'FORWARD BREAKDOWN':^71s} │")
        print(f"  ├{'─'*73}┤")
        print(f"  │ {'Module':<22s} {'Mem Delta':>10s} {'Peak':>10s} {'Time':>8s} {'%Fwd':>6s} │")
        print(f"  ├{'─'*73}┤")
        total_profiled_time = 0
        for name, p in profilers.items():
            pct = p.elapsed / t_fwd_total * 100 if t_fwd_total > 0 else 0
            total_profiled_time += p.elapsed
            print(f"  │ {name:<22s} {fmt(p.delta):>10s} {fmt(p.peak):>10s} {p.elapsed:>7.3f}s {pct:>5.1f}% │")
        print(f"  ├{'─'*73}┤")
        print(f"  │ {'TOTAL FORWARD':<22s} {fmt(peak_fwd - mem_pre_fwd):>10s} {fmt(peak_fwd):>10s} {t_fwd_total:>7.3f}s {'100%':>6s} │")
        print(f"  └{'─'*73}┘")

        # ── Loss ──
        with Profiler("loss_compute") as p_loss:
            with torch.cuda.amp.autocast(amp):
                loss_input = {'metas': data}
                for lk, lv in cfg.loss_input_convertion.items():
                    if lk not in result_dict:
                        loss_input[lk] = result_dict['metas'].get(lv)
                    else:
                        loss_input[lk] = result_dict[lv]
                loss, loss_dict = loss_func(loss_input)
        print(f"\n{p_loss.report()}")

        # ── Backward ──
        with Profiler("backward") as p_bwd:
            loss.backward()
        peak_total = peak_mb()
        print(f"{p_bwd.report()}  peak_total={fmt(peak_total)}")

        # ── Optimizer ──
        with Profiler("optimizer_step") as p_opt:
            torch.nn.utils.clip_grad_norm_(my_model.parameters(), cfg.grad_max_norm)
            optimizer.step()
            optimizer.zero_grad()
        print(f"{p_opt.report()}")

        # ── Summary table ──
        all_stages = [("data_loading", p_data)] + list(profilers.items()) + [
            ("loss_compute", p_loss), ("backward", p_bwd), ("optimizer_step", p_opt)]
        total_time = sum(s[1].elapsed for s in all_stages)

        print(f"\n  ╔{'═'*73}╗")
        print(f"  ║ {'ITER SUMMARY':^71s} ║")
        print(f"  ╠{'═'*73}╣")
        print(f"  ║ {'Stage':<22s} {'Time':>8s} {'%Total':>7s} {'Mem Delta':>10s} {'Peak':>10s}    ║")
        print(f"  ╠{'═'*73}╣")
        for name, p in all_stages:
            pct = p.elapsed / total_time * 100
            print(f"  ║ {name:<22s} {p.elapsed:>7.3f}s {pct:>6.1f}% {fmt(p.delta):>10s} {fmt(p.peak):>10s}    ║")
        print(f"  ╠{'═'*73}╣")
        print(f"  ║ {'TOTAL':22s} {total_time:>7.3f}s {'100%':>7s} {'':>10s} {fmt(peak_total):>10s}    ║")
        print(f"  ╚{'═'*73}╝")

    # ── Final ──
    print(f"\n{'=' * 75}")
    print(f"  Peak GPU Memory: {fmt(peak_total)}")
    print(f"  H20 Single GPU:  {fmt(97871)}")
    print(f"  Usage:           {peak_total/97871*100:.1f}%")
    print(f"  Batch=2 needs:   peak < {fmt(97871/2)}")
    if peak_total < 97871 / 2:
        print(f"  ✓ Batch=2 is FEASIBLE with current settings!")
    else:
        overshoot = peak_total - 97871/2
        print(f"  ✗ Need to save {fmt(overshoot)} for batch=2")
    print(f"{'=' * 75}")


if __name__ == "__main__":
    main()
