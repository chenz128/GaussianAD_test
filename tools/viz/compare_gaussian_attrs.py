"""
对比多个模型的 Gaussian attr.pth，输出透明度、漂浮、语义分布等指标。
用法：
  python tools/viz/compare_gaussian_attrs.py
"""
import torch, numpy as np, os

NUSC_CLASSES = [
    "barrier","bicycle","bus","car","construction","motorcycle",
    "pedestrian","traffic_cone","trailer","truck",
    "driveable","other_flat","sidewalk","terrain","manmade","vegetation","noise"
]

MODELS = {
    "base_ep30":        "out/nuscenes_gs25600_base/vis/val_0_gaussian_attr.pth",
    "ft_depth_ep10":    "out/nuscenes_gs25600_base_ft_depth/vis/val_0_gaussian_attr.pth",
    "mf_depth_ep15":    "out/nuscenes_gs25600_mf_depth/vis/val_0_gaussian_attr.pth",
    "acc_ep10":         "out/nuscenes_gs25600_acc/vis/val_0_gaussian_attr.pth",
    "concentrate_ep10": "out/nuscenes_gs25600_concentrate/vis/val_0_gaussian_attr.pth",
}

def load_attr(path):
    data = torch.load(path, map_location="cpu")
    if hasattr(data, "means"):
        return data
    class G: pass
    g = G()
    for k, v in data.items():
        setattr(g, k, v)
    return g

# ── 主指标表 ─────────────────────────────────────────────────────────────────
print("=" * 95)
print(f"{'Model':<22} {'N':>6} {'opa_mean':>9} {'opa>0.5':>8} {'opa>0.1':>8} "
      f"{'scale_mean':>11} {'hi-float':>9} {'floating':>9}")
print(f"{'':22} {'':6} {'':9} {'(%)':>8} {'(%)':>8} "
      f"{'(m)':>11} {'z>1m(%)':>9} {'z>0.5,nonGnd':>9}")
print("-" * 95)

results = {}
for model_name, relpath in MODELS.items():
    path = f"/data/chenz/GaussianAD/{relpath}"
    try:
        g = load_attr(path)
        means  = g.means.squeeze(0)               # (N, 3)
        opas   = g.opacities.squeeze(0).squeeze(-1)  # (N,)
        scales = g.scales.squeeze(0)              # (N, 3)
        sems   = g.semantics.squeeze(0)           # (N, 17)
        N = means.shape[0]

        opa_mean  = opas.mean().item()
        opa_gt05  = (opas > 0.5).float().mean().item() * 100
        opa_gt01  = (opas > 0.1).float().mean().item() * 100
        scale_mean = scales.mean().item()

        z = means[:, 2]
        hi_float  = (z > 1.0).float().mean().item() * 100

        # floating: z > 0.5m 且 非地面类（driveable=10, other_flat=11, terrain=13）
        ground_cls = torch.tensor([10, 11, 13])
        pred_cls = sems.argmax(-1)
        not_ground = ~torch.isin(pred_cls, ground_cls)
        floating = ((z > 0.5) & not_ground).float().mean().item() * 100

        results[model_name] = dict(
            N=N, opa_mean=opa_mean, opa_gt05=opa_gt05, opa_gt01=opa_gt01,
            scale_mean=scale_mean, hi_float=hi_float, floating=floating,
            opas=opas, z=z, sems=sems, pred_cls=pred_cls
        )
        print(f"{model_name:<22} {N:>6} {opa_mean:>9.3f} {opa_gt05:>7.1f}% "
              f"{opa_gt01:>7.1f}% {scale_mean:>10.3f}m {hi_float:>8.1f}% {floating:>8.1f}%")
    except Exception as e:
        print(f"{model_name:<22} ERROR: {e}")

print("=" * 95)

# ── opacity 直方图（分桶） ────────────────────────────────────────────────────
print()
print("Opacity histogram (% of Gaussians in each bucket):")
buckets = [(0.0,0.1),(0.1,0.3),(0.3,0.5),(0.5,0.7),(0.7,0.9),(0.9,1.01)]
hdr = f"{'Model':<22}" + "".join(f"  [{lo:.1f},{hi:.1f})" for lo,hi in buckets)
print(hdr)
print("-" * len(hdr))
for name, r in results.items():
    opas = r["opas"]
    row = f"{name:<22}"
    for lo, hi in buckets:
        pct = ((opas >= lo) & (opas < hi)).float().mean().item() * 100
        row += f"  {pct:>10.1f}%"
    print(row)

# ── 语义分布（opa>0.1 高斯） ─────────────────────────────────────────────────
print()
print("Top-5 predicted classes (opa > 0.1 Gaussians):")
print("-" * 80)
for name, r in results.items():
    mask = r["opas"] > 0.1
    pred_cls = r["pred_cls"][mask]
    total = mask.sum().item()
    counts = torch.bincount(pred_cls, minlength=17)
    top5 = counts.argsort(descending=True)[:5]
    parts = []
    for ci in top5.tolist():
        cls_name = NUSC_CLASSES[ci] if ci < len(NUSC_CLASSES) else f"cls{ci}"
        pct = counts[ci].item() / max(total, 1) * 100
        parts.append(f"{cls_name}:{pct:.0f}%")
    print(f"  {name:<22}: {', '.join(parts)}  (visible={total})")

# ── Z 高度分布 ───────────────────────────────────────────────────────────────
print()
print("Z-height distribution (all Gaussians, percentiles in meters):")
print(f"{'Model':<22} {'p5':>7} {'p25':>7} {'p50':>7} {'p75':>7} {'p95':>7} {'p99':>7} {'mean':>7}")
print("-" * 70)
for name, r in results.items():
    z = r["z"].numpy()
    ps = np.percentile(z, [5, 25, 50, 75, 95, 99])
    print(f"{name:<22} {ps[0]:>7.2f} {ps[1]:>7.2f} {ps[2]:>7.2f} "
          f"{ps[3]:>7.2f} {ps[4]:>7.2f} {ps[5]:>7.2f} {z.mean():>7.2f}")

# ── scale 分布 ───────────────────────────────────────────────────────────────
print()
print("Scale distribution (per-axis mean, all Gaussians):")
print(f"{'Model':<22} {'sx_mean':>9} {'sy_mean':>9} {'sz_mean':>9} {'s_max':>9} {'s_min':>9}")
print("-" * 65)
for model_name, relpath in MODELS.items():
    path = f"/data/chenz/GaussianAD/{relpath}"
    try:
        g = load_attr(path)
        sc = g.scales.squeeze(0)
        print(f"{model_name:<22} {sc[:,0].mean():>9.4f} {sc[:,1].mean():>9.4f} "
              f"{sc[:,2].mean():>9.4f} {sc.max():>9.4f} {sc.min():>9.4f}")
    except Exception as e:
        print(f"{model_name:<22} ERROR: {e}")
