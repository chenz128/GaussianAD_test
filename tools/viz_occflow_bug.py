"""
可视化 empty-gaussian bug 的效果：
  当前帧预测（正确）vs 未来帧预测（有bug，全占用）vs 未来帧GT

用法：
  python tools/viz_occflow_bug.py \
      --npz out/nuscenes_gs25600_gtbox_oracle/vis/val_0_occflow.npz \
      --out /tmp/occflow_bug.html
"""
import argparse, os
import numpy as np

# nuScenes 17 类 + empty 的颜色（BGR→RGB转换后的调色板）
PALETTE = [
    [255, 120,  50],  # 1 barrier        橙
    [255, 192, 203],  # 2 bicycle        粉
    [255, 255,   0],  # 3 bus            黄
    [  0, 150, 245],  # 4 car            蓝
    [  0, 255, 255],  # 5 construction   青
    [200, 180,   0],  # 6 motorcycle     土黄
    [255,   0,   0],  # 7 pedestrian     红
    [255, 240, 150],  # 8 traffic_cone   浅黄
    [135,  60,   0],  # 9 trailer        棕
    [160,  32, 240],  # 10 truck         紫
    [255,   0, 255],  # 11 driveable     品红
    [139, 137, 137],  # 12 other_flat    灰
    [ 75, 181, 180],  # 13 sidewalk      蓝绿
    [222, 184, 135],  # 14 terrain       沙色
    [  0, 175,   0],  # 15 manmade       深绿
    [ 34, 139,  34],  # 16 vegetation    绿
]
EMPTY_COLOR = [20, 20, 20]      # class 17: 近黑（空）
UNKNOWN_COLOR = [50, 50, 50]    # class 0 / -1 / 初始化默认值

def label_to_rgb(label_2d):
    """label_2d: (H, W) int → (H, W, 3) RGB"""
    H, W = label_2d.shape
    rgb = np.full((H, W, 3), UNKNOWN_COLOR, dtype=np.uint8)
    for c, col in enumerate(PALETTE, start=1):
        rgb[label_2d == c] = col
    rgb[label_2d == 17] = EMPTY_COLOR
    return rgb


def bev_topdown(xyz, labels, pc_range=(-30, -30, -2, 30, 30, 2), res=0.3):
    """
    将点云 + 标签渲染成俯视图（Z轴最高标签覆盖）
    xyz:    (N, 3)
    labels: (N,) int
    返回 (H, W, 3) RGB image
    """
    xmin, ymin, _, xmax, ymax, _ = pc_range
    W = int((xmax - xmin) / res)
    H = int((ymax - ymin) / res)
    img = np.full((H, W, 3), EMPTY_COLOR, dtype=np.uint8)

    # 只画非空体素
    mask = labels != 17
    if mask.sum() == 0:
        return img
    pts = xyz[mask]
    labs = labels[mask]

    xi = np.clip(((pts[:, 0] - xmin) / res).astype(int), 0, W - 1)
    yi = np.clip(((pts[:, 1] - ymin) / res).astype(int), 0, H - 1)
    # 按 Z 从低到高画（高处覆盖低处）
    zi_order = np.argsort(pts[:, 2])
    for idx in zi_order:
        c = labs[idx]
        col = PALETTE[c - 1] if 1 <= c <= 16 else UNKNOWN_COLOR
        img[yi[idx], xi[idx]] = col

    # 以 x 轴（前方）朝上显示
    return img[::-1, :, :]


def make_html(panels):
    """
    panels: list of dict {title, img_b64, note}
    生成一页横向展示的 HTML
    """
    import base64, io
    from PIL import Image

    cells = ""
    for p in panels:
        # img 是 (H, W, 3) numpy
        pil = Image.fromarray(p["img"])
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        note_html = f'<p style="color:#aaa;font-size:12px">{p.get("note","")}</p>' if p.get("note") else ""
        cells += f"""
        <td style="text-align:center;padding:8px;vertical-align:top">
          <b style="color:white">{p["title"]}</b><br>
          <img src="data:image/png;base64,{b64}" style="width:220px;image-rendering:pixelated"><br>
          {note_html}
        </td>"""

    legend = ""
    class_names = ["barrier","bicycle","bus","car","construction","motorcycle",
                   "pedestrian","traffic_cone","trailer","truck",
                   "driveable","other_flat","sidewalk","terrain","manmade","vegetation"]
    for i, (name, col) in enumerate(zip(class_names, PALETTE)):
        rgb = f"rgb({col[0]},{col[1]},{col[2]})"
        legend += f'<span style="background:{rgb};padding:2px 8px;margin:2px;font-size:11px;color:black;display:inline-block">{name}</span>'
    legend += f'<span style="background:rgb(20,20,20);padding:2px 8px;margin:2px;font-size:11px;color:white;border:1px solid #555;display:inline-block">empty (空)</span>'

    return f"""<!DOCTYPE html><html><body style="background:#111;font-family:sans-serif">
<h2 style="color:white;text-align:center">Empty-Gaussian Bug 可视化（BEV 俯视图）</h2>
<p style="color:#ccc;text-align:center">自车在中心，前方朝上；深色=空，彩色=占用</p>
<table style="margin:auto"><tr>{cells}</tr></table>
<div style="text-align:center;margin-top:16px">{legend}</div>
</body></html>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    parser.add_argument("--out", default="/tmp/occflow_bug.html")
    parser.add_argument("--frame", type=int, default=0,
                        help="哪一帧作为'未来帧'示例 (0-5, 默认0=0.5s后)")
    args = parser.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    xyz      = d["xyz"]         # (N, 3)
    occ_now  = d["occ_now"]     # (N,)
    occ_fut  = d["occ_fut"]     # (6, N)
    occ_gt   = d["occ_fut_gt"]  # (6, N)
    valid    = d["valid"]       # (6,)

    fi = args.frame
    pred_fut = occ_fut[fi].astype(int)
    gt_fut   = occ_gt[fi].astype(int)
    gt_fut[gt_fut < 0] = 17  # 未初始化的填 empty

    # 统计
    N = len(occ_now)
    def pct_empty(lab): return 100.0 * (lab == 17).sum() / N
    def pct_occ(lab):   return 100.0 * (lab != 17).sum() / N

    # 生成 BEV 图
    img_now  = bev_topdown(xyz, occ_now)
    img_pred = bev_topdown(xyz, pred_fut)
    img_gt   = bev_topdown(xyz, gt_fut)

    # 差异图：预测错误处标红，正确处透明
    diff = np.full_like(img_gt, EMPTY_COLOR)
    wrong_mask  = (pred_fut != gt_fut) & (gt_fut != 17)   # GT有物体但预测错了
    miss_mask   = (pred_fut != 17) & (gt_fut == 17)        # GT是空但预测成有物体（核心bug）
    mask_x = np.clip(((xyz[:, 0] - (-30)) / 0.3).astype(int), 0, diff.shape[1]-1)
    mask_y = np.clip(((xyz[:, 1] - (-30)) / 0.3).astype(int), 0, diff.shape[0]-1)
    for i in range(N):
        if miss_mask[i]:
            diff[diff.shape[0]-1-mask_y[i], mask_x[i]] = [255, 50, 50]   # 红：GT空但预测成有物体
        elif wrong_mask[i]:
            diff[diff.shape[0]-1-mask_y[i], mask_x[i]] = [255, 200, 0]   # 黄：语义预测错

    panels = [
        {
            "title": "当前帧预测（正确）",
            "img": img_now,
            "note": f"含 empty 高斯 ✓<br>空体素占比: {pct_empty(occ_now):.1f}%"
        },
        {
            "title": f"未来帧预测（有bug，t+{(fi+1)*0.5:.1f}s）",
            "img": img_pred,
            "note": f"丢失 empty 高斯 ✗<br>空体素占比: {pct_empty(pred_fut):.1f}%（应为~{pct_empty(gt_fut):.0f}%）"
        },
        {
            "title": f"未来帧 GT（t+{(fi+1)*0.5:.1f}s）",
            "img": img_gt,
            "note": f"空体素占比: {pct_empty(gt_fut):.1f}%"
        },
        {
            "title": "差异图",
            "img": diff,
            "note": '<span style="color:#f55">红=GT空但预测成占用（bug主因）</span><br>'
                    '<span style="color:#fc0">黄=语义类别预测错</span>'
        },
    ]

    html = make_html(panels)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(html)
    print(f"saved: {args.out}")
    print(f"\n统计（未来帧 t+{(fi+1)*0.5:.1f}s）:")
    print(f"  当前帧   空体素: {pct_empty(occ_now):.1f}%  占用: {pct_occ(occ_now):.1f}%")
    print(f"  GT未来帧 空体素: {pct_empty(gt_fut):.1f}%  占用: {pct_occ(gt_fut):.1f}%")
    print(f"  预测未来帧空体素: {pct_empty(pred_fut):.1f}%  占用: {pct_occ(pred_fut):.1f}%  ← 应接近GT")


if __name__ == "__main__":
    main()
