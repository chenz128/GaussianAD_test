import os
import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from . import OPENOCC_LOSS
from .base_loss import BaseLoss

# nuScenes occupancy 17类颜色表（0-indexed，对应 pseudo_seg 中 label 1~17 减1后的索引）
_NUSC_PALETTE = np.array([
    [112, 128, 144],  # 0: barrier
    [220,  20,  60],  # 1: bicycle
    [255, 127,  80],  # 2: bus
    [255, 158,   0],  # 3: car
    [233, 150,  70],  # 4: construction_vehicle
    [255,  61,  99],  # 5: motorcycle
    [  0,   0, 230],  # 6: pedestrian
    [ 47,  79,  79],  # 7: traffic_cone
    [255, 140,   0],  # 8: trailer
    [255,  99,  71],  # 9: truck
    [  0, 207, 191],  # 10: driveable_surface
    [175,   0,  75],  # 11: other_flat
    [ 75,   0,  75],  # 12: sidewalk
    [112, 180,  60],  # 13: terrain
    [222, 184, 135],  # 14: manmade
    [  0, 175,   0],  # 15: vegetation
    [  0,   0,   0],  # 16: free/empty
], dtype=np.uint8)


def _colorize_sem(cls_map_0indexed):
    """cls_map_0indexed: (H, W) int, values 0-16 → RGB (H, W, 3)"""
    cls = np.clip(cls_map_0indexed, 0, 16)
    return _NUSC_PALETTE[cls]


def _depth_to_rgb(depth_np, vmin=0.0, vmax=40.0):
    """depth_np: (H, W) float → RGB (H, W, 3) using turbo-like colormap"""
    norm = np.clip((depth_np - vmin) / (vmax - vmin + 1e-6), 0.0, 1.0)
    # simple heat map: black→blue→cyan→green→yellow→red
    r = np.clip(norm * 4 - 2, 0, 1)
    g = np.clip(np.minimum(norm * 4, 4 - norm * 4), 0, 1)
    b = np.clip(1 - norm * 4, 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    # invalid pixels (depth==0) → gray
    invalid = depth_np <= 0
    rgb[invalid] = 0.5
    return (rgb * 255).astype(np.uint8)


@OPENOCC_LOSS.register_module()
class RenderLoss(BaseLoss):
    """
    Pseudo-label supervision loss using 2D Gaussian splatting renders.
    Computes semantic CE loss + depth MSE loss against pseudo labels.
    """

    def __init__(
        self,
        weight=1.0,
        sem_lw=2.0,
        depth_lw=0.05,
        vis_dir=None,
        vis_every=500,
        input_dict=None,
        **kwargs,
    ):
        if input_dict is None:
            input_dict = {
                'rendered_sem': 'rendered_sem',
                'rendered_depth': 'rendered_depth',
                'pseudo_seg': 'pseudo_seg',
                'pseudo_depth': 'pseudo_depth',
            }
        super().__init__(weight=weight, input_dict=input_dict, **kwargs)
        # BaseLoss.__init__ sets self.loss_func = lambda: 0 as instance attr,
        # which shadows our loss_func method. Delete it to restore method lookup.
        del self.loss_func

        self.sem_lw = sem_lw
        self.depth_lw = depth_lw
        self.vis_dir = vis_dir
        self.vis_every = vis_every

        # dynamic classes — depth loss unreliable for moving objects
        self.dynamic_classes = torch.tensor([2, 3, 4, 5, 6, 7, 9, 10])

        # class frequency weights for balanced CE loss (nuScenes 17 classes)
        nusc_class_freq = torch.tensor([
            944004, 1897170, 152386, 2391677, 16957802, 724139,
            189027, 2074468, 413451, 2384460, 5916653, 175883646,
            4275424, 51393615, 61411620, 105975596, 116424404
        ], dtype=torch.float32)
        log_w = torch.log(nusc_class_freq.sum() / nusc_class_freq)
        self.register_buffer('class_weight', log_w / log_w.mean())

        self.loss_fn_ce = nn.CrossEntropyLoss(reduction='none')
        self.loss_fn_depth = nn.MSELoss()

    def loss_func(self, rendered_sem, rendered_depth, pseudo_seg, pseudo_depth):
        """
        Args:
            rendered_sem:   (B, nC, H, W, 17) — rendered semantic logits
            rendered_depth: (B, nC, H, W)     — rendered depth
            pseudo_seg:     (B, nC, H, W)     — pseudo semantic labels (0=invalid)
            pseudo_depth:   (B, nC, H, W)     — pseudo depth (0=invalid)
        """
        # eval mode: rendering was skipped, or val dataset has no pseudo labels
        if rendered_sem is None or rendered_depth is None or pseudo_seg is None or pseudo_depth is None:
            return torch.tensor(0.0, requires_grad=False)

        # ── semantic loss ──semantic是指渲染结果的语义分割图，pseudo_seg是指根据点云生成的伪标签语义分割图。rendered_sem和pseudo_seg都是(B, nC, H, W)的形状，其中nC是相机数量，H和W是图像的高和宽。semantic loss是计算rendered_sem和pseudo_seg之间的交叉熵损失，注意pseudo_seg中的0表示无效像素，不参与损失计算。
        pred_sem = rendered_sem.flatten(0, -2)     # (N, 17)
        target_sem = pseudo_seg.flatten().long()    # (N,)
        valid_sem = target_sem > 0
        if valid_sem.any():
            pw = self.class_weight[target_sem[valid_sem]]#这是一个权重张量，用于平衡不同类别的损失。class_weight是根据nuScenes数据集中各个类别的频率计算得到的，频率越低的类别权重越高。通过索引target_sem[valid_sem]，我们可以得到每个有效像素对应的类别权重，从而在计算交叉熵损失时给予稀有类别更大的关注。
            loss_sem = self.sem_lw * (
                pw * self.loss_fn_ce(pred_sem[valid_sem], target_sem[valid_sem] - 1)
            ).mean()#注意这里的target_sem[valid_sem] - 1是因为pseudo_seg中的类别标签是从1开始的，而CrossEntropyLoss要求类别标签从0开始，所以需要减1进行调整。
        else:
            loss_sem = pred_sem.sum() * 0.0

        # ── depth loss ──depth是指渲染结果的深度图，pseudo_depth是指根据点云生成的伪标签深度图。rendered_depth和pseudo_depth都是(B, nC, H, W)的形状，其中nC是相机数量，H和W是图像的高和宽。depth loss是计算rendered_depth和pseudo_depth之间的均方误差损失，注意pseudo_depth中的0表示无效像素，不参与损失计算。
        pred_d = rendered_depth.flatten()
        target_d = pseudo_depth.flatten()
        dyn_mask = torch.isin(
            pseudo_seg.flatten(),
            self.dynamic_classes.to(pseudo_seg.device)
        )
        valid_d = (target_d > 0.5) & ~dyn_mask
        if valid_d.any():
            loss_depth = self.depth_lw * self.loss_fn_depth(
                pred_d[valid_d], target_d[valid_d]
            )
        else:
            loss_depth = pred_d.sum() * 0.0

        # ── diagnostics（每 vis_every iter 打印一次，并保存渲染可视化图片）──
        self._diag_counter = getattr(self, '_diag_counter', 0) + 1
        if self._diag_counter % self.vis_every == 1:
            valid_sem_ratio = valid_sem.float().mean().item()
            valid_d_ratio   = valid_d.float().mean().item()
            pred_depth_mean = pred_d[valid_d].mean().item() if valid_d.any() else 0.0
            pred_depth_std  = pred_d[valid_d].std().item()  if valid_d.any() else 0.0
            gt_depth_mean   = target_d[valid_d].mean().item() if valid_d.any() else 0.0
            with torch.no_grad():
                prob = torch.softmax(pred_sem[valid_sem], dim=-1) if valid_sem.any() else None
                pred_entropy = (-(prob * (prob + 1e-8).log()).sum(-1)).mean().item() if prob is not None else float('nan')
                rand_entropy = math.log(17)   # 随机猜测基准 ≈ 2.833
            logger = logging.getLogger('mmengine')
            logger.info(
                f'[RenderLoss Diag] iter={self._diag_counter} | '
                f'valid_sem={valid_sem_ratio:.2%} valid_depth={valid_d_ratio:.2%} | '
                f'pred_depth: mean={pred_depth_mean:.2f}m std={pred_depth_std:.2f}m  '
                f'gt_depth_mean={gt_depth_mean:.2f}m | '
                f'sem_entropy={pred_entropy:.3f} (rand={rand_entropy:.3f}) | '
                f'loss_sem={loss_sem.item():.4f} loss_depth={loss_depth.item():.4f}'
            )
            # 保存渲染可视化图片
            if self.vis_dir is not None:
                self._save_vis(rendered_sem, rendered_depth, pseudo_seg, pseudo_depth,
                               step=self._diag_counter)

        return loss_sem + loss_depth

    def _save_vis(self, rendered_sem, rendered_depth, pseudo_seg, pseudo_depth, step):
        """
        保存所有相机的渲染结果对比图（batch 0）。
        每张图为横向拼接: [pred_sem | gt_sem | pred_depth | gt_depth]
        所有相机纵向堆叠，保存为 render_vis/step_{step:06d}.jpg
        """
        try:
            from PIL import Image
        except ImportError:
            return
        try:
            os.makedirs(self.vis_dir, exist_ok=True)
            B, nC, H, W, _ = rendered_sem.shape
            rows = []
            for cam in range(nC):
                # 语义：渲染预测 argmax（0-indexed）
                pred_cls = rendered_sem[0, cam].detach().cpu().argmax(dim=-1).numpy()  # (H, W)
                pred_sem_rgb = _colorize_sem(pred_cls)  # (H, W, 3)

                # 语义：伪标签 GT（1-indexed, 0=invalid）→ 0-indexed for palette
                gt_cls_raw = pseudo_seg[0, cam].detach().cpu().numpy().astype(np.int32)  # (H, W)
                gt_sem_rgb = np.where(
                    (gt_cls_raw[..., None] > 0),
                    _NUSC_PALETTE[(np.clip(gt_cls_raw, 1, 17) - 1)],
                    np.array([128, 128, 128], dtype=np.uint8)   # invalid → 灰色
                ).astype(np.uint8)

                # 深度：渲染预测
                pred_d_np = rendered_depth[0, cam].detach().cpu().numpy()  # (H, W)
                pred_d_rgb = _depth_to_rgb(pred_d_np)  # (H, W, 3)

                # 深度：伪标签 GT
                gt_d_np = pseudo_depth[0, cam].detach().cpu().numpy()  # (H, W)
                gt_d_rgb = _depth_to_rgb(gt_d_np)  # (H, W, 3)

                # 添加标签栏（在图片顶部写相机编号，用黑色像素行分隔）
                separator = np.zeros((2, W * 4, 3), dtype=np.uint8)
                row = np.concatenate([pred_sem_rgb, gt_sem_rgb, pred_d_rgb, gt_d_rgb], axis=1)
                rows.append(separator)
                rows.append(row)

            combined = np.concatenate(rows, axis=0)
            out_path = os.path.join(self.vis_dir, f'step_{step:06d}.jpg')
            Image.fromarray(combined).save(out_path, quality=90)
        except Exception as e:
            logging.getLogger('mmengine').warning(f'[RenderLoss] vis save failed: {e}')
