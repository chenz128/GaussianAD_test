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


def _depth_to_rgb(depth_np, vmin=0.0, vmax=30.0):
    """depth_np: (H, W) float → RGB (H, W, 3) using a jet colormap.

    vmax=30 matches the Gaussian geometric range (pc_range ±30m): rendered
    depth physically cannot exceed ~30m, so anchoring the warm end at 30m
    spreads the full color gradient over the depth range actually present.

    The previous map flattened 10-30m to a single saturated green (g==1 for
    norm in [0.25, 0.75]) — no gradient where most rendered pixels live, hence
    the "all green" look. jet keeps a continuous blue→cyan→green→yellow→red
    ramp across the whole range. invalid pixels (depth<=0, e.g. sky / no
    Gaussian coverage) → gray.
    """
    norm = np.clip((depth_np - vmin) / (vmax - vmin + 1e-6), 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * norm - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * norm - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * norm - 1.0), 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    # invalid pixels (depth==0) → gray
    rgb[depth_np <= 0] = 0.5
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
        extra_depth_lw=0.1,
        acc_lw=0.0,
        acc_tau_fg=0.8,
        acc_tau_sky=0.2,
        vis_dir=None,
        vis_every=500,
        input_dict=None,
        **kwargs,
    ):
        if input_dict is None:
            input_dict = {
                'rendered_sem': 'rendered_sem',
                'rendered_depth': 'rendered_depth',
                'rendered_acc': 'rendered_acc',
                'pseudo_seg': 'pseudo_seg',
                'pseudo_depth': 'pseudo_depth',
                'rendered_extra_depth': 'rendered_extra_depth',
                'extra_pseudo_depth': 'extra_pseudo_depth',
                'extra_valid': 'extra_valid',
                'input_imgs': 'input_imgs',
                'aug_flip': 'aug_flip',
            }
        super().__init__(weight=weight, input_dict=input_dict, **kwargs)#这里调用了BaseLoss的__init__方法，设置了self.weight、self.input_dict、self.loss_func等属性。注意BaseLoss.__init__中将self.loss_func初始化为lambda: 0，这会覆盖我们下面定义的loss_func方法，因此我们需要删除这个实例属性来恢复方法的正常调用。
        # BaseLoss.__init__ sets self.loss_func = lambda: 0 as instance attr,
        # which shadows our loss_func method. Delete it to restore method lookup.
        del self.loss_func

        self.sem_lw = sem_lw
        self.depth_lw = depth_lw
        self.extra_depth_lw = extra_depth_lw
        # A1 accumulation loss: hinge-softened opacity prior.
        # acc_lw=0 disables it (default). acc_tau_fg: foreground only penalized
        # when A < tau_fg; acc_tau_sky: sky only penalized when A > tau_sky.
        self.acc_lw = acc_lw
        self.acc_tau_fg = acc_tau_fg
        self.acc_tau_sky = acc_tau_sky
        self.vis_dir = vis_dir
        self.vis_every = vis_every


        # class frequency weights for balanced CE loss (nuScenes 17 classes)
        nusc_class_freq = torch.tensor([
            944004, 1897170, 152386, 2391677, 16957802, 724139,
            189027, 2074468, 413451, 2384460, 5916653, 175883646,
            4275424, 51393615, 61411620, 105975596, 116424404
        ], dtype=torch.float32)#这是根据nuScenes数据集的统计得到的每个类别的像素频率。我们使用这些频率来计算类别权重，以平衡交叉熵损失中不同类别的重要性。频率较低的类别会得到较高的权重，频率较高的类别会得到较低的权重，从而避免模型过度关注频率较高的类别。
        log_w = torch.log(nusc_class_freq.sum() / nusc_class_freq)
        self.register_buffer('class_weight', log_w / log_w.mean())

        self.loss_fn_ce = nn.CrossEntropyLoss(reduction='none')#交叉熵损失函数，reduction='none'表示不对损失进行平均或求和，而是返回每个样本的损失值。这是因为我们需要对有效像素应用类别权重后再进行平均。
        # Huber loss is more robust than MSE for depth: quadratic for small errors,
        # linear for large errors. delta=2m is a reasonable threshold for outdoor scenes.
        self.loss_fn_depth = nn.HuberLoss(delta=2.0, reduction='mean')#Huber损失函数，也称为平滑L1损失函数。对于小误差（绝对值小于delta），它与MSE相同，计算平方误差；对于大误差（绝对值大于delta），它计算线性误差，即绝对值减去delta。这使得Huber损失对异常值更鲁棒，因为它不会像MSE那样对大误差产生过大的梯度。delta=2.0表示我们认为2米以内的深度误差是合理的，而超过2米的误差可能是异常值，我们希望降低它们的影响。

    def forward(self, inputs):
        """Override BaseLoss.forward to return (total, sub_dict) for separate logging."""
        actual_inputs = {}#这里我们根据self.input_dict中定义的键值对，从inputs字典中提取对应的输入张量，并存储在actual_inputs字典中。这样做的目的是为了将输入张量的名称与loss_func方法中的参数名称进行映射，使得我们可以灵活地指定输入张量的来源，而不需要在loss_func方法中硬编码输入张量的名称。
        for input_key, input_val in self.input_dict.items():
            actual_inputs.update({input_key: inputs.get(input_val)})
        loss_sem, loss_depth, loss_extra_depth, loss_acc = self.loss_func(**actual_inputs)
        total = self.weight * (loss_sem + loss_depth + loss_extra_depth + loss_acc)
        return total, {
            'RenderSemLoss': (self.weight * loss_sem).detach().item(),
            'RenderDepthLoss': (self.weight * loss_depth).detach().item(),
            'RenderExtraDepthLoss': (self.weight * loss_extra_depth).detach().item(),
            'RenderAccLoss': (self.weight * loss_acc).detach().item(),
        }

    def _extra_depth_loss(self, rendered_extra_depth, extra_pseudo_depth, extra_valid):
        """Multi-frame (history + future) depth Huber loss.

        rendered_extra_depth: (B, K, nC, H, W)  — rendered depth per extra frame
        extra_pseudo_depth:   (B, K, nC, H, W)  — pseudo depth (0=invalid; for
                                                  history frames dynamic pixels
                                                  are already zeroed)
        extra_valid:          (B, K) bool       — frame existence flag
        """
        if rendered_extra_depth is None or extra_pseudo_depth is None:
            return None
        pred = rendered_extra_depth
        tgt = extra_pseudo_depth.to(pred.device).float()
        valid_d = (tgt > 0.5) & (pred.detach() > 0)
        if extra_valid is not None:
            vmask = extra_valid.to(pred.device).bool().view(
                extra_valid.shape[0], extra_valid.shape[1], 1, 1, 1)
            valid_d = valid_d & vmask
        if valid_d.any():
            return self.extra_depth_lw * self.loss_fn_depth(pred[valid_d], tgt[valid_d])
        return pred.sum() * 0.0

    def _acc_loss(self, rendered_acc, pseudo_seg, pseudo_depth):
        """A1 accumulation loss: a hinge-softened opacity prior on the rendered
        accumulation map A = 1 - T_final (per-pixel total opacity along the ray).

        Target A* is CONSTRUCTED from existing pseudo labels (no external data):
          - foreground (pseudo_depth > 0.5): a real surface exists -> A should -> 1
          - sky/empty  (pseudo_seg == 0 AND pseudo_depth == 0): nothing there -> A -> 0
          - middle ground (seg > 0 but depth == 0, e.g. far buildings > 40m where
            Metric3D is unreliable): NOT constrained (skipped) -- we don't know.

        Hinge softening avoids forcing exact binarization (which fights legit
        thin/semi-transparent surfaces):
          - foreground: penalize only when A < tau_fg  -> relu(tau_fg - A)
          - sky:        penalize only when A > tau_sky  -> relu(A - tau_sky)

        Effect: kills low-alpha floaters scattered in empty space (sky gets
        non-zero A) and pushes foggy foreground surfaces to commit opacity,
        WITHOUT touching scale or directly supervising occluded gaussians
        (occluded gaussians have T_i~0 so their gradient ~0).
        """
        if rendered_acc is None or pseudo_seg is None or pseudo_depth is None:
            return None
        acc = rendered_acc.flatten().clamp(0.0, 1.0)
        seg = pseudo_seg.flatten().long()
        dep = pseudo_depth.flatten()

        fg_mask = dep > 0.5
        sky_mask = (seg == 0) & (dep <= 0.0)

        terms = []
        if fg_mask.any():
            terms.append(F.relu(self.acc_tau_fg - acc[fg_mask]).mean())
        if sky_mask.any():
            terms.append(F.relu(acc[sky_mask] - self.acc_tau_sky).mean())
        if not terms:
            return acc.sum() * 0.0
        return self.acc_lw * torch.stack(terms).sum()

    def loss_func(self, rendered_sem, rendered_depth, pseudo_seg, pseudo_depth,
                  rendered_acc=None,
                  rendered_extra_depth=None, extra_pseudo_depth=None, extra_valid=None,
                  input_imgs=None, aug_flip=None):
        """
        Args:
            rendered_sem:   (B, nC, H, W, 17) — rendered semantic logits
            rendered_depth: (B, nC, H, W)     — rendered depth
            rendered_acc:   (B, nC, H, W)     — rendered accumulation map A (A1)
            pseudo_seg:     (B, nC, H, W)     — pseudo semantic labels (0=invalid)
            pseudo_depth:   (B, nC, H, W)     — pseudo depth (0=invalid)
            input_imgs:     (B, F, N, C, H, W) — original camera images (optional, for vis)
            aug_flip:       bool or None       — whether input image was horizontally flipped
        """
        # eval mode: rendering was skipped, or val dataset has no pseudo labels
        if rendered_sem is None or rendered_depth is None or pseudo_seg is None or pseudo_depth is None:
            zero = torch.tensor(0.0, requires_grad=False)
            return zero, zero, zero, zero

        # ── semantic loss ──semantic是指渲染结果的语义分割图，pseudo_seg是指根据点云生成的伪标签语义分割图。rendered_sem和pseudo_seg都是(B, nC, H, W)的形状，其中nC是相机数量，H和W是图像的高和宽。semantic loss是计算rendered_sem和pseudo_seg之间的交叉熵损失，注意pseudo_seg中的0表示无效像素，不参与损失计算。
        pred_sem = rendered_sem.flatten(0, -2)     # (N, 17)
        target_sem = pseudo_seg.flatten().long()    # (N,)
        valid_sem = target_sem > 0
        if valid_sem.any():#如果存在有效的语义标签像素，我们首先根据target_sem中的类别索引，从self.class_weight中获取对应的类别权重pw。由于pseudo_seg中的标签是从1到17的，而self.class_weight是从0到16的，所以我们需要使用target_sem[valid_sem]作为索引来获取权重。然后，我们计算交叉熵损失self.loss_fn_ce(pred_sem[valid_sem], target_sem[valid_sem])，并乘以类别权重pw和语义损失权重self.sem_lw，最后对所有有效像素的损失进行平均，得到最终的语义损失loss_sem。
            pw = self.class_weight[target_sem[valid_sem]]
            # pseudo_seg labels 1-16 directly correspond to Gaussian semantic channels 1-16
            # (channel 0 = noise/others in OccupancyLoss, not supervised by RenderLoss)
            # Do NOT subtract 1: the CE target must match the OccupancyLoss class indices
            loss_sem = self.sem_lw * (
                pw * self.loss_fn_ce(pred_sem[valid_sem], target_sem[valid_sem])
            ).mean()
        else:
            loss_sem = pred_sem.sum() * 0.0

        # ── depth loss ──depth是指渲染结果的深度图，pseudo_depth是指根据点云生成的伪标签深度图。rendered_depth和pseudo_depth都是(B, nC, H, W)的形状，其中nC是相机数量，H和W是图像的高和宽。depth loss是计算rendered_depth和pseudo_depth之间的均方误差损失，注意pseudo_depth中的0表示无效像素，不参与损失计算。
        pred_d = rendered_depth.flatten()
        target_d = pseudo_depth.flatten()
        # Only exclude pixels with no valid GT depth and pixels where Gaussians
        # haven't rendered anything (pred_d=0 → spurious MSE gradient).
        # Dynamic classes are NOT excluded: Metric3D is single-frame so moving
        # object depth is valid for the current frame. GaussianFlowOcc does the same.
        valid_d = (target_d > 0.5) & (pred_d.detach() > 0)
        if valid_d.any():
            loss_depth = self.depth_lw * self.loss_fn_depth(
                pred_d[valid_d], target_d[valid_d]
            )
        else:
            loss_depth = pred_d.sum() * 0.0

        # ── multi-frame (history + future) depth loss ──
        loss_extra_depth = self._extra_depth_loss(
            rendered_extra_depth, extra_pseudo_depth, extra_valid)
        if loss_extra_depth is None:
            loss_extra_depth = loss_depth.new_zeros(())

        # ── A1 accumulation loss (current frame only) ──
        loss_acc = None
        if self.acc_lw > 0:
            loss_acc = self._acc_loss(rendered_acc, pseudo_seg, pseudo_depth)
        if loss_acc is None:
            loss_acc = loss_depth.new_zeros(())

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
                # A1 accumulation diagnostics
                if self.acc_lw > 0 and rendered_acc is not None:
                    acc_flat = rendered_acc.flatten().clamp(0.0, 1.0)
                    seg_flat = pseudo_seg.flatten().long()
                    dep_flat = pseudo_depth.flatten()
                    fg_m = dep_flat > 0.5
                    sky_m = (seg_flat == 0) & (dep_flat <= 0.0)
                    acc_fg_mean = acc_flat[fg_m].mean().item() if fg_m.any() else float('nan')
                    acc_sky_mean = acc_flat[sky_m].mean().item() if sky_m.any() else float('nan')
                else:
                    acc_fg_mean = acc_sky_mean = float('nan')
            logger = logging.getLogger('mmengine')
            logger.info(
                f'[RenderLoss Diag] iter={self._diag_counter} | '
                f'valid_sem={valid_sem_ratio:.2%} valid_depth={valid_d_ratio:.2%} | '
                f'pred_depth: mean={pred_depth_mean:.2f}m std={pred_depth_std:.2f}m  '
                f'gt_depth_mean={gt_depth_mean:.2f}m | '
                f'sem_entropy={pred_entropy:.3f} (rand={rand_entropy:.3f}) | '
                f'acc_fg={acc_fg_mean:.3f} acc_sky={acc_sky_mean:.3f} | '
                f'loss_sem={loss_sem.item():.4f} loss_depth={loss_depth.item():.4f} '
                f'loss_extra_depth={loss_extra_depth.item():.4f} loss_acc={loss_acc.item():.4f}'
            )
            # 保存渲染可视化图片
            if self.vis_dir is not None:
                self._save_vis(rendered_sem, rendered_depth, pseudo_seg, pseudo_depth,
                               step=self._diag_counter, input_imgs=input_imgs, aug_flip=aug_flip)
                # 多帧（历史+未来）深度可视化
                if rendered_extra_depth is not None and extra_pseudo_depth is not None:
                    self._save_extra_vis(rendered_extra_depth, extra_pseudo_depth,
                                         extra_valid, step=self._diag_counter)

        return loss_sem, loss_depth, loss_extra_depth, loss_acc

    def _save_vis(self, rendered_sem, rendered_depth, pseudo_seg, pseudo_depth, step, input_imgs=None, aug_flip=None):
        """
        保存所有相机的渲染结果对比图（batch 0）。
        每张图为横向拼接: [pred_sem | gt_sem | pred_depth | gt_depth | orig_img]
        所有相机纵向堆叠，保存为 render_vis/step_{step:06d}.jpg

        Alignment: rendered/pseudo are in ORIGINAL camera orientation (never flipped).
        input_imgs may be flipped by augmentation — we un-flip it here for display.
        input_imgs covers a different vertical region (top 36px crop) vs pseudo labels
        (top ~318px crop in original space), so we crop input_imgs to match.
        """
        # DDP: only rank 0 saves to avoid 8 processes writing the same file concurrently
        import torch.distributed as dist
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        try:
            from PIL import Image
        except ImportError:
            return
        try:
            os.makedirs(self.vis_dir, exist_ok=True)
            B, nC, H, W, _ = rendered_sem.shape
            rows = []
            for cam in range(nC):
                # 语义：渲染预测 argmax — model class 0=noise, 1=barrier, ..., 16=vegetation
                pred_cls = rendered_sem[0, cam].detach().cpu().argmax(dim=-1).numpy()  # (H, W)
                pred_sem_rgb = np.where(
                    (pred_cls[..., None] > 0),
                    _NUSC_PALETTE[np.clip(pred_cls - 1, 0, 16)],
                    np.array([128, 128, 128], dtype=np.uint8)   # class 0 (noise) → gray
                ).astype(np.uint8)

                # 语义：伪标签 GT — pseudo_seg 0=invalid, 1=barrier, ..., 16=vegetation
                gt_cls_raw = pseudo_seg[0, cam].detach().cpu().numpy().astype(np.int32)  # (H, W)
                gt_sem_rgb = np.where(
                    (gt_cls_raw[..., None] > 0),
                    _NUSC_PALETTE[np.clip(gt_cls_raw - 1, 0, 16)],
                    np.array([128, 128, 128], dtype=np.uint8)   # invalid → gray
                ).astype(np.uint8)

                # 深度：渲染预测
                pred_d_np = rendered_depth[0, cam].detach().cpu().numpy()  # (H, W)
                pred_d_rgb = _depth_to_rgb(pred_d_np)  # (H, W, 3)

                # 深度：伪标签 GT
                gt_d_np = pseudo_depth[0, cam].detach().cpu().numpy()  # (H, W)
                gt_d_rgb = _depth_to_rgb(gt_d_np)  # (H, W, 3)

                # 原始相机图像（当前帧，batch 0）
                orig_img_rgb = None
                if input_imgs is not None:
                    try:
                        # input_imgs: (B, F, N, C, H_img, W_img) — augmented (may be flipped)
                        # Frame ordering: imgs[:, 0] = current frame t, imgs[:, -1] = oldest t-(F-1).
                        # Pseudo labels are for current frame t, so always use index 0.
                        img_t = input_imgs[0, 0, cam].detach().cpu().float().numpy()  # (C, H_img, W_img)
                        img_t = img_t.transpose(1, 2, 0)  # (H_img, W_img, C)
                        # un-normalize: ImageNet mean/std, RGB
                        img_t = img_t * np.array([58.395, 57.12, 57.375], dtype=np.float32) \
                                      + np.array([123.675, 116.28, 103.53], dtype=np.float32)
                        img_t = np.clip(img_t, 0, 255).astype(np.uint8)
                        # Un-flip: rendering/pseudo labels are in original orientation
                        flip_val = False
                        if aug_flip is not None:
                            flip_val = aug_flip.item() if hasattr(aug_flip, 'item') else bool(aug_flip)
                        if flip_val:
                            img_t = img_t[:, ::-1, :].copy()
                        # Crop to match pseudo label region:
                        # input_imgs covers original rows 36-899 (H_img=864 pixels)
                        # pseudo labels cover original rows ~318-899 (after 0.44x + crop_top=140)
                        # Crop top portion of input_imgs to show same region
                        H_img = img_t.shape[0]  # 864
                        # original row 318 maps to input_imgs row (318-36) = 282
                        crop_start = int((318 - 36) / (900 - 36) * H_img)  # ~282
                        img_t = img_t[crop_start:, :, :]
                        orig_img_pil = Image.fromarray(img_t).resize((W, H), Image.BILINEAR)
                        orig_img_rgb = np.array(orig_img_pil)
                    except Exception:
                        pass

                # 用黑色像素行分隔相机
                n_cols = 5 if orig_img_rgb is not None else 4
                separator = np.zeros((2, W * n_cols, 3), dtype=np.uint8)
                cols = [pred_sem_rgb, gt_sem_rgb, pred_d_rgb, gt_d_rgb]
                if orig_img_rgb is not None:
                    cols.append(orig_img_rgb)
                row = np.concatenate(cols, axis=1)
                rows.append(separator)
                rows.append(row)

            combined = np.concatenate(rows, axis=0)
            out_path = os.path.join(self.vis_dir, f'step_{step:06d}.jpg')
            Image.fromarray(combined).save(out_path, quality=90)
        except Exception as e:
            logging.getLogger('mmengine').warning(f'[RenderLoss] vis save failed: {e}')

    def _save_extra_vis(self, rendered_extra_depth, extra_pseudo_depth, extra_valid, step):
        """保存多帧（历史+未来）深度渲染对比图（batch 0）。

        rendered_extra_depth: (B, K, nC, H, W)
        extra_pseudo_depth:   (B, K, nC, H, W)
        extra_valid:          (B, K) bool 或 None

        帧顺序由 dataset 固定：[hist t-1, t-2, ..., fut t+1, t+2, ...]。
        每相机一行，横向拼接 K 帧 × [pred_depth | gt_depth]，
        所有相机纵向堆叠，保存为 render_vis/step_{step:06d}_extra.jpg。
        无效帧（extra_valid=False）深度全 0 → 显示全灰。
        """
        import torch.distributed as dist
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        try:
            from PIL import Image
        except ImportError:
            return
        try:
            os.makedirs(self.vis_dir, exist_ok=True)
            B, K, nC, H, W = rendered_extra_depth.shape
            valid_cpu = extra_valid[0].detach().cpu().bool().numpy() if extra_valid is not None \
                else np.ones(K, dtype=bool)
            rows = []
            for cam in range(nC):
                cols = []
                for k in range(K):
                    pred_d_np = rendered_extra_depth[0, k, cam].detach().cpu().numpy()  # (H, W)
                    gt_d_np   = extra_pseudo_depth[0, k, cam].detach().cpu().numpy()    # (H, W)
                    if not bool(valid_cpu[k]):
                        # 无效帧：置 0 → _depth_to_rgb 显示为灰色
                        pred_d_np = np.zeros_like(pred_d_np)
                        gt_d_np   = np.zeros_like(gt_d_np)
                    cols.append(_depth_to_rgb(pred_d_np))
                    cols.append(_depth_to_rgb(gt_d_np))
                separator = np.zeros((2, W * len(cols), 3), dtype=np.uint8)
                row = np.concatenate(cols, axis=1)
                rows.append(separator)
                rows.append(row)

            combined = np.concatenate(rows, axis=0)
            out_path = os.path.join(self.vis_dir, f'step_{step:06d}_extra.jpg')
            Image.fromarray(combined).save(out_path, quality=90)
        except Exception as e:
            logging.getLogger('mmengine').warning(f'[RenderLoss] extra vis save failed: {e}')
