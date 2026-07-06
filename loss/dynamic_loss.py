import os
import logging
import torch
import torch.nn as nn
import numpy as np
from . import OPENOCC_LOSS
from .base_loss import BaseLoss

try:
    from model.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu
except Exception:  # pragma: no cover - only needed when use_gt_box=True
    points_in_boxes_gpu = None


def _dyn_prob_to_rgb(prob):
    """prob: (H, W) float in [0,1] → RGB (H, W, 3) uint8.
    蓝(静态 p=0) → 紫 → 红(动态 p=1)，连续热图直观看预测置信。"""
    p = np.clip(prob, 0.0, 1.0)
    r = (p * 255).astype(np.uint8)
    g = np.zeros_like(r)
    b = ((1.0 - p) * 255).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


def _dyn_gt_to_rgb(gt):
    """gt: (H, W) int {0=ignore,1=static,2=dynamic} → RGB (H, W, 3) uint8."""
    rgb = np.empty((*gt.shape, 3), dtype=np.uint8)
    rgb[gt == 0] = (128, 128, 128)   # ignore → gray
    rgb[gt == 1] = (40, 80, 220)     # static → blue
    rgb[gt == 2] = (220, 40, 40)     # dynamic → red
    return rgb


@OPENOCC_LOSS.register_module()
class DynamicLoss(BaseLoss):
    """
    Dynamic/static supervision via 2D Gaussian splatting renders.

    Renders a per-gaussian dynamic logit to 2D (rendered_dynamic) and supervises
    it against an offline-generated, ego-speed-gated LiDAR dynamic GT mask
    (pseudo_dyn). Mask convention:
        0 = ignore (no loss)
        1 = static
        2 = dynamic
    Binary cross-entropy is applied only on labeled pixels (pseudo_dyn > 0),
    with target = (pseudo_dyn == 2).
    """

    def __init__(
        self,
        weight=1.0,
        pos_weight=30.0,
        extra_weight=0.3,
        vis_dir=None,
        vis_every=500,
        use_gt_box=False,
        v_thresh=0.5,
        z_margin=0.2,
        input_dict=None,
        **kwargs,
    ):
        if input_dict is None:
            if use_gt_box:
                # oracle mode: supervise per-gaussian dynamic_logits directly in
                # 3D via point-in-box, bypassing 2D render and noisy pseudo_dyn.
                input_dict = {
                    'dynamic_logits': 'dynamic_logits',
                    'gaussian': 'gaussian',
                    'gt_boxes': 'gt_boxes',
                }
            else:
                input_dict = {
                    'rendered_dynamic': 'rendered_dynamic',
                    'pseudo_dyn': 'pseudo_dyn',
                    'input_imgs': 'input_imgs',
                    'aug_flip': 'aug_flip',
                    'rendered_extra_dynamic': 'rendered_extra_dynamic',
                    'extra_pseudo_dyn': 'extra_pseudo_dyn',
                    'extra_dyn_valid': 'extra_dyn_valid',
                }
        super().__init__(weight=weight, input_dict=input_dict, **kwargs)
        # BaseLoss.__init__ sets self.loss_func = lambda: 0 as instance attr,
        # which shadows our loss_func method. Delete it to restore method lookup.
        del self.loss_func

        self.vis_dir = vis_dir
        self.vis_every = vis_every
        self.use_gt_box = use_gt_box
        self.v_thresh = v_thresh
        if use_gt_box:
            assert points_in_boxes_gpu is not None, \
                'points_in_boxes_gpu unavailable but use_gt_box=True'
        # Ground gate: a moving box's floor slice also encloses static ground
        # voxels (road/terrain). Only the thin bottom layer (< z_margin above the
        # box floor) is forced static; the object body is kept. z_margin is small
        # so genuine dynamic gaussians are never dropped. 0 -> gate disabled.
        self.z_margin = z_margin
        # dynamic pixels are rare (~1.6% of labeled px) → up-weight positive class
        self.register_buffer('pos_weight', torch.tensor(float(pos_weight)))
        self.extra_weight = extra_weight

    @torch.no_grad()
    def _gt_box_membership(self, means, gt_boxes):
        """Assign each gaussian to a GT box and derive a dynamic mask.

        Args:
            means:     (B, G, 3) gaussian centers in LIDAR frame
            gt_boxes:  (B, T, >=9) padded GT boxes (pad rows all-zero, never hit)
        Returns:
            dyn_mask: (B, G) bool, True if inside a moving box (|v| > v_thresh)
                      AND above the box-floor ground slice (height > z_margin).
        """
        means = means.detach().float().contiguous()
        gt_boxes = gt_boxes.to(means.device).float()
        boxes7 = gt_boxes[..., :7].contiguous()          # (B, T, 7)
        box_idx = points_in_boxes_gpu(means, boxes7).long()  # (B, G), -1 bg
        speed = torch.linalg.norm(gt_boxes[..., 7:9], dim=-1)  # (B, T)
        moving_box = speed > self.v_thresh                     # (B, T) bool
        B, G = box_idx.shape
        dyn_mask = torch.zeros((B, G), dtype=torch.bool, device=means.device)
        for b in range(B):
            valid = box_idx[b] >= 0
            if valid.any():
                dyn_mask[b, valid] = moving_box[b][box_idx[b, valid]]
        # ground gate: heading is yaw-only (rotation about z), so z needs no
        # un-rotation. Force gaussians in the box's bottom slice (ground) static.
        if self.z_margin > 0:
            box_bottom = gt_boxes[..., 2] - 0.5 * gt_boxes[..., 5]  # (B, T)
            for b in range(B):
                sel = box_idx[b] >= 0
                if not sel.any():
                    continue
                bi = box_idx[b].clamp_min(0)                       # (G,)
                h_above = means[b, :, 2] - box_bottom[b][bi]       # (G,)
                ground = sel & (h_above < self.z_margin)
                dyn_mask[b, ground] = False
        return dyn_mask

    def forward(self, inputs):
        actual_inputs = {}
        for input_key, input_val in self.input_dict.items():
            # .get(): multi-frame keys are absent unless the config maps them,
            # so missing keys resolve to None and the loss skips that branch.
            actual_inputs.update({input_key: inputs.get(input_val)})
        if self.use_gt_box:
            loss = self._gt_box_loss_func(**actual_inputs)
        else:
            loss = self.loss_func(**actual_inputs)
        return self.weight * loss, {
            'DynamicLoss': (self.weight * loss).detach().item(),
        }

    def _gt_box_loss_func(self, dynamic_logits, gaussian=None, gt_boxes=None):
        """Direct 3D supervision of per-gaussian dynamic_logits via GT boxes.

        A gaussian is labeled dynamic (target=1) iff its center falls inside a
        GT box moving faster than ``v_thresh``; everything else (static boxes and
        background) is static (target=0). Background=static is a safe assumption
        for nuScenes (buildings/road/vegetation). Eval passes None → returns 0.
        """
        if dynamic_logits is None or gaussian is None or gt_boxes is None:
            return torch.tensor(0.0, requires_grad=False)

        dyn_mask = self._gt_box_membership(
            gaussian.means, gt_boxes)  # (B, G) bool
        logit = dynamic_logits
        if logit.dim() == 3:
            logit = logit[..., 0]                        # (B, G, 1) -> (B, G)
        pred_v = logit.flatten()                         # (N,)
        target_v = dyn_mask.flatten().float()            # (N,)
        loss = nn.functional.binary_cross_entropy_with_logits(
            pred_v, target_v, pos_weight=self.pos_weight.to(pred_v.device)
        )

        # ── diagnostics ──
        self._diag_counter = getattr(self, '_diag_counter', 0) + 1
        if self._diag_counter % self.vis_every == 1:
            with torch.no_grad():
                n_total = target_v.numel()
                n_dyn = int(target_v.sum().item())
                prob = torch.sigmoid(pred_v)
                pred_dyn_ratio = (prob > 0.5).float().mean().item()
            logging.getLogger('mmengine').info(
                f'[DynamicLoss Diag] iter={self._diag_counter} | gt_box=True | '
                f'gaussians={n_total} dyn_gt={n_dyn} ({n_dyn / max(n_total, 1):.2%}) | '
                f'pred_dyn_ratio={pred_dyn_ratio:.2%} | loss={loss.item():.4f}'
            )
        return loss


    def loss_func(self, rendered_dynamic, pseudo_dyn, input_imgs=None, aug_flip=None,
                  rendered_extra_dynamic=None, extra_pseudo_dyn=None, extra_dyn_valid=None):
        """
        Args:
            rendered_dynamic: (B, nC, H, W) raw logits, or None (eval)
            pseudo_dyn:       (B, nC, H, W) int, 0=ignore/1=static/2=dynamic, or None
            input_imgs:       (B, F, N, C, H, W) original cam images (optional, for vis)
            aug_flip:         bool or None — whether input image was horizontally flipped
            rendered_extra_dynamic: (B, K, nC, H, W) multi-frame logits, or None
            extra_pseudo_dyn:       (B, K, nC, H, W) multi-frame dynamic GT, or None
            extra_dyn_valid:        (B, K) bool — which extra frames are usable, or None
        """
        if rendered_dynamic is None or pseudo_dyn is None:
            return torch.tensor(0.0, requires_grad=False)

        pred = rendered_dynamic.flatten()            # (N,)
        gt = pseudo_dyn.flatten().long()             # (N,) 
        valid = gt > 0
        if not valid.any():
            return pred.sum() * 0.0

        pred_v = pred[valid]
        target_v = (gt[valid] == 2).float()
        loss = nn.functional.binary_cross_entropy_with_logits(
            pred_v, target_v, pos_weight=self.pos_weight.to(pred_v.device)
        )

        # ── multi-frame (history + future) supervision ──
        extra_loss_val = 0.0
        if rendered_extra_dynamic is not None and extra_pseudo_dyn is not None:
            extra = self._extra_dynamic_loss(
                rendered_extra_dynamic, extra_pseudo_dyn, extra_dyn_valid)
            loss = loss + self.extra_weight * extra
            extra_loss_val = float(extra.detach().item())

        # ── diagnostics ──
        self._diag_counter = getattr(self, '_diag_counter', 0) + 1
        if self._diag_counter % self.vis_every == 1:
            with torch.no_grad():
                n_valid = int(valid.sum().item())
                n_dyn = int(target_v.sum().item())
                prob = torch.sigmoid(pred_v)
                pred_dyn_ratio = (prob > 0.5).float().mean().item()
            logging.getLogger('mmengine').info(
                f'[DynamicLoss Diag] iter={self._diag_counter} | '
                f'valid_px={n_valid} dyn_gt={n_dyn} ({n_dyn / max(n_valid, 1):.2%}) | '
                f'pred_dyn_ratio={pred_dyn_ratio:.2%} | loss={loss.item():.4f}'
                + (f' | extra_loss={extra_loss_val:.4f}' if extra_loss_val else '')
            )
            if self.vis_dir is not None:
                self._save_vis(rendered_dynamic, pseudo_dyn, step=self._diag_counter,
                               input_imgs=input_imgs, aug_flip=aug_flip)
        return loss

    def _extra_dynamic_loss(self, rendered_extra, extra_pseudo, extra_valid):
        """BCE on multi-frame rendered dynamic logits vs adjacent-frame dynamic GT.

        Args:
            rendered_extra: (B, K, nC, H, W) raw logits
            extra_pseudo:   (B, K, nC, H, W) int, 0=ignore/1=static/2=dynamic
            extra_valid:    (B, K) bool, or None
        History frames contribute only static (gt==2 already masked to 0 in the
        dataset -> all negatives); future frames add rare dynamic positives.
        """
        B, K = rendered_extra.shape[:2]
        dev = rendered_extra.device
        pred_list, tgt_list = [], []
        for b in range(B):
            for k in range(K):
                if extra_valid is not None and not bool(extra_valid[b, k]):
                    continue
                pred = rendered_extra[b, k].flatten()
                gt = extra_pseudo[b, k].flatten().long().to(dev)
                m = gt > 0
                if not m.any():
                    continue
                pred_list.append(pred[m])
                tgt_list.append((gt[m] == 2).float())
        if not pred_list:
            return rendered_extra.sum() * 0.0
        pred_v = torch.cat(pred_list)
        tgt_v = torch.cat(tgt_list)
        return nn.functional.binary_cross_entropy_with_logits(
            pred_v, tgt_v, pos_weight=self.pos_weight.to(dev)
        )

    def _save_vis(self, rendered_dynamic, pseudo_dyn, step, input_imgs=None, aug_flip=None):
        """
        保存所有相机的动静分离对比图（batch 0）。
        每行横向拼接: [pred_dynamic_prob | gt_dynamic | orig_img]，相机纵向堆叠，
        存为 {vis_dir}/step_{step:06d}.jpg。

        颜色：pred 蓝(静)→红(动) 连续热图；gt 灰=ignore/蓝=static/红=dynamic。
        渲染/伪标签始终在原始相机朝向；input_imgs 可能被增广翻转，这里反翻转对齐。
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
            B, nC, H, W = rendered_dynamic.shape
            rows = []
            for cam in range(nC):
                prob = torch.sigmoid(rendered_dynamic[0, cam]).detach().cpu().numpy()  # (H, W)
                pred_rgb = _dyn_prob_to_rgb(prob)

                gt_np = pseudo_dyn[0, cam].detach().cpu().numpy().astype(np.int32)  # (H, W)
                gt_rgb = _dyn_gt_to_rgb(gt_np)

                # 原始相机图像（当前帧 t = index 0，batch 0）
                orig_img_rgb = None
                if input_imgs is not None:
                    try:
                        img_t = input_imgs[0, 0, cam].detach().cpu().float().numpy()  # (C, H_img, W_img)
                        img_t = img_t.transpose(1, 2, 0)  # (H_img, W_img, C)
                        img_t = img_t * np.array([58.395, 57.12, 57.375], dtype=np.float32) \
                                      + np.array([123.675, 116.28, 103.53], dtype=np.float32)
                        img_t = np.clip(img_t, 0, 255).astype(np.uint8)
                        flip_val = False
                        if aug_flip is not None:
                            flip_val = aug_flip.item() if hasattr(aug_flip, 'item') else bool(aug_flip)
                        if flip_val:
                            img_t = img_t[:, ::-1, :].copy()
                        # 裁剪上部对齐伪标签区域（与 RenderLoss 同逻辑）
                        H_img = img_t.shape[0]
                        crop_start = int((318 - 36) / (900 - 36) * H_img)
                        img_t = img_t[crop_start:, :, :]
                        orig_img_pil = Image.fromarray(img_t).resize((W, H), Image.BILINEAR)
                        orig_img_rgb = np.array(orig_img_pil)
                    except Exception:
                        pass

                n_cols = 3 if orig_img_rgb is not None else 2
                separator = np.zeros((2, W * n_cols, 3), dtype=np.uint8)
                cols = [pred_rgb, gt_rgb]
                if orig_img_rgb is not None:
                    cols.append(orig_img_rgb)
                row = np.concatenate(cols, axis=1)
                rows.append(separator)
                rows.append(row)

            combined = np.concatenate(rows, axis=0)
            out_path = os.path.join(self.vis_dir, f'step_{step:06d}.jpg')
            Image.fromarray(combined).save(out_path, quality=90)
        except Exception as e:
            logging.getLogger('mmengine').warning(f'[DynamicLoss] vis save failed: {e}')
