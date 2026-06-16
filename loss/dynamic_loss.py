import os
import logging
import torch
import torch.nn as nn
import numpy as np
from . import OPENOCC_LOSS
from .base_loss import BaseLoss


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
        vis_dir=None,
        vis_every=500,
        input_dict=None,
        **kwargs,
    ):
        if input_dict is None:
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
        # dynamic pixels are rare (~1.6% of labeled px) → up-weight positive class
        self.register_buffer('pos_weight', torch.tensor(float(pos_weight)))

    def forward(self, inputs):
        actual_inputs = {}
        for input_key, input_val in self.input_dict.items():
            # .get(): multi-frame keys are absent unless the config maps them,
            # so missing keys resolve to None and the loss skips that branch.
            actual_inputs.update({input_key: inputs.get(input_val)})
        loss = self.loss_func(**actual_inputs)
        return self.weight * loss, {
            'DynamicLoss': (self.weight * loss).detach().item(),
        }

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
            loss = loss + extra
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
