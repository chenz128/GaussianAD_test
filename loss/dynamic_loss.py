import logging
import torch
import torch.nn as nn
from . import OPENOCC_LOSS
from .base_loss import BaseLoss


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
        pos_weight=5.0,
        vis_every=500,
        input_dict=None,
        **kwargs,
    ):
        if input_dict is None:
            input_dict = {
                'rendered_dynamic': 'rendered_dynamic',
                'pseudo_dyn': 'pseudo_dyn',
            }
        super().__init__(weight=weight, input_dict=input_dict, **kwargs)
        # BaseLoss.__init__ sets self.loss_func = lambda: 0 as instance attr,
        # which shadows our loss_func method. Delete it to restore method lookup.
        del self.loss_func

        self.vis_every = vis_every
        # dynamic pixels are rare → up-weight the positive class
        self.register_buffer('pos_weight', torch.tensor(float(pos_weight)))

    def forward(self, inputs):
        actual_inputs = {}
        for input_key, input_val in self.input_dict.items():
            actual_inputs.update({input_key: inputs[input_val]})
        loss = self.loss_func(**actual_inputs)
        return self.weight * loss, {
            'DynamicLoss': (self.weight * loss).detach().item(),
        }

    def loss_func(self, rendered_dynamic, pseudo_dyn):
        """
        Args:
            rendered_dynamic: (B, nC, H, W) raw logits, or None (eval)
            pseudo_dyn:       (B, nC, H, W) int, 0=ignore/1=static/2=dynamic, or None
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
            )
        return loss
