import logging
import torch
import torch.nn as nn
from . import OPENOCC_LOSS
from .base_loss import BaseLoss


@OPENOCC_LOSS.register_module()
class PhysicsLoss(BaseLoss):
    """
    Physical priors on Gaussian motion via offset predictions:
    1. Static: static gaussians should have zero offset (no motion)
    2. Smoothness: dynamic gaussians should have small acceleration (constant velocity)

    Requires:
        offset:          (B, G, 6, 2) predicted future displacements (xy)
        dynamic_logits:  (B, G, 1) raw dynamic/static logit per gaussian
    """

    def __init__(
        self,
        static_w=5.0,
        smooth_w=50.0,
        warmup_epoch=2,
        dyn_threshold=0.5,
        weight=1.0,
        input_dict=None,
        **kwargs,
    ):
        if input_dict is None:
            input_dict = {
                'offset': 'offset',
                'dynamic_logits': 'dynamic_logits',
                'current_epoch': 'current_epoch',
            }
        super().__init__(weight=weight, input_dict=input_dict, **kwargs)
        # BaseLoss.__init__ sets self.loss_func as instance attr, shadowing method
        if hasattr(self, 'loss_func'):
            del self.loss_func

        self.static_w = static_w
        self.smooth_w = smooth_w
        self.warmup_epoch = warmup_epoch
        self.dyn_threshold = dyn_threshold
        self._diag_counter = 0

    def forward(self, inputs):
        actual_inputs = {}
        for input_key, input_val in self.input_dict.items():
            actual_inputs[input_key] = inputs.get(input_val)
        loss = self.loss_func(**actual_inputs)
        return self.weight * loss, {
            'PhysicsLoss': (self.weight * loss).detach().item(),
        }

    def loss_func(self, offset, dynamic_logits, current_epoch=None):
        """
        Args:
            offset:          (B, G, 6, 2) or flat tensor needing reshape
            dynamic_logits:  (B, G, 1) raw logit, or None (eval mode)
            current_epoch:   int or None
        """
        if offset is None or dynamic_logits is None:
            return torch.tensor(0.0, requires_grad=False)

        # Reshape offset if needed
        if offset.dim() == 2:
            # (G, 12) -> (1, G, 6, 2)
            G = offset.shape[0]
            offset = offset.reshape(1, G, 6, 2)
        elif offset.dim() == 3:
            # (B, G, 12) -> (B, G, 6, 2)
            offset = offset.reshape(offset.shape[0], offset.shape[1], 6, 2)

        # dynamic_logits: ensure (B, G)
        if dynamic_logits.dim() == 3:
            dynamic_logits = dynamic_logits[..., 0]  # (B, G, 1) -> (B, G)

        p_dyn = torch.sigmoid(dynamic_logits)  # (B, G) in [0, 1]

        # ====== Static constraint: static gaussians should not move ======
        static_mask = (1.0 - p_dyn).unsqueeze(-1).unsqueeze(-1)  # (B, G, 1, 1)
        loss_static = self.static_w * (static_mask * offset.pow(2)).mean()

        # ====== Smoothness constraint: dynamic accel should be small ======
        velocity = offset[..., 1:, :] - offset[..., :-1, :]  # (B, G, 5, 2)
        acceleration = velocity[..., 1:, :] - velocity[..., :-1, :]  # (B, G, 4, 2)
        dyn_mask = p_dyn.unsqueeze(-1).unsqueeze(-1)  # (B, G, 1, 1)
        loss_smooth = self.smooth_w * (dyn_mask * acceleration.pow(2)).mean()

        total = loss_static + loss_smooth

        # Warmup: skip for early epochs when dynamic_logits are unreliable
        if current_epoch is not None and current_epoch < self.warmup_epoch:
            total = total * 0.0

        # Diagnostics
        self._diag_counter += 1
        if self._diag_counter % 500 == 1:
            with torch.no_grad():
                n_static = (p_dyn < self.dyn_threshold).sum().item()
                n_dyn = (p_dyn >= self.dyn_threshold).sum().item()
                off_mag = offset.pow(2).sum(-1).sqrt().mean().item()
            logging.getLogger('mmengine').info(
                f'[PhysicsLoss Diag] iter={self._diag_counter} | '
                f'static={n_static} dyn={n_dyn} | '
                f'offset_rms={off_mag:.4f} | '
                f'loss_static={loss_static.item():.4f} '
                f'loss_smooth={loss_smooth.item():.4f} '
                f'total={total.item():.4f}'
            )

        return total
