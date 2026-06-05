import torch
from . import OPENOCC_LOSS
from .base_loss import BaseLoss


@OPENOCC_LOSS.register_module()
class GaussianRegLoss(BaseLoss):
    """
    Regularization on Gaussian primitives for the pure-2D (only2D) supervision
    scheme. Without the 3D OccupancyLoss pushing Gaussians to be large/opaque,
    these soft penalties keep the rendered 2D semantics/depth sharp:

      - scale_reg:   L1 penalty on Gaussian scales (meters). Encourages small
                     Gaussians → sharper 2D projection boundaries.
      - opacity_reg: binary entropy penalty on opacities. Pushes each opacity
                     toward 0 (prune useless Gaussian) or 1 (solid surface),
                     removing the semi-transparent "foggy" overlap.

    Returns a (total, sub_dict) tuple so MultiLoss logs ScaleReg / OpacityReg
    separately (same convention as RenderLoss).
    """

    def __init__(
        self,
        weight=1.0,
        scale_lw=0.05,
        opacity_lw=0.05,
        input_dict=None,
        **kwargs,
    ):
        if input_dict is None:
            input_dict = {
                'gaussian': 'gaussian',
            }
        super().__init__(weight=weight, input_dict=input_dict, **kwargs)
        # BaseLoss.__init__ sets self.loss_func = lambda: 0 as an instance attr,
        # which shadows our loss_func method. Delete it to restore method lookup.
        del self.loss_func

        self.scale_lw = scale_lw
        self.opacity_lw = opacity_lw

    def forward(self, inputs):
        """Override BaseLoss.forward to return (total, sub_dict) for separate logging."""
        actual_inputs = {}
        for input_key, input_val in self.input_dict.items():
            actual_inputs.update({input_key: inputs[input_val]})
        loss_scale, loss_opacity = self.loss_func(**actual_inputs)
        total = self.weight * (loss_scale + loss_opacity)
        return total, {
            'ScaleReg': (self.weight * loss_scale).detach().item(),
            'OpacityReg': (self.weight * loss_opacity).detach().item(),
        }

    def loss_func(self, gaussian):
        """
        Args:
            gaussian: GaussianPrediction namedtuple with
                scales:    (B, G, 3) in meters, range [scale_range[0], scale_range[1]]
                opacities: (B, G, 1) in (0, 1) after sigmoid
        """
        scales = gaussian.scales
        opacities = gaussian.opacities

        # L1 scale penalty (mean over all Gaussians & axes)
        loss_scale = self.scale_lw * scales.abs().mean()

        # binary entropy penalty on opacities: -(p*log p + (1-p)*log(1-p))
        p = opacities.clamp(1e-6, 1.0 - 1e-6)
        entropy = -(p * p.log() + (1.0 - p) * (1.0 - p).log())
        loss_opacity = self.opacity_lw * entropy.mean()

        return loss_scale, loss_opacity
