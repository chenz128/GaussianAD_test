"""Training-mode guard for planner-only strong-OCC fine-tuning."""

from mmseg.models import SEGMENTORS

from .bev_segmentor_v3_future_plan_isolated import (
    BEVSegmentorV3FuturePlanIsolated,
)


@SEGMENTORS.register_module()
class BEVSegmentorV3FuturePlanFrozenFrontend(
        BEVSegmentorV3FuturePlanIsolated):
    """Keep every pre-planner module in eval mode during planner training.

    ``train.py`` freezes parameters from ``cfg.frozen_modules`` once, but its
    epoch loop subsequently calls ``model.train()``.  Frozen BatchNorm buffers
    and dropout behavior can therefore still change.  This parameter-free
    subclass preserves the exact inherited forward and only reapplies eval
    mode to the immutable frontend after every train-mode transition.
    """

    _FROZEN_FRONTEND_MODULES = (
        'img_backbone',
        'img_neck',
        'lifter',
        'encoder',
        'temporal_encoder',
        'decoder',
        'map_decoder',
        'head',
    )

    def train(self, mode=True):
        super().train(mode)
        if mode:
            for name in self._FROZEN_FRONTEND_MODULES:
                module = getattr(self, name, None)
                if module is None:
                    raise AttributeError(
                        f'frozen frontend module does not exist: {name}')
                module.eval()
        return self
