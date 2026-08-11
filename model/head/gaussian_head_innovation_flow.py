from mmengine.registry import MODELS

from .gaussian_head_frontier_v3 import GaussianHeadFrontierV3
from .innovation_flow_generator import InnovationFlowGenerator


@MODELS.register_module()
class GaussianHeadInnovationFlow(GaussianHeadFrontierV3):
    """V3 retained branch plus latent flow-matched innovation Gaussians."""

    def __init__(self, innovation_flow=None, direct_generator=None, **kwargs):
        super().__init__(direct_generator=direct_generator or {}, **kwargs)
        config = dict(innovation_flow or {})
        config.setdefault('pc_range', tuple(self.pc_range))
        config.setdefault(
            'num_classes', self.num_classes - 1 if self.with_emtpy
            else self.num_classes)
        config.setdefault('current_frame_index', self.current_frame_index)
        config.setdefault(
            'target_pose_mode',
            getattr(self, 'future_pose_mode', 'translation'))
        self.future_generator = InnovationFlowGenerator(**config)
        self._flow_matching_loss = None

    def forward_flow(self, *args, **kwargs):
        predictions = super().forward_flow(*args, **kwargs)
        masks = self.future_generator.last_innovation_masks
        if masks is not None:
            for step, prediction in enumerate(predictions):
                prediction[0]['innovation_mask'] = masks[0, step]
        self._flow_matching_loss = (
            self.future_generator.last_flow_matching_loss)
        return predictions

    def forward(self, *args, **kwargs):
        output = super().forward(*args, **kwargs)
        output['flow_matching_loss'] = self._flow_matching_loss
        return output