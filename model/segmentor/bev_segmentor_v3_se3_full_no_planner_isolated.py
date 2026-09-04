"""Isolated full-data V3-SE3 perception model without a trajectory Planner."""

from mmseg.models import SEGMENTORS

from .bev_segmentor import BEVSegmentor


@SEGMENTORS.register_module()
class BEVSegmentorV3SE3FullNoPlannerIsolated(BEVSegmentor):
    """Run the V3-SE3 perception, map and OCC paths only.

    The class intentionally rejects a ``planner_head`` configuration so a
    future config refactor cannot silently put trajectory planning back into
    this Stage-1 model.  All inherited perception modules and their execution
    order remain unchanged.
    """

    def __init__(self, *args, planner_head=None, **kwargs):
        if planner_head is not None:
            raise ValueError(
                'BEVSegmentorV3SE3FullNoPlannerIsolated forbids planner_head')
        super().__init__(*args, planner_head=None, **kwargs)

    def forward(self, imgs=None, metas=None, points=None,
                extra_backbone=False, occ_only=False, rep_only=False,
                **kwargs):
        if extra_backbone:
            return self.forward_extra_img_backbone(imgs=imgs)

        results = {
            'imgs': imgs,
            'metas': metas,
            'points': points,
            'gt_boxes': metas['gt_boxes'],
        }
        results.update(kwargs)
        results.update(self.extract_img_feat(**results))
        results.update(self.lifter(**results))

        if (self.training and self.history_no_grad and imgs is not None
                and imgs.dim() >= 2 and imgs.size(1) > 1):
            results.update(self._encoder_forward_split(results))
        else:
            results.update(self.encoder(**results))
        if hasattr(self, 'temporal_encoder'):
            results.update(self.temporal_encoder(**results))

        results.update(self.decoder(results))
        if hasattr(self, 'map_decoder'):
            results.update(self.map_decoder(results))

        # No planner forward is allowed in this Stage-1 model.
        results.update(self.head(**results))

        # Labels are appended only after prediction forward.  They supervise
        # losses but cannot enter any perception component as hidden inputs.
        results['ego_fut_trajs'] = metas['ego_fut_trajs']
        return results
