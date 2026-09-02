"""Isolated V3-SE3 -> predicted Future Gaussian -> Planner execution order."""

from mmseg.models import SEGMENTORS

from .bev_segmentor import BEVSegmentor


@SEGMENTORS.register_module()
class BEVSegmentorV3FuturePlanIsolated(BEVSegmentor):
    """Preserve V3-SE3 OCC while giving Planner a GT-free input whitelist."""

    _PLANNER_KEYS = (
        'final_box_dicts',
        'all_cls_scores',
        'all_pts_preds',
        'gaussian_output',
        'planner_future_gaussians',
        'planner_future_gaussian_mask',
    )

    def forward(self, imgs=None, metas=None, points=None,
                extra_backbone=False, occ_only=False, rep_only=False,
                **kwargs):
        if extra_backbone:
            return self.forward_extra_img_backbone(imgs=imgs)

        # The original V3-SE3 OCC branch still receives its original labels and
        # oracle motion metadata.  They never enter ``planner_inputs`` below.
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

        # Generate model-predicted future Gaussians BEFORE Planner.  This method
        # has its own metadata whitelist and ignores the oracle V3 offset.
        results.update(self.head.predict_planner_future_gaussians(
            representation_temp=results['representation_temp'],
            metas=metas,
            temporal_context_features=results['temporal_context_features'],
            temporal_context_indices=results['temporal_context_indices'],
            ms_img_feats=results['ms_img_feats'],
        ))

        missing = [key for key in self._PLANNER_KEYS if key not in results]
        if missing:
            raise KeyError(f'missing Planner prediction inputs: {missing}')
        planner_inputs = {key: results[key] for key in self._PLANNER_KEYS}
        results.update(self.planner_head(planner_inputs))

        # Run the untouched V3-SE3 OCC/flow path after planning.  Keeping this
        # branch unchanged preserves the source model's OCC behaviour.
        results.update(self.head(**results))

        # Loss-only label plumbing happens after every prediction forward.
        results['ego_fut_trajs'] = metas['ego_fut_trajs']
        return results
