"""Frozen V3-SE3 frontend with the original current-frame VAD Planner.

This Stage-2 model deliberately has no predicted-Future-Gaussian adapter.
The Planner receives only frozen model predictions: agent boxes, map vectors
and the current 28-D Gaussian bank.
"""

import torch
from mmseg.models import SEGMENTORS

from .bev_segmentor import BEVSegmentor


@SEGMENTORS.register_module()
class BEVSegmentorV3SE3FrozenVADIsolated(BEVSegmentor):
    """Train only an original :class:`VADHead` on a frozen V3-SE3 frontend."""

    _FRONTEND_MODULES = (
        'img_backbone',
        'img_neck',
        'lifter',
        'encoder',
        'temporal_encoder',
        'decoder',
        'map_decoder',
        'head',
    )
    _PLANNER_KEYS = (
        'final_box_dicts',
        'all_cls_scores',
        'all_pts_preds',
        'gaussian_output',
    )

    def __init__(self, *args, planner_head=None, **kwargs):
        planner_type = (planner_head.get('type')
                        if hasattr(planner_head, 'get') else None)
        if planner_type != 'VADHead':
            raise ValueError(
                'BEVSegmentorV3SE3FrozenVADIsolated requires the original '
                f'VADHead, got {planner_type!r}')
        super().__init__(*args, planner_head=planner_head, **kwargs)
        self._freeze_frontend()

    def _freeze_frontend(self):
        for name in self._FRONTEND_MODULES:
            module = getattr(self, name, None)
            if module is not None:
                module.requires_grad_(False)
                module.eval()

    def train(self, mode=True):
        """Keep every frozen frontend module in eval mode across epochs."""
        super().train(mode)
        if mode:
            self._freeze_frontend()
            self.planner_head.train(True)
        return self

    @staticmethod
    def _detach_tree(value):
        if torch.is_tensor(value):
            return value.detach()
        if isinstance(value, dict):
            return {
                key: BEVSegmentorV3SE3FrozenVADIsolated._detach_tree(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [
                BEVSegmentorV3SE3FrozenVADIsolated._detach_tree(item)
                for item in value
            ]
        if isinstance(value, tuple):
            return tuple(
                BEVSegmentorV3SE3FrozenVADIsolated._detach_tree(item)
                for item in value
            )
        return value

    def _forward_frozen_frontend(self, results):
        """Run the epoch-20 frontend without autograd or state updates."""
        with torch.no_grad():
            results.update(self.extract_img_feat(**results))
            results.update(self.lifter(**results))
            # history_no_grad is intentionally disabled by the Stage-2 config;
            # the whole frontend already runs under no_grad in its eval mode.
            results.update(self.encoder(**results))
            if hasattr(self, 'temporal_encoder'):
                results.update(self.temporal_encoder(**results))
            results.update(self.decoder(results))
            if hasattr(self, 'map_decoder'):
                results.update(self.map_decoder(results))
        return results

    def forward(self, imgs=None, metas=None, points=None,
                extra_backbone=False, occ_only=False, rep_only=False,
                **kwargs):
        if extra_backbone:
            with torch.no_grad():
                return self.forward_extra_img_backbone(imgs=imgs)

        results = {
            'imgs': imgs,
            'metas': metas,
            'points': points,
            # Preserves the original V3-SE3 OCC/offset behaviour.  This field
            # is never included in the explicit Planner input whitelist.
            'gt_boxes': metas['gt_boxes'],
        }
        results.update(kwargs)
        self._forward_frozen_frontend(results)

        missing = [key for key in self._PLANNER_KEYS if key not in results]
        if missing:
            raise KeyError(f'missing frozen VAD input predictions: {missing}')
        planner_inputs = {
            key: self._detach_tree(results[key])
            for key in self._PLANNER_KEYS
        }
        if set(planner_inputs) != set(self._PLANNER_KEYS):
            raise AssertionError('unexpected field reached VADHead')
        results.update(self.planner_head(planner_inputs))

        # The frozen OCC head is unnecessary for Planner gradients.  Run it
        # only in evaluation so training is faster while OCC metrics remain
        # available for the numerical-preservation check.
        if not self.training:
            with torch.no_grad():
                results.update(self.head(**results))

        # Loss-only supervision is appended after Planner forward.
        results['ego_fut_trajs'] = metas['ego_fut_trajs']
        return results
