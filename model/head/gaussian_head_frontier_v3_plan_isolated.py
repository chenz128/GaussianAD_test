"""Planner-only, GT-free Future-Gaussian adapter for the V3-SE3 head.

The inherited V3-SE3 OCC/flow forward is intentionally left untouched.  This
subclass adds a second, planner-only feature path which is built exclusively
from current/past predictions and poses.  No future pose, ego trajectory, GT
box, or occupancy label is read by this path.
"""

import torch
import torch.nn as nn

from mmengine.registry import MODELS

from .gaussian_head_frontier_v3_isolated import (
    GaussianHeadFrontierV3Isolated,
)


@MODELS.register_module()
class GaussianHeadFrontierV3PlanIsolated(GaussianHeadFrontierV3Isolated):
    """Keep the proven V3-SE3 OCC branch and expose predicted future Gaussians.

    The current 25,600 Gaussian bank is kept intact.  A bounded subset of the
    V3 direct-future bank is appended, yielding a fixed-size tensor and an
    activation mask for the Planner.  Current-object motion is predicted by a
    planner-only offset head from temporal features; the oracle ``offset`` from
    the V3 OCC branch is never used here.
    """

    _PLANNER_META_KEYS = ('projection_mat', 'image_wh', 'lidar2global')

    def __init__(
        self,
        planner_direct_budget=6400,
        planner_future_grad_scale=0.0,
        planner_fut_ts=6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.planner_direct_budget = int(planner_direct_budget)
        self.planner_future_grad_scale = float(planner_future_grad_scale)
        self.planner_fut_ts = int(planner_fut_ts)
        if self.planner_fut_ts != 6:
            raise ValueError('V3 direct-future generation requires 6 steps')
        if self.planner_direct_budget < 0:
            raise ValueError('planner_direct_budget must be non-negative')

        context_dims = self.future_generator.temporal_in_dims
        self.planner_offset_head = nn.Sequential(
            nn.Linear(context_dims, context_dims),
            nn.LayerNorm(context_dims),
            nn.SiLU(),
            nn.Linear(context_dims, self.planner_fut_ts * 2),
        )
        # Start from the safe static-current-bank behaviour.  Unlike a
        # speed-norm kinematic head, this linear output has a valid gradient at
        # zero and is learned end-to-end from the planning losses.
        nn.init.zeros_(self.planner_offset_head[-1].weight)
        nn.init.zeros_(self.planner_offset_head[-1].bias)

    @staticmethod
    def _batch_gather(tensor, index):
        """Gather ``tensor`` along dimension 1 with one index set per batch."""
        view_shape = list(index.shape) + [1] * (tensor.dim() - 2)
        expand_shape = list(index.shape) + list(tensor.shape[2:])
        gather_index = index.view(*view_shape).expand(*expand_shape)
        return torch.gather(tensor, 1, gather_index)

    def _planner_gradient_blend(self, tensor):
        """Preserve forward values while limiting Planner gradients to V3."""
        scale = self.planner_future_grad_scale
        if scale <= 0:
            return tensor.detach()
        if scale >= 1:
            return tensor
        return scale * tensor + (1.0 - scale) * tensor.detach()

    def _history_future_to_current(self, metas, reference):
        """Extrapolate six ego poses from current/past LiDAR poses only.

        V3 orders the current frame at ``current_frame_index``.  The nearest
        history pose supplies a constant body-frame SE(3) delta.  Repeated
        composition gives future ego poses in current-LiDAR coordinates without
        reading ``future_lidar2global`` or ``ego_fut_trajs``.
        """
        poses = metas['lidar2global']
        if not torch.is_tensor(poses):
            poses = torch.as_tensor(poses)
        poses = poses.to(reference)
        if poses.dim() == 3:
            poses = poses[None]
        if poses.dim() != 4 or poses.shape[-2:] != (4, 4):
            raise ValueError(
                'lidar2global must have shape (B,F,4,4), got '
                f'{tuple(poses.shape)}')
        frame_count = poses.shape[1]
        if frame_count < 2:
            raise ValueError('at least one history pose is required')
        current_index = self.current_frame_index % frame_count
        history_index = (current_index + 1 if current_index == 0
                         else current_index - 1)
        current = poses[:, current_index]
        history = poses[:, history_index]
        body_delta = torch.linalg.inv(history) @ current
        transforms = [
            torch.linalg.matrix_power(body_delta, step)
            for step in range(1, self.planner_fut_ts + 1)
        ]
        return torch.stack(transforms, dim=1)

    def _current_temporal_features(self, features, indices, batch_size,
                                   current_count):
        """Return temporal features aligned with the current Gaussian order."""
        per_batch = []
        frame_index = self.current_frame_index % self.future_generator.num_frames
        for batch_index in range(batch_size):
            mask = ((indices[:, 0].long() == batch_index)
                    & (indices[:, 1].long() == frame_index))
            selected = features[mask]
            if selected.shape[0] != current_count:
                raise AssertionError(
                    'current temporal/Gaussian count mismatch: '
                    f'{selected.shape[0]} vs {current_count}')
            per_batch.append(selected)
        return torch.stack(per_batch, dim=0)

    @staticmethod
    def _gaussian_features(gaussian):
        features = torch.cat([
            gaussian.means,
            gaussian.scales,
            gaussian.rotations,
            gaussian.opacities,
            gaussian.semantics,
        ], dim=-1)
        if features.shape[-1] != 28:
            raise AssertionError(
                f'Planner requires 28-D Gaussians, got {features.shape[-1]}')
        return features

    @staticmethod
    def _generated_features(generated):
        features = torch.cat([
            generated['means'],
            generated['scales'],
            generated['rotations'],
            generated['opacities'],
            generated['semantics'],
        ], dim=-1)
        if features.shape[-1] != 28:
            raise AssertionError(
                f'Planner requires 28-D direct Gaussians, got '
                f'{features.shape[-1]}')
        return features

    def predict_planner_future_gaussians(
        self,
        representation_temp,
        metas,
        temporal_context_features,
        temporal_context_indices,
        ms_img_feats,
    ):
        """Build a fixed GT-free ``(B,6,K,28)`` Planner input and mask."""
        gaussian = representation_temp['gaussian']
        current = self._gaussian_features(gaussian)
        batch_size, current_count = current.shape[:2]
        if batch_size != 1:
            raise AssertionError(
                'V3-SE3 joint Planner requires batch size 1 per GPU')
        self._check_current_bank(current_count)

        # Planner-specific motion: current/past temporal predictions only.
        current_context = self._current_temporal_features(
            temporal_context_features,
            temporal_context_indices,
            batch_size,
            current_count,
        )
        planner_offset = self.planner_offset_head(
            current_context.detach()).reshape(
                batch_size, current_count, self.planner_fut_ts, 2)
        planner_offset = torch.nan_to_num(
            planner_offset, nan=0.0, posinf=0.0, neginf=0.0)
        current_xy = current[:, None, :, :2].detach()
        current_rest = current[:, None, :, 2:].detach().expand(
            -1, self.planner_fut_ts, -1, -1)
        moved_current = torch.cat([
            current_xy + planner_offset.permute(0, 2, 1, 3),
            current_rest,
        ], dim=-1)

        # Direct V3 bank: query placement is driven by a history-only SE(3)
        # extrapolation.  Pass an explicit metadata whitelist so future GT cannot
        # be consumed accidentally by the generator.
        future_to_current = self._history_future_to_current(metas, current)
        safe_metas = {key: metas[key] for key in self._PLANNER_META_KEYS}
        # Default scale=0 protects the original V3 generator exactly and avoids
        # retaining a second 12,800-query autograd graph.  A small positive
        # scale can be enabled explicitly for a later joint-gradient ablation.
        planner_generator_grad = (
            torch.is_grad_enabled() and self.planner_future_grad_scale > 0)
        with torch.set_grad_enabled(planner_generator_grad):
            generated = self.future_generator(
                ego_cumulative=future_to_current[..., :3, 3],
                temporal_features=temporal_context_features,
                temporal_indices=temporal_context_indices,
                ms_img_feats=ms_img_feats,
                metas=safe_metas,
                batch_size=batch_size,
                future_to_current_rotations=future_to_current[..., :3, :3],
            )
        direct = self._generated_features(generated)
        direct = torch.nan_to_num(direct, nan=0.0, posinf=0.0, neginf=0.0)

        direct_count = min(self.planner_direct_budget, direct.shape[1])
        if direct_count > 0:
            importance = generated['opacities'][..., 0]
            importance = importance * generated['semantics'].amax(dim=-1)
            direct_index = torch.topk(
                importance, direct_count, dim=1, sorted=False).indices
            direct = self._batch_gather(direct, direct_index)
            enter_time = self._batch_gather(
                generated['enter_time'], direct_index)
            direct = self._planner_gradient_blend(direct)
            direct = direct[:, None].expand(
                -1, self.planner_fut_ts, -1, -1)
            times = torch.arange(
                1, self.planner_fut_ts + 1,
                device=direct.device,
                dtype=enter_time.dtype,
            ).reshape(1, self.planner_fut_ts, 1) / self.planner_fut_ts
            direct_padding_mask = enter_time[:, None, :] > times
            future = torch.cat([moved_current, direct], dim=2)
            current_mask = torch.zeros(
                batch_size, self.planner_fut_ts, current_count,
                dtype=torch.bool, device=future.device)
            padding_mask = torch.cat(
                [current_mask, direct_padding_mask], dim=2)
        else:
            future = moved_current
            padding_mask = torch.zeros(
                batch_size, self.planner_fut_ts, current_count,
                dtype=torch.bool, device=future.device)

        if padding_mask.all(dim=-1).any():
            raise AssertionError('every Planner timestep needs a valid Gaussian')
        return {
            'planner_future_gaussians': future,
            'planner_future_gaussian_mask': padding_mask,
            'planner_predicted_offset': planner_offset,
        }
