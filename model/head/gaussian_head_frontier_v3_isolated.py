import torch
import torch.nn.functional as F

from mmengine.registry import MODELS

from .future_gaussian_direct_generator_v3_isolated import (
    FutureGaussianDirectGenerator,
)
from .future_local_aggregate_v3_isolated import aggregate_v3_future
from .gaussian_head import GaussianHead


@MODELS.register_module()
class GaussianHeadFrontierV3Isolated(GaussianHead):
    """Render retained current Gaussians plus one shared direct future bank."""

    def __init__(self, target_num_gaussians=25600, direct_generator=None,
                 current_frame_index=0, min_current_gaussian_ratio=0.99,
                 dynamic_class_multiplier=3.0, future_pose_mode='translation',
                 strict_range_mask=False, range_mask_sigma=0.0,
                 center_only_mask=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.target_num_gaussians = target_num_gaussians
        self.current_frame_index = current_frame_index
        self.min_current_gaussian_ratio = min_current_gaussian_ratio
        self.dynamic_class_multiplier = dynamic_class_multiplier
        if future_pose_mode not in ('translation', 'se3'):
            raise ValueError(
                f'unsupported future_pose_mode={future_pose_mode!r}')
        self.future_pose_mode = future_pose_mode
        self.strict_range_mask = strict_range_mask
        self.range_mask_sigma = range_mask_sigma
        self.center_only_mask = center_only_mask
        config = dict(direct_generator or {})
        config.setdefault('pc_range', tuple(self.pc_range))
        config.setdefault(
            'num_classes', self.num_classes - 1 if self.with_emtpy
            else self.num_classes)
        config.setdefault('current_frame_index', current_frame_index)
        self.future_generator = FutureGaussianDirectGenerator(**config)

    def get_future_lidar_transforms(self, metas, reference,
                                    provided_transforms=None):
        """Return current-LiDAR to future-LiDAR transforms."""
        if provided_transforms is not None:
            transforms = provided_transforms
            if not torch.is_tensor(transforms):
                transforms = torch.as_tensor(transforms)
            transforms = transforms.to(reference)
        else:
            if 'future_lidar2global' not in metas:
                raise KeyError(
                    'future_lidar2global is required when no planner-provided '
                    'future_lidar_transforms are available')
            current = metas['lidar2global']
            future = metas['future_lidar2global']
            if not torch.is_tensor(current):
                current = torch.as_tensor(current)
            if not torch.is_tensor(future):
                future = torch.as_tensor(future)
            current = current.to(reference)
            future = future.to(reference)
            if current.dim() == 3:
                current = current[:, None]
            elif current.dim() == 4:
                current = current[
                    :, self.current_frame_index:self.current_frame_index + 1]
            else:
                raise ValueError(
                    f'unsupported lidar2global shape {tuple(current.shape)}')
            transforms = torch.linalg.inv(future) @ current

        if transforms.dim() == 3:
            transforms = transforms[None]
        if transforms.dim() != 4 or transforms.shape[-2:] != (4, 4):
            raise ValueError(
                'future_lidar_transforms must have shape (B,T,4,4), got '
                f'{tuple(transforms.shape)}')
        return transforms

    @staticmethod
    def transform_points(points, transforms):
        """Apply one transform per batch item to batched 3D points."""
        rotation = transforms[:, :3, :3]
        translation = transforms[:, :3, 3]
        return torch.matmul(
            points, rotation.transpose(-1, -2)) + translation[:, None]

    @staticmethod
    def rotate_covariances(covariances, transforms):
        """Rotate covariance or inverse-covariance matrices."""
        rotation = transforms[:, None, :3, :3]
        return rotation @ covariances @ rotation.transpose(-1, -2)

    def get_in_range_mask(self, points, scales=None):
        """Keep Gaussians whose footprint overlaps the future render ROI."""
        grid = ((points - self.pc_min) / self.grid_size).to(torch.int)
        if self.center_only_mask:
            # Retain any Gaussian whose centre lies inside the render volume.
            # SE(3) rotation of the old bank across a turn swings edge
            # gaussians out of the box; their centres are still inside the
            # future render ROI, so keeping them avoids the "lost history"
            # holes without considering scale-margin at all.
            pc_min = self.pc_min[0]
            pc_max = self.pc_min[0] + self.grid_size[0] * points.new_tensor(
                [120, 120, 8])
            return ((points[..., 0] >= pc_min[0])
                    & (points[..., 0] < pc_max[0])
                    & (points[..., 1] >= pc_min[1])
                    & (points[..., 1] < pc_max[1])
                    & (points[..., 2] >= pc_min[2])
                    & (points[..., 2] < pc_max[2]))
        if self.strict_range_mask:
            if scales is not None and self.range_mask_sigma > 0:
                margin = scales.abs().amax(dim=-1, keepdim=True)
                margin = margin * self.range_mask_sigma
                lower = self.pc_min - margin
                upper = self.pc_min + self.grid_size * points.new_tensor(
                    [120, 120, 8]) + margin
                return ((points >= lower) & (points < upper)).all(dim=-1)
            grid = (points - self.pc_min) / self.grid_size
        return ((grid[..., 0] >= 0) & (grid[..., 0] < 120)
                & (grid[..., 1] >= 0) & (grid[..., 1] < 120)
                & (grid[..., 2] >= 0) & (grid[..., 2] < 8))

    def _check_current_bank(self, current_count):
        minimum = int(
            self.target_num_gaussians * self.min_current_gaussian_ratio)
        if current_count < minimum:
            raise AssertionError(
                f'current Gaussian count {current_count} is below the '
                f'minimum {minimum}; check temporal frame order for a '
                f'cropped current-frame bank')
        if current_count > self.target_num_gaussians:
            raise AssertionError(
                f'current Gaussian count {current_count} exceeds target '
                f'{self.target_num_gaussians}')

    def forward_flow(self, sampled_xyz, representation_temp, metas=None,
                     gs=None, **kwargs):
        gaussian = representation_temp['gaussian']
        means = self._flow_blend(gaussian.means)
        gs = tuple(self._flow_blend(tensor) for tensor in gs)
        batch_size, current_count = means.shape[:2]
        if batch_size != 1:
            raise AssertionError(
                'GaussianHeadFrontierV3 currently requires batch size 1 per GPU')
        self._check_current_bank(current_count)

        offset = kwargs['offset'].reshape(
            batch_size, current_count, 6, 2)
        offset = torch.cat(
            [offset, offset.new_zeros(*offset.shape[:-1], 1)], dim=-1)
        means_future = means[..., None, :] + offset
        provided_transforms = kwargs.get('future_lidar_transforms')
        if self.future_pose_mode == 'se3' or provided_transforms is not None:
            future_transforms = self.get_future_lidar_transforms(
                metas, means, provided_transforms=provided_transforms)
        else:
            ego = metas['ego_fut_trajs']
            if not torch.is_tensor(ego):
                ego = torch.as_tensor(ego)
            ego = torch.nan_to_num(ego.to(means).float()).cumsum(dim=1)
            future_transforms = torch.eye(
                4, device=means.device, dtype=means.dtype
            ).reshape(1, 1, 4, 4).repeat(batch_size, 6, 1, 1)
            future_transforms[..., :2, 3] = -ego
        if future_transforms.shape[1] != 6:
            raise ValueError(
                'GaussianHeadFrontierV3 requires 6 future transforms, got '
                f'{future_transforms.shape[1]}')
        future_to_current = torch.linalg.inv(future_transforms)
        future_origins = future_to_current[..., :3, 3]

        generated = self.future_generator(
            ego_cumulative=future_origins,
            temporal_features=kwargs['temporal_context_features'],
            temporal_indices=kwargs['temporal_context_indices'],
            ms_img_feats=kwargs['ms_img_feats'],
            metas=metas,
            batch_size=batch_size,
            future_to_current_rotations=(
                future_to_current[..., :3, :3]))

        original_opacity, semantics_all, scales_all, cov_all = gs
        generated_semantics = generated['semantics']
        if generated_semantics.shape[-1] < semantics_all.shape[-1]:
            generated_semantics = F.pad(
                generated_semantics,
                (0, semantics_all.shape[-1]
                 - generated_semantics.shape[-1]))

        predictions = []
        for step in range(6):
            transform = future_transforms[:, step]
            warped_old = self.transform_points(
                means_future[..., step, :], transform)
            old_inside = self.get_in_range_mask(
                warped_old, scales_all[:, :current_count])[0]
            new_means = self.transform_points(generated['means'], transform)
            new_inside = self.get_in_range_mask(
                new_means, generated['scales'])[0]
            entered = generated['enter_time'][0] <= ((step + 1) / 6.0)
            new_active = new_inside & entered

            old_cov = self.rotate_covariances(
                cov_all[:, :current_count], transform)
            new_cov = self.rotate_covariances(
                generated['cov_inv'], transform)

            means_step = torch.cat([
                warped_old[:, old_inside],
                new_means[:, new_active]], dim=1)
            opacity_step = torch.cat([
                original_opacity[:, :current_count][:, old_inside],
                generated['opacities'][:, new_active]], dim=1)
            semantics_step = torch.cat([
                semantics_all[:, :current_count][:, old_inside],
                generated_semantics[:, new_active]], dim=1)
            scales_step = torch.cat([
                scales_all[:, :current_count][:, old_inside],
                generated['scales'][:, new_active]], dim=1)
            cov_step = torch.cat([
                old_cov[:, old_inside],
                new_cov[:, new_active]], dim=1)

            if self.with_emtpy and self.flow_include_empty:
                means_step = torch.cat(
                    [means_step, self.empty_mean.to(means_step.dtype)], 1)
                opacity_step = torch.cat(
                    [opacity_step, original_opacity[:, -1:]], 1)
                semantics_step = torch.cat(
                    [semantics_step, semantics_all[:, -1:]], 1)
                scales_step = torch.cat(
                    [scales_step, scales_all[:, -1:]], 1)
                cov_step = torch.cat([cov_step, cov_all[:, -1:]], 1)

            _, count = means_step.shape[:2]
            # The current-frame path keeps the shared LocalAggregator and its
            # centre-in-grid assertion.  Only V3 future-flow accepts centres
            # outside the ROI whose finite Gaussian footprint overlaps it,
            # matching the original V3-SE3 implementation.
            semantics = aggregate_v3_future(
                self.aggregator, sampled_xyz.clone().float(), means_step,
                opacity_step.reshape(batch_size, count), semantics_step,
                scales_step, cov_step)[None].transpose(1, 2)
            predictions.append([dict(
                pred_flow=semantics,
                sampled_label=metas['flow_info'][0][step]['occ_label'],
                flow_valid_flag=(
                    metas['flow_info'][0][step]['flow_valid_flag']),
                dynamic_class_multiplier=self.dynamic_class_multiplier,
                num_retained_gaussians=int(old_inside.sum().detach()),
                num_generated_gaussians=int(new_active.sum().detach()),
                num_render_gaussians=count,
                image_visible_ratio=generated['image_visible_ratio'])])
        return predictions
