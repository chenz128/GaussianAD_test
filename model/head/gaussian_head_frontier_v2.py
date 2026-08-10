import torch
import torch.nn.functional as F

from mmengine.registry import MODELS

from .frontier_context_generator import FrontierContextGenerator
from .gaussian_head import GaussianHead


@MODELS.register_module()
class GaussianHeadFrontierV2(GaussianHead):
    """Future head with exactly 25,600 real slots plus one empty Gaussian."""

    def __init__(self, target_num_gaussians=25600, frontier_context=None,
                 current_frame_index=-1,
                 min_current_gaussian_ratio=0.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.target_num_gaussians = target_num_gaussians
        self.current_frame_index = current_frame_index
        self.min_current_gaussian_ratio = min_current_gaussian_ratio
        config = dict(frontier_context or {})
        config.setdefault('pc_range', tuple(self.pc_range))
        config.setdefault(
            'num_classes', self.num_classes - 1 if self.with_emtpy
            else self.num_classes)
        self.frontier_generator = FrontierContextGenerator(**config)

    def get_in_range_mask(self, points):
        grid = ((points - self.pc_min) / self.grid_size).to(torch.int)
        return ((grid[..., 0] >= 0) & (grid[..., 0] < 120)
                & (grid[..., 1] >= 0) & (grid[..., 1] < 120)
                & (grid[..., 2] >= 0) & (grid[..., 2] < 8))

    def _current_camera_inputs(self, ms_img_feats, metas, batch_size):
        feature = ms_img_feats[0]
        frames = feature.shape[0] // batch_size
        current_index = self.current_frame_index % frames
        feature = feature.reshape(
            batch_size, frames, *feature.shape[1:])[:, current_index]
        projection = metas['projection_mat']
        cameras = feature.shape[1]
        projection = projection.reshape(
            batch_size, -1, cameras, 4, 4)[:, current_index]
        image_wh = metas['image_wh']
        image_wh = image_wh.reshape(
            batch_size, -1, cameras, 2)[:, current_index]
        return feature.detach(), projection, image_wh

    def forward_flow(self, sampled_xyz, representation_temp, metas=None,
                     gs=None, **kwargs):
        gaussian = representation_temp['gaussian']
        means = self._flow_blend(gaussian.means)
        gs = tuple(self._flow_blend(tensor) for tensor in gs)
        batch_size, current_count = means.shape[:2]
        min_current_count = int(
            self.target_num_gaussians * self.min_current_gaussian_ratio)
        if current_count < min_current_count:
            raise AssertionError(
                f'current Gaussian count {current_count} is below the '
                f'minimum {min_current_count}; check temporal frame order '
                f'for a cropped current-frame bank')
        if current_count > self.target_num_gaussians:
            raise AssertionError(
                f'current Gaussian count {current_count} exceeds target '
                f'{self.target_num_gaussians}')

        offset = kwargs['offset'].reshape(batch_size, current_count, 6, 2)
        offset = torch.cat(
            [offset, offset.new_zeros(*offset.shape[:-1], 1)], dim=-1)
        means_future = means[..., None, :] + offset

        ego = metas['ego_fut_trajs']
        if not torch.is_tensor(ego):
            ego = torch.as_tensor(ego)
        ego = ego.to(offset.device).float()
        if ego.dim() == 2:
            ego = ego[None]
        ego = torch.nan_to_num(ego).cumsum(dim=1)
        ego = torch.cat([ego, ego.new_zeros(*ego.shape[:-1], 1)], dim=-1)

        image_map, projection, image_wh = self._current_camera_inputs(
            kwargs['ms_img_feats'], metas, batch_size)
        image_features = self.frontier_generator.prepare_image_features(
            image_map)
        original_opacity, semantics_all, scales_all, cov_all = gs
        context = dict(
            means=means,
            scales=scales_all[:, :current_count],
            rotations=gaussian.rotations,
            opacities=original_opacity[:, :current_count],
            semantics=semantics_all[:, :current_count, :self.num_classes - 1])

        predictions = []
        for step in range(6):
            warped = means_future[..., step, :] - ego[:, step:step + 1]
            inside = self.get_in_range_mask(warped)
            generated = self.frontier_generator(
                ego_disp=ego[:, step],
                num_gaussians=self.target_num_gaussians,
                time_index=step,
                context_gaussian=context,
                context_valid=inside,
                image_features=image_features,
                projection_mat=projection,
                image_wh=image_wh)

            keep = inside[..., None]
            keep_cov = inside[..., None, None]
            generated_semantics = generated['semantics']
            if generated_semantics.shape[-1] < semantics_all.shape[-1]:
                generated_semantics = F.pad(
                    generated_semantics,
                    (0, semantics_all.shape[-1] - generated_semantics.shape[-1]))

            means_step = torch.where(
                keep, warped, generated['means'][:, :current_count])
            opacity_step = torch.where(
                keep, original_opacity[:, :current_count],
                generated['opacities'][:, :current_count])
            semantics_step = torch.where(
                keep, semantics_all[:, :current_count],
                generated_semantics[:, :current_count])
            scales_step = torch.where(
                keep, scales_all[:, :current_count],
                generated['scales'][:, :current_count])
            cov_step = torch.where(
                keep_cov, cov_all[:, :current_count],
                generated['cov_inv'][:, :current_count])

            missing = self.target_num_gaussians - current_count
            if missing:
                tail = slice(current_count, self.target_num_gaussians)
                means_step = torch.cat([means_step, generated['means'][:, tail]], 1)
                opacity_step = torch.cat(
                    [opacity_step, generated['opacities'][:, tail]], 1)
                semantics_step = torch.cat(
                    [semantics_step, generated_semantics[:, tail]], 1)
                scales_step = torch.cat(
                    [scales_step, generated['scales'][:, tail]], 1)
                cov_step = torch.cat(
                    [cov_step, generated['cov_inv'][:, tail]], 1)

            if means_step.shape[1] != self.target_num_gaussians:
                raise AssertionError(
                    f'future real Gaussian count={means_step.shape[1]}, '
                    f'expected={self.target_num_gaussians}')
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
            expected_render = self.target_num_gaussians + int(
                self.with_emtpy and self.flow_include_empty)
            if means_step.shape[1] != expected_render:
                raise AssertionError(
                    f'future render Gaussian count={means_step.shape[1]}, '
                    f'expected={expected_render}')

            semantics = self.aggregator(
                sampled_xyz.clone().float(), means_step,
                opacity_step[..., 0], semantics_step, scales_step,
                cov_step)[None].transpose(1, 2)
            predictions.append([dict(
                pred_flow=semantics,
                sampled_label=metas['flow_info'][0][step]['occ_label'],
                flow_valid_flag=metas['flow_info'][0][step]['flow_valid_flag'],
                frontier_ratio=(
                    (~inside).sum() + missing) / self.target_num_gaussians,
                num_real_gaussians=self.target_num_gaussians,
                num_render_gaussians=expected_render,
                image_visible_ratio=generated['image_visible_ratio'])])
        return predictions