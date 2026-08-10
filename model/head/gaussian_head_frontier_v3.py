import torch
import torch.nn.functional as F

from mmengine.registry import MODELS

from .future_gaussian_direct_generator import FutureGaussianDirectGenerator
from .gaussian_head import GaussianHead


@MODELS.register_module()
class GaussianHeadFrontierV3(GaussianHead):
    """Render retained current Gaussians plus one shared direct future bank."""

    def __init__(self, target_num_gaussians=25600, direct_generator=None,
                 current_frame_index=0, min_current_gaussian_ratio=0.99,
                 dynamic_class_multiplier=3.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.target_num_gaussians = target_num_gaussians
        self.current_frame_index = current_frame_index
        self.min_current_gaussian_ratio = min_current_gaussian_ratio
        self.dynamic_class_multiplier = dynamic_class_multiplier
        config = dict(direct_generator or {})
        config.setdefault('pc_range', tuple(self.pc_range))
        config.setdefault(
            'num_classes', self.num_classes - 1 if self.with_emtpy
            else self.num_classes)
        config.setdefault('current_frame_index', current_frame_index)
        self.future_generator = FutureGaussianDirectGenerator(**config)

    def get_in_range_mask(self, points):
        grid = ((points - self.pc_min) / self.grid_size).to(torch.int)
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

        ego = metas['ego_fut_trajs']
        if not torch.is_tensor(ego):
            ego = torch.as_tensor(ego)
        ego = ego.to(offset.device).float()
        if ego.dim() == 2:
            ego = ego[None]
        ego = torch.nan_to_num(ego).cumsum(dim=1)
        ego = torch.cat(
            [ego, ego.new_zeros(*ego.shape[:-1], 1)], dim=-1)

        generated = self.future_generator(
            ego_cumulative=ego,
            temporal_features=kwargs['temporal_context_features'],
            temporal_indices=kwargs['temporal_context_indices'],
            ms_img_feats=kwargs['ms_img_feats'],
            metas=metas,
            batch_size=batch_size)

        original_opacity, semantics_all, scales_all, cov_all = gs
        generated_semantics = generated['semantics']
        if generated_semantics.shape[-1] < semantics_all.shape[-1]:
            generated_semantics = F.pad(
                generated_semantics,
                (0, semantics_all.shape[-1]
                 - generated_semantics.shape[-1]))

        predictions = []
        for step in range(6):
            warped_old = means_future[..., step, :] - ego[:, step:step + 1]
            old_inside = self.get_in_range_mask(warped_old)[0]
            new_means = generated['means'] - ego[:, step:step + 1]
            new_inside = self.get_in_range_mask(new_means)[0]
            entered = generated['enter_time'][0] <= ((step + 1) / 6.0)
            new_active = new_inside & entered

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
                cov_all[:, :current_count][:, old_inside],
                generated['cov_inv'][:, new_active]], dim=1)

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
            semantics = self.aggregator(
                sampled_xyz.clone().float(), means_step,
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