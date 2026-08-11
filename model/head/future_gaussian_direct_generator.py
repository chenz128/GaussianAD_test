import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.utils import get_rotation_matrix


class FutureGaussianDirectGenerator(nn.Module):
    """Directly decode a shared 3-second Gaussian bank without attribute bases."""

    _ALPHA_U = 0.7548776662466927
    _ALPHA_V = 0.5698402909980532
    _ALPHA_T = 0.6180339887498949

    def __init__(
        self,
        num_gaussians=12800,
        pc_range=(-30.0, -30.0, -2.0, 30.0, 30.0, 2.0),
        num_classes=17,
        embed_dims=64,
        temporal_in_dims=128,
        image_in_dims=128,
        num_frames=4,
        current_frame_index=0,
        scale_range=(0.08, 0.64),
        min_band=0.5,
        front_fraction=0.75,
        responsibility_size=1.0,
        initial_opacity=0.02,
        detach_context=True,
    ):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.temporal_in_dims = temporal_in_dims
        self.num_frames = num_frames
        self.current_frame_index = current_frame_index
        self.min_band = min_band
        self.front_fraction = front_fraction
        self.responsibility_size = responsibility_size
        self.detach_context = detach_context
        self.register_buffer(
            'pc_range', torch.tensor(pc_range, dtype=torch.float32), False)
        self.register_buffer(
            'scale_range', torch.tensor(scale_range, dtype=torch.float32), False)

        # Region center(3), normalized enter time(1), full 6-step ego path(12).
        self.query_encoder = nn.Sequential(
            nn.Linear(16, embed_dims), nn.LayerNorm(embed_dims), nn.SiLU(),
            nn.Linear(embed_dims, embed_dims))
        self.query_embedding = nn.Embedding(num_gaussians, embed_dims)
        self.temporal_proj = nn.Linear(temporal_in_dims, embed_dims)
        self.temporal_frame_embedding = nn.Embedding(num_frames, embed_dims)
        self.image_proj = nn.Conv2d(image_in_dims, embed_dims, 1)
        self.image_frame_embedding = nn.Embedding(num_frames, embed_dims)
        self.image_temporal_fusion = nn.Sequential(
            nn.Linear(num_frames * embed_dims, embed_dims),
            nn.LayerNorm(embed_dims), nn.SiLU())
        self.fusion = nn.Sequential(
            nn.Linear(embed_dims * 3, embed_dims * 2),
            nn.LayerNorm(embed_dims * 2), nn.SiLU(),
            nn.Linear(embed_dims * 2, 11 + num_classes))

        # Neutral direct-output initialization. These are numerical initializers,
        # not attributes copied from current Gaussians.
        nn.init.normal_(self.fusion[-1].weight, std=1e-3)
        nn.init.zeros_(self.fusion[-1].bias)
        with torch.no_grad():
            scale_target = (0.20 - scale_range[0]) / (
                scale_range[1] - scale_range[0])
            scale_logit = math.log(scale_target / (1.0 - scale_target))
            self.fusion[-1].bias[3:6].fill_(scale_logit)
            self.fusion[-1].bias[6] = 1.0
            self.fusion[-1].bias[10] = math.log(
                initial_opacity / (1.0 - initial_opacity))

    def _sample_responsibility_regions(self, ego_cumulative,
                                       future_to_current_rotations=None):
        """Create deterministic future-strip regions in current coordinates."""
        batch = ego_cumulative.shape[0]
        device, dtype = ego_cumulative.device, ego_cumulative.dtype
        lo = self.pc_range[:3].to(device=device, dtype=dtype)
        hi = self.pc_range[3:].to(device=device, dtype=dtype)
        span = hi - lo
        index = torch.arange(self.num_gaussians, device=device, dtype=dtype)
        u = torch.frac(index * self._ALPHA_U)[None].expand(batch, -1)
        v = torch.frac(index * self._ALPHA_V)[None].expand(batch, -1)
        step = torch.floor(torch.frac(index * self._ALPHA_T) * 6).long()
        step = step.clamp_max(5)[None].expand(batch, -1)
        gather_index = step[..., None].expand(-1, -1, 3)
        ego = torch.gather(ego_cumulative, 1, gather_index)

        if future_to_current_rotations is None:
            future_to_current_rotations = torch.eye(
                3, device=device, dtype=dtype
            ).reshape(1, 1, 3, 3).expand(batch, 6, -1, -1)
        rotation_index = step[..., None, None].expand(-1, -1, 3, 3)
        future_to_current = torch.gather(
            future_to_current_rotations, 1, rotation_index)
        motion_future = torch.matmul(
            future_to_current.transpose(-1, -2), ego[..., None]
        ).squeeze(-1)

        dx, dy = motion_future[..., 0], motion_future[..., 1]
        band_x = dx.abs().clamp(self.min_band, float(span[0]))
        band_y = dy.abs().clamp(self.min_band, float(span[1]))
        dominant_x = dx.abs() >= dy.abs()

        # front_fraction reserves most queries for the leading strip. Remaining
        # queries cover the orthogonal strip, which is important during turns.
        front_query = torch.frac(index * 0.4142135623730950)[None] < self.front_fraction
        use_x = torch.where(front_query, dominant_x, ~dominant_x)

        x_local_lo = torch.where(dx >= 0, hi[0] - band_x, lo[0])
        x_local_hi = torch.where(dx >= 0, hi[0], lo[0] + band_x)
        y_local_lo = torch.where(dy >= 0, hi[1] - band_y, lo[1])
        y_local_hi = torch.where(dy >= 0, hi[1], lo[1] + band_y)

        # Concentrate the lateral coordinate toward the motion centerline while
        # retaining full side coverage.
        lateral = 0.5 + 4.0 * (v - 0.5).pow(3)
        x_future = torch.where(
            use_x, x_local_lo + u * (x_local_hi - x_local_lo),
            lo[0] + lateral * span[0])
        y_future = torch.where(
            use_x, lo[1] + lateral * span[1],
            y_local_lo + u * (y_local_hi - y_local_lo))

        local_center = torch.stack(
            [x_future, y_future, torch.zeros_like(x_future)], dim=-1)
        center = torch.matmul(
            future_to_current, local_center[..., None]
        ).squeeze(-1) + ego
        center_xy = center[..., :2]
        half = center_xy.new_full(center_xy.shape, self.responsibility_size / 2)
        region_lo = center_xy - half
        region_hi = center_xy + half
        enter_time = (step.to(dtype) + 1.0) / 6.0
        return center_xy, region_lo, region_hi, enter_time

    def _temporal_context(self, query, features, indices, batch_size):
        if features.shape[-1] != self.temporal_in_dims:
            raise AssertionError(
                f'temporal feature width={features.shape[-1]}, expected '
                f'{self.temporal_in_dims}')
        features = features.detach() if self.detach_context else features
        projected = self.temporal_proj(features)
        frame_context = projected.new_zeros(
            batch_size, self.num_frames, self.embed_dims)
        counts = projected.new_zeros(batch_size, self.num_frames, 1)
        flat = indices[:, 0].long() * self.num_frames + indices[:, 1].long()
        frame_context.view(-1, self.embed_dims).scatter_add_(
            0, flat[:, None].expand(-1, self.embed_dims), projected)
        counts.view(-1, 1).scatter_add_(
            0, flat[:, None], counts.new_ones(flat.numel(), 1))
        frame_context = frame_context / counts.clamp_min(1.0)
        frame_context = frame_context + self.temporal_frame_embedding.weight[None]
        scores = torch.einsum('bnc,bfc->bnf', query, frame_context)
        scores = scores * (self.embed_dims ** -0.5)
        scores = scores.masked_fill(counts[:, None, :, 0] <= 0, -1e4)
        weights = scores.softmax(dim=-1)
        return torch.einsum('bnf,bfc->bnc', weights, frame_context)

    def _image_context(self, xyz_current, ms_img_feats, metas, batch_size):
        feature = ms_img_feats[0]
        frames = feature.shape[0] // batch_size
        cameras, _, height, width = feature.shape[1:]
        feature = feature.reshape(
            batch_size * frames * cameras, *feature.shape[2:])
        if self.detach_context:
            feature = feature.detach()
        feature = self.image_proj(feature).reshape(
            batch_size, frames, cameras, self.embed_dims, height, width)

        projection = metas['projection_mat'].reshape(
            batch_size, frames, cameras, 4, 4)
        image_wh = metas['image_wh'].reshape(
            batch_size, frames, cameras, 2)
        lidar2global = metas['lidar2global']
        if not torch.is_tensor(lidar2global):
            lidar2global = torch.as_tensor(lidar2global)
        lidar2global = lidar2global.to(xyz_current).reshape(
            batch_size, frames, 4, 4)
        current_index = self.current_frame_index % frames
        current_to_frame = torch.linalg.inv(lidar2global) @ (
            lidar2global[:, current_index:current_index + 1])

        points = torch.cat(
            [xyz_current, torch.ones_like(xyz_current[..., :1])], dim=-1)
        points_frame = torch.matmul(
            current_to_frame[:, :, None], points[:, None, ..., None])
        camera_points = torch.matmul(
            projection[:, :, :, None], points_frame[:, :, None]).squeeze(-1)
        depth = camera_points[..., 2]
        pixels = camera_points[..., :2] / depth.clamp_min(1e-5)[..., None]
        normalized = pixels / image_wh[:, :, :, None].clamp_min(1.0)
        visible = ((depth > 1e-5) & (normalized[..., 0] >= 0)
                   & (normalized[..., 0] <= 1) & (normalized[..., 1] >= 0)
                   & (normalized[..., 1] <= 1))
        grid = (normalized * 2 - 1).reshape(
            batch_size * frames * cameras, -1, 1, 2)
        sampled = F.grid_sample(
            feature.reshape(-1, self.embed_dims, height, width), grid,
            align_corners=False).reshape(
                batch_size, frames, cameras, self.embed_dims, -1)
        sampled = sampled.permute(0, 4, 1, 2, 3)
        mask = visible.permute(0, 3, 1, 2)[..., None]
        per_frame = (sampled * mask).sum(dim=3) / mask.sum(dim=3).clamp_min(1)
        frame_visible = mask.any(dim=3).to(per_frame.dtype)
        frame_embedding = self.image_frame_embedding.weight[None, None]
        per_frame = per_frame + frame_visible * frame_embedding
        context = self.image_temporal_fusion(per_frame.flatten(-2))
        image_visible = mask.any(dim=3).any(dim=2).to(context.dtype)
        return context, image_visible

    @staticmethod
    def covariance_inverse(scales, rotations):
        rotation = get_rotation_matrix(rotations)
        inverse_scale = torch.diag_embed(scales.clamp_min(1e-4).pow(-2))
        return rotation.transpose(-1, -2) @ inverse_scale @ rotation

    def forward(self, ego_cumulative, temporal_features, temporal_indices,
                ms_img_feats, metas, batch_size,
                future_to_current_rotations=None):
        center_xy, region_lo, region_hi, enter_time = (
            self._sample_responsibility_regions(
                ego_cumulative, future_to_current_rotations))
        center_z = center_xy.new_zeros(*center_xy.shape[:-1], 1)
        center = torch.cat([center_xy, center_z], dim=-1)
        ego_flat = ego_cumulative[..., :2].reshape(batch_size, 1, 12)
        ego_flat = ego_flat.expand(-1, self.num_gaussians, -1)
        pc_center = (self.pc_range[:3] + self.pc_range[3:]) / 2
        pc_half = (self.pc_range[3:] - self.pc_range[:3]) / 2
        normalized_center = (center - pc_center.to(center)) / pc_half.to(center)
        query_input = torch.cat(
            [normalized_center, enter_time[..., None], ego_flat / 60.0], dim=-1)
        query = self.query_encoder(query_input)
        query = query + self.query_embedding.weight[None]

        temporal = self._temporal_context(
            query, temporal_features, temporal_indices, batch_size)
        image, image_visible = self._image_context(
            center, ms_img_feats, metas, batch_size)
        raw = self.fusion(torch.cat([query, temporal, image], dim=-1))

        xy = region_lo + (region_hi - region_lo) * torch.sigmoid(raw[..., :2])
        lo, hi = self.pc_range[:3].to(raw), self.pc_range[3:].to(raw)
        z = lo[2] + (hi[2] - lo[2]) * torch.sigmoid(raw[..., 2:3])
        means = torch.cat([xy, z], dim=-1)
        scale_lo, scale_hi = self.scale_range.to(raw)
        scales = scale_lo + (scale_hi - scale_lo) * torch.sigmoid(raw[..., 3:6])
        rotations = F.normalize(raw[..., 6:10], dim=-1)
        opacities = torch.sigmoid(raw[..., 10:11])
        semantics = F.softplus(raw[..., 11:])
        return dict(
            means=means, scales=scales, rotations=rotations,
            opacities=opacities, semantics=semantics,
            cov_inv=self.covariance_inverse(scales, rotations),
            image_visible_ratio=image_visible.mean().detach(),
            enter_time=enter_time,
        )