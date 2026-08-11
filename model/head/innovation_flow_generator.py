import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.utils import get_rotation_matrix


class ResidualVelocityBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)

    def forward(self, feature):
        identity = feature
        feature = self.conv1(F.silu(self.norm1(feature)))
        feature = self.conv2(F.silu(self.norm2(feature)))
        return feature + identity


class InnovationFlowGenerator(nn.Module):
    """Conditional flow matching over future innovation OCC latents."""

    _ALPHA_U = 0.7548776662466927
    _ALPHA_V = 0.5698402909980532
    _ALPHA_T = 0.6180339887498949
    _STATIC_CLASSES = (1, 8, 11, 12, 13, 14, 15, 16)
    _DYNAMIC_CLASSES = (2, 3, 4, 5, 6, 7, 9, 10)

    def __init__(
        self,
        num_gaussians=12800,
        pc_range=(-30.0, -30.0, -2.0, 30.0, 30.0, 2.0),
        num_classes=17,
        latent_dims=64,
        query_dims=64,
        temporal_in_dims=128,
        image_in_dims=128,
        num_frames=4,
        current_frame_index=0,
        scale_range=(0.08, 0.64),
        min_band=0.5,
        front_fraction=0.75,
        responsibility_size=1.0,
        initial_opacity=0.02,
        num_flow_steps=4,
        match_radius=0.76,
        target_pose_mode='translation',
        detach_context=False,
    ):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.num_classes = num_classes
        self.latent_dims = latent_dims
        self.query_dims = query_dims
        self.temporal_in_dims = temporal_in_dims
        self.image_in_dims = image_in_dims
        self.num_frames = num_frames
        self.current_frame_index = current_frame_index
        self.min_band = min_band
        self.front_fraction = front_fraction
        self.responsibility_size = responsibility_size
        self.num_flow_steps = num_flow_steps
        self.match_radius = match_radius
        if target_pose_mode not in ('translation', 'se3'):
            raise ValueError(
                f'unsupported target_pose_mode={target_pose_mode!r}')
        self.target_pose_mode = target_pose_mode
        self.detach_context = detach_context
        self.register_buffer(
            'pc_range', torch.tensor(pc_range, dtype=torch.float32), False)
        self.register_buffer(
            'scale_range', torch.tensor(scale_range, dtype=torch.float32), False)

        self.semantic_embedding = nn.Embedding(18, 16)
        self.innovation_encoder = nn.Sequential(
            nn.Conv2d(8 * 16, latent_dims, 3, stride=2, padding=1),
            nn.GroupNorm(8, latent_dims), nn.SiLU(),
            nn.Conv2d(latent_dims, latent_dims, 3, stride=2, padding=1),
            nn.GroupNorm(8, latent_dims), nn.SiLU(),
        )
        self.endpoint_temporal = nn.Sequential(
            nn.Conv3d(latent_dims, latent_dims, 3, padding=1),
            nn.GroupNorm(8, latent_dims), nn.SiLU())

        self.temporal_proj = nn.Linear(temporal_in_dims, latent_dims)
        self.image_proj = nn.Linear(image_in_dims, latent_dims)
        self.ego_proj = nn.Sequential(
            nn.Linear(3, latent_dims), nn.SiLU(),
            nn.Linear(latent_dims, latent_dims))
        self.condition_fusion = nn.Sequential(
            nn.Linear(latent_dims * 3, latent_dims),
            nn.LayerNorm(latent_dims), nn.SiLU())
        self.spatial_embedding = nn.Parameter(
            torch.zeros(1, latent_dims, 1, 30, 30))

        self.time_mlp = nn.Sequential(
            nn.Linear(latent_dims, latent_dims * 2), nn.SiLU(),
            nn.Linear(latent_dims * 2, latent_dims))
        self.velocity_in = nn.Conv3d(latent_dims, latent_dims, 3, padding=1)
        self.velocity_blocks = nn.ModuleList([
            ResidualVelocityBlock(latent_dims) for _ in range(4)])
        self.velocity_out = nn.Conv3d(latent_dims, latent_dims, 3, padding=1)

        self.query_encoder = nn.Sequential(
            nn.Linear(16, query_dims), nn.LayerNorm(query_dims), nn.SiLU(),
            nn.Linear(query_dims, query_dims))
        self.query_embedding = nn.Embedding(num_gaussians, query_dims)
        self.decoder = nn.Sequential(
            nn.Linear(query_dims + latent_dims * 2, query_dims * 2),
            nn.LayerNorm(query_dims * 2), nn.SiLU(),
            nn.Linear(query_dims * 2, 11 + num_classes))

        nn.init.normal_(self.spatial_embedding, std=0.02)
        nn.init.normal_(self.decoder[-1].weight, std=1e-3)
        nn.init.zeros_(self.decoder[-1].bias)
        with torch.no_grad():
            scale_target = (0.20 - scale_range[0]) / (
                scale_range[1] - scale_range[0])
            scale_logit = math.log(scale_target / (1.0 - scale_target))
            self.decoder[-1].bias[3:6].fill_(scale_logit)
            self.decoder[-1].bias[6] = 1.0
            self.decoder[-1].bias[10] = math.log(
                initial_opacity / (1.0 - initial_opacity))

        self.last_flow_matching_loss = None
        self.last_innovation_masks = None

    @staticmethod
    def _time_embedding(time, dims):
        half = dims // 2
        frequency = torch.exp(
            torch.arange(half, device=time.device, dtype=time.dtype)
            * (-math.log(10000.0) / max(half - 1, 1)))
        phase = time[:, None] * frequency[None]
        embedding = torch.cat([phase.sin(), phase.cos()], dim=-1)
        if embedding.shape[-1] < dims:
            embedding = F.pad(embedding, (0, dims - embedding.shape[-1]))
        return embedding

    def _sample_responsibility_regions(self, ego_cumulative):
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
        ego = torch.gather(
            ego_cumulative, 1, step[..., None].expand(-1, -1, 3))
        dx, dy = ego[..., 0], ego[..., 1]
        band_x = dx.abs().clamp(self.min_band, float(span[0]))
        band_y = dy.abs().clamp(self.min_band, float(span[1]))
        dominant_x = dx.abs() >= dy.abs()
        front_query = (
            torch.frac(index * 0.4142135623730950)[None]
            < self.front_fraction)
        use_x = torch.where(front_query, dominant_x, ~dominant_x)
        x_lo = torch.where(dx >= 0, hi[0] - band_x, lo[0])
        x_hi = torch.where(dx >= 0, hi[0], lo[0] + band_x)
        y_lo = torch.where(dy >= 0, hi[1] - band_y, lo[1])
        y_hi = torch.where(dy >= 0, hi[1], lo[1] + band_y)
        lateral = 0.5 + 4.0 * (v - 0.5).pow(3)
        x_future = torch.where(
            use_x, x_lo + u * (x_hi - x_lo), lo[0] + lateral * span[0])
        y_future = torch.where(
            use_x, lo[1] + lateral * span[1], y_lo + u * (y_hi - y_lo))
        center_xy = torch.stack([x_future + dx, y_future + dy], dim=-1)
        half_size = center_xy.new_full(
            center_xy.shape, self.responsibility_size / 2)
        enter_time = (step.to(dtype) + 1.0) / 6.0
        return center_xy, center_xy - half_size, center_xy + half_size, enter_time, step

    def _pool_temporal(self, features, indices, batch_size):
        if self.detach_context:
            features = features.detach()
        projected = self.temporal_proj(features)
        pooled = projected.new_zeros(batch_size, self.num_frames, self.latent_dims)
        counts = projected.new_zeros(batch_size, self.num_frames, 1)
        flat = indices[:, 0].long() * self.num_frames + indices[:, 1].long()
        pooled.view(-1, self.latent_dims).scatter_add_(
            0, flat[:, None].expand(-1, self.latent_dims), projected)
        counts.view(-1, 1).scatter_add_(
            0, flat[:, None], counts.new_ones(flat.numel(), 1))
        pooled = pooled / counts.clamp_min(1.0)
        return pooled.mean(dim=1)

    def _pool_images(self, ms_img_feats, batch_size):
        feature = ms_img_feats[0]
        frames = feature.shape[0] // batch_size
        if self.detach_context:
            feature = feature.detach()
        feature = feature.reshape(batch_size, frames, *feature.shape[1:])
        feature = feature.mean(dim=(1, 2, 4, 5))
        return self.image_proj(feature)

    def _condition(self, temporal_features, temporal_indices, ms_img_feats,
                   ego_cumulative, batch_size):
        temporal = self._pool_temporal(
            temporal_features, temporal_indices, batch_size)
        image = self._pool_images(ms_img_feats, batch_size)
        temporal = temporal[:, None].expand(-1, 6, -1)
        image = image[:, None].expand(-1, 6, -1)
        ego = self.ego_proj(ego_cumulative)
        condition = self.condition_fusion(
            torch.cat([temporal, image, ego], dim=-1))
        return condition.permute(0, 2, 1)[:, :, :, None, None]

    def _velocity(self, latent, time, condition):
        time_feature = self.time_mlp(
            self._time_embedding(time, self.latent_dims))
        feature = latent + condition + self.spatial_embedding
        feature = feature + time_feature[:, :, None, None, None]
        feature = self.velocity_in(feature)
        for block in self.velocity_blocks:
            feature = block(feature)
        return self.velocity_out(F.silu(feature))

    @staticmethod
    def _as_current_labels(metas, device):
        labels = metas['occ_label']
        if not torch.is_tensor(labels):
            labels = torch.as_tensor(labels)
        labels = labels.to(device=device, dtype=torch.long)
        return labels.reshape(labels.shape[0], -1)

    @staticmethod
    def _future_labels(metas, device):
        batch_labels = []
        for batch_info in metas['flow_info']:
            labels = [
                torch.as_tensor(frame['occ_label'], device=device).reshape(-1)
                for frame in batch_info]
            batch_labels.append(torch.stack(labels))
        return torch.stack(batch_labels).long()

    def _future_transforms(self, metas, reference):
        if self.target_pose_mode == 'translation':
            ego = metas['ego_fut_trajs']
            if not torch.is_tensor(ego):
                ego = torch.as_tensor(ego)
            ego = torch.nan_to_num(ego.to(reference).float()).cumsum(dim=1)
            transforms = torch.eye(
                4, device=reference.device, dtype=reference.dtype
            ).reshape(1, 1, 4, 4).repeat(ego.shape[0], 6, 1, 1)
            transforms[..., :2, 3] = -ego
            return transforms
        current = metas['lidar2global']
        future = metas['future_lidar2global']
        if not torch.is_tensor(current):
            current = torch.as_tensor(current)
        if not torch.is_tensor(future):
            future = torch.as_tensor(future)
        current = current.to(reference)
        future = future.to(reference)
        if current.dim() == 4:
            current = current[:, self.current_frame_index]
        if current.dim() == 2:
            current = current[None]
        if future.dim() == 3:
            future = future[None]
        return torch.linalg.inv(future) @ current[:, None]

    def _innovation_targets(self, metas, reference):
        current_labels = self._as_current_labels(metas, reference.device)
        future_labels = self._future_labels(metas, reference.device)
        transforms = self._future_transforms(metas, reference)
        batch = current_labels.shape[0]
        grid = torch.stack(torch.meshgrid(
            torch.arange(120, device=reference.device),
            torch.arange(120, device=reference.device),
            torch.arange(8, device=reference.device), indexing='ij'), -1)
        xyz = grid.to(reference).reshape(-1, 3) * 0.5
        xyz = xyz + reference.new_tensor([-29.75, -29.75, -1.75])
        cell_radius = int(math.ceil(self.match_radius / 0.5))
        offsets = torch.tensor([
            (x, y, z) for x in range(-cell_radius, cell_radius + 1)
            for y in range(-cell_radius, cell_radius + 1)
            for z in range(-cell_radius, cell_radius + 1)
            if 0.5 * math.sqrt(x * x + y * y + z * z)
            <= self.match_radius
        ], device=reference.device, dtype=torch.long)
        innovations = []
        masks = []
        for batch_index in range(batch):
            batch_targets = []
            batch_masks = []
            for step in range(6):
                transform = transforms[batch_index, step]
                warped = xyz @ transform[:3, :3].T + transform[:3, 3]
                index = torch.floor(
                    (warped - reference.new_tensor([-30.0, -30.0, -2.0]))
                    / 0.5).long()
                explained = torch.zeros(
                    17, xyz.shape[0], device=reference.device,
                    dtype=torch.bool)
                for cls in self._STATIC_CLASSES:
                    source = index[current_labels[batch_index] == cls]
                    if source.numel() == 0:
                        continue
                    neighbors = source[:, None] + offsets[None]
                    valid = ((neighbors >= 0) & (neighbors < neighbors.new_tensor(
                        [120, 120, 8]))).all(dim=-1)
                    neighbors = neighbors[valid]
                    flat = neighbors[:, 0] * 120 * 8 + neighbors[:, 1] * 8 + neighbors[:, 2]
                    explained[cls, flat] = True
                target = future_labels[batch_index, step]
                dynamic = torch.zeros_like(target, dtype=torch.bool)
                for cls in self._DYNAMIC_CLASSES:
                    dynamic |= target == cls
                static = torch.zeros_like(target, dtype=torch.bool)
                for cls in self._STATIC_CLASSES:
                    static |= target == cls
                flat_index = torch.arange(target.numel(), device=target.device)
                matched = explained[target.clamp(0, 16), flat_index]
                innovation = dynamic | (static & ~matched)
                innovation_target = torch.full_like(target, 17)
                innovation_target[innovation] = target[innovation]
                batch_targets.append(innovation_target)
                batch_masks.append(innovation)
            innovations.append(torch.stack(batch_targets))
            masks.append(torch.stack(batch_masks))
        return torch.stack(innovations), torch.stack(masks)

    def _encode_endpoint(self, innovation_labels):
        batch, steps = innovation_labels.shape[:2]
        labels = innovation_labels.reshape(batch * steps, 120, 120, 8)
        embedded = self.semantic_embedding(labels)
        embedded = embedded.permute(0, 3, 4, 1, 2).reshape(
            batch * steps, 8 * 16, 120, 120)
        latent = self.innovation_encoder(embedded)
        latent = latent.reshape(
            batch, steps, self.latent_dims, 30, 30).permute(0, 2, 1, 3, 4)
        return self.endpoint_temporal(latent)

    def _sample_flow(self, shape, condition, reference):
        latent = torch.randn(shape, device=reference.device, dtype=reference.dtype)
        step_size = 1.0 / self.num_flow_steps
        for step in range(self.num_flow_steps):
            time = latent.new_full((shape[0],), step * step_size)
            velocity = self._velocity(latent, time, condition)
            proposal = latent + step_size * velocity
            next_time = latent.new_full((shape[0],), (step + 1) * step_size)
            next_velocity = self._velocity(proposal, next_time, condition)
            latent = latent + 0.5 * step_size * (velocity + next_velocity)
        return latent

    def _decode(self, latent, condition, ego_cumulative):
        batch = latent.shape[0]
        center_xy, region_lo, region_hi, enter_time, step = (
            self._sample_responsibility_regions(ego_cumulative))
        lo = self.pc_range[:3].to(latent)
        hi = self.pc_range[3:].to(latent)
        grid = center_xy.clone()
        grid[..., 0] = 2.0 * (grid[..., 0] - lo[0]) / (hi[0] - lo[0]) - 1.0
        grid[..., 1] = 2.0 * (grid[..., 1] - lo[1]) / (hi[1] - lo[1]) - 1.0
        selected = []
        condition_frames = condition[:, :, :, 0, 0].permute(0, 2, 1)
        for batch_index in range(batch):
            frame_features = []
            for frame in range(6):
                query_mask = step[batch_index] == frame
                sampled = F.grid_sample(
                    latent[batch_index:batch_index + 1, :, frame],
                    grid[batch_index, query_mask][None, :, None],
                    align_corners=False).squeeze(0).squeeze(-1).T
                frame_features.append((query_mask, sampled))
            query_latent = latent.new_zeros(
                self.num_gaussians, self.latent_dims)
            for query_mask, sampled in frame_features:
                query_latent[query_mask] = sampled
            selected.append(query_latent)
        selected = torch.stack(selected)
        step_condition = torch.gather(
            condition_frames, 1,
            step[..., None].expand(-1, -1, self.latent_dims))
        ego_flat = ego_cumulative[..., :2].reshape(batch, 1, 12)
        ego_flat = ego_flat.expand(-1, self.num_gaussians, -1)
        center_z = center_xy.new_zeros(*center_xy.shape[:-1], 1)
        center = torch.cat([center_xy, center_z], dim=-1)
        pc_center = (lo + hi) / 2
        pc_half = (hi - lo) / 2
        query_input = torch.cat([
            (center - pc_center) / pc_half,
            enter_time[..., None], ego_flat / 60.0], dim=-1)
        query = self.query_encoder(query_input)
        query = query + self.query_embedding.weight[None]
        raw = self.decoder(torch.cat([
            query, selected, step_condition], dim=-1))
        xy = region_lo + (region_hi - region_lo) * torch.sigmoid(raw[..., :2])
        z = lo[2] + (hi[2] - lo[2]) * torch.sigmoid(raw[..., 2:3])
        means = torch.cat([xy, z], dim=-1)
        scale_lo, scale_hi = self.scale_range.to(raw)
        scales = scale_lo + (scale_hi - scale_lo) * torch.sigmoid(raw[..., 3:6])
        rotations = F.normalize(raw[..., 6:10], dim=-1)
        opacities = torch.sigmoid(raw[..., 10:11])
        semantics = F.softplus(raw[..., 11:])
        rotation = get_rotation_matrix(rotations)
        inverse_scale = torch.diag_embed(scales.clamp_min(1e-4).pow(-2))
        covariance_inverse = (
            rotation.transpose(-1, -2) @ inverse_scale @ rotation)
        return dict(
            means=means, scales=scales, rotations=rotations,
            opacities=opacities, semantics=semantics,
            cov_inv=covariance_inverse, enter_time=enter_time,
            image_visible_ratio=means.new_tensor(0.0))

    def forward(self, ego_cumulative, temporal_features, temporal_indices,
                ms_img_feats, metas, batch_size,
                future_to_current_rotations=None):
        condition = self._condition(
            temporal_features, temporal_indices, ms_img_feats,
            ego_cumulative, batch_size)
        if self.training:
            innovation_labels, innovation_masks = self._innovation_targets(
                metas, ego_cumulative)
            endpoint = self._encode_endpoint(innovation_labels)
            flow_target = endpoint.detach()
            noise = torch.randn_like(flow_target)
            time = torch.rand(batch_size, device=endpoint.device, dtype=endpoint.dtype)
            view = time.reshape(batch_size, 1, 1, 1, 1)
            interpolated = (1.0 - view) * noise + view * flow_target
            target_velocity = flow_target - noise
            predicted_velocity = self._velocity(interpolated, time, condition)
            self.last_flow_matching_loss = F.mse_loss(
                predicted_velocity.float(), target_velocity.float())
            latent = endpoint
            self.last_innovation_masks = innovation_masks
        else:
            latent = self._sample_flow(
                (batch_size, self.latent_dims, 6, 30, 30),
                condition, ego_cumulative)
            self.last_flow_matching_loss = ego_cumulative.sum() * 0.0
            self.last_innovation_masks = None
        return self._decode(latent, condition, ego_cumulative)