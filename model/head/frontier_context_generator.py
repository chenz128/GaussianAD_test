import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.utils import get_rotation_matrix


class FrontierContextGenerator(nn.Module):
    """Generate frontier attributes from local 3D and visible image context."""

    _ALPHA_U = 0.7548776662466927
    _ALPHA_V = 0.5698402909980532
    _ALPHA_W = 0.6180339887498949

    def __init__(
        self,
        pc_range=(-30.0, -30.0, -2.0, 30.0, 30.0, 2.0),
        num_classes=17,
        context_dims=64,
        image_in_dims=128,
        bev_size=30,
        local_radius=1,
        scale_range=(0.08, 0.64),
        min_band=0.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.context_dims = context_dims
        self.bev_size = bev_size
        self.local_radius = local_radius
        self.min_band = min_band
        self.register_buffer(
            'pc_range', torch.tensor(pc_range, dtype=torch.float32), False)
        self.register_buffer(
            'scale_range', torch.tensor(scale_range, dtype=torch.float32), False)

        # z + scale(3) + quaternion(4) + opacity + semantics(C)
        self.raw_dims = 9 + num_classes
        self.gaussian_value = nn.Sequential(
            nn.Linear(self.raw_dims, context_dims), nn.LayerNorm(context_dims),
            nn.SiLU())
        self.query = nn.Sequential(
            nn.Linear(7, context_dims), nn.LayerNorm(context_dims), nn.SiLU(),
            nn.Linear(context_dims, context_dims))
        self.image_proj = nn.Conv2d(image_in_dims, context_dims, 1)
        self.image_gate = nn.Sequential(
            nn.Linear(context_dims * 2 + 1, context_dims), nn.Sigmoid())
        self.fusion = nn.Sequential(
            nn.Linear(context_dims * 2, context_dims),
            nn.LayerNorm(context_dims), nn.SiLU(),
            nn.Linear(context_dims, 3 + 4 + 1 + num_classes))
        nn.init.normal_(self.fusion[-1].weight, std=1e-3)
        nn.init.zeros_(self.fusion[-1].bias)

        offsets = []
        for dy in range(-local_radius, local_radius + 1):
            for dx in range(-local_radius, local_radius + 1):
                offsets.append((dx, dy))
        self.register_buffer(
            'local_offsets', torch.tensor(offsets, dtype=torch.long), False)

    def prepare_image_features(self, feature_map):
        """Project current-frame FPN features before six future queries."""
        batch, cameras, channels, height, width = feature_map.shape
        projected = self.image_proj(
            feature_map.reshape(batch * cameras, channels, height, width))
        return projected.reshape(
            batch, cameras, self.context_dims, height, width)

    def sample_frontier(self, ego_disp, num_gaussians):
        device, dtype = ego_disp.device, ego_disp.dtype
        lo = self.pc_range[:3].to(device=device, dtype=dtype)
        hi = self.pc_range[3:].to(device=device, dtype=dtype)
        span = hi - lo
        index = torch.arange(num_gaussians, device=device, dtype=dtype)
        u = torch.frac(index * self._ALPHA_U)[None]
        v = torch.frac(index * self._ALPHA_V)[None]
        selector = torch.frac(index * self._ALPHA_W)[None]

        dx, dy = ego_disp[:, 0:1], ego_disp[:, 1:2]
        band_x = dx.abs().clamp(self.min_band, span[0])
        band_y = dy.abs().clamp(0.0, span[1])
        x_lo = torch.where(dx >= 0, hi[0] - band_x, lo[0])
        x_hi = torch.where(dx >= 0, hi[0], lo[0] + band_x)
        y_lo = torch.where(dy >= 0, hi[1] - band_y, lo[1])
        y_hi = torch.where(dy >= 0, hi[1], lo[1] + band_y)
        area_x, area_y = band_x * span[1], band_y * span[0]
        use_x = selector < area_x / (area_x + area_y + 1e-6)
        x = torch.where(
            use_x, x_lo + u * (x_hi - x_lo), lo[0] + u * span[0])
        y = torch.where(
            use_x, lo[1] + v * span[1], y_lo + v * (y_hi - y_lo))
        # z is replaced by the attended local static geometry in forward().
        return torch.stack([x, y, torch.zeros_like(x)], dim=-1)

    def _pool_gaussians(self, means, scales, rotations, opacities,
                        semantics, valid):
        batch, count = means.shape[:2]
        lo, hi = self.pc_range[:3].to(means), self.pc_range[3:].to(means)
        xy = (means[..., :2] - lo[:2]) / (hi[:2] - lo[:2])
        cell = (xy * self.bev_size).long().clamp(0, self.bev_size - 1)
        flat_cell = cell[..., 1] * self.bev_size + cell[..., 0]
        raw = torch.cat([
            means[..., 2:3], scales, rotations, opacities, semantics], dim=-1)
        cells = self.bev_size * self.bev_size
        pooled = raw.new_zeros(batch, cells, self.raw_dims)
        counts = raw.new_zeros(batch, cells, 1)
        for batch_index in range(batch):
            mask = valid[batch_index]
            indices = flat_cell[batch_index, mask]
            pooled[batch_index].scatter_add_(
                0, indices[:, None].expand(-1, self.raw_dims),
                raw[batch_index, mask])
            counts[batch_index].scatter_add_(
                0, indices[:, None], counts.new_ones(indices.numel(), 1))
        pooled = pooled / counts.clamp_min(1.0)
        return pooled, counts

    def _attend_gaussians(self, xyz, ego_disp, time_index, pooled, counts):
        lo, hi = self.pc_range[:3].to(xyz), self.pc_range[3:].to(xyz)
        norm_xyz = (xyz - (lo + hi) / 2) / ((hi - lo) / 2)
        norm_ego = (ego_disp / (hi - lo))[:, None].expand_as(xyz)
        time = xyz.new_full((*xyz.shape[:2], 1), (time_index + 1) / 6)
        query = self.query(torch.cat([norm_xyz, norm_ego, time], dim=-1))

        xy = (xyz[..., :2] - lo[:2]) / (hi[:2] - lo[:2])
        center = (xy * self.bev_size).long().clamp(0, self.bev_size - 1)
        neighbor = center[:, :, None] + self.local_offsets[None, None]
        neighbor = neighbor.clamp(0, self.bev_size - 1)
        flat = neighbor[..., 1] * self.bev_size + neighbor[..., 0]
        batch_index = torch.arange(xyz.shape[0], device=xyz.device)[:, None, None]
        local_raw = pooled[batch_index, flat]
        local_count = counts[batch_index, flat, 0]
        local_value = self.gaussian_value(local_raw)
        scores = (query[:, :, None] * local_value).sum(-1) * (
            self.context_dims ** -0.5)
        scores = scores.masked_fill(local_count <= 0, -1e4)
        weights = scores.softmax(dim=-1)
        has_local = (local_count > 0).any(dim=-1, keepdim=True)

        context = (weights[..., None] * local_value).sum(dim=2)
        base = (weights[..., None] * local_raw).sum(dim=2)
        global_count = counts.sum(dim=1).clamp_min(1.0)
        global_raw = (pooled * counts).sum(dim=1) / global_count
        global_context = self.gaussian_value(global_raw)[:, None].expand_as(context)
        context = torch.where(has_local, context, global_context)
        base = torch.where(
            has_local, base, global_raw[:, None].expand_as(base))
        return query, context, base

    def _sample_images(self, xyz_current, image_features, projection_mat,
                       image_wh):
        batch, cameras, channels, height, width = image_features.shape
        points = torch.cat(
            [xyz_current, torch.ones_like(xyz_current[..., :1])], dim=-1)
        camera_points = torch.matmul(
            projection_mat[:, :, None], points[:, None, ..., None]).squeeze(-1)
        depth = camera_points[..., 2]
        pixels = camera_points[..., :2] / depth.clamp_min(1e-5)[..., None]
        normalized = pixels / image_wh[:, :, None].clamp_min(1.0)
        visible = ((depth > 1e-5) & (normalized[..., 0] >= 0)
                   & (normalized[..., 0] <= 1) & (normalized[..., 1] >= 0)
                   & (normalized[..., 1] <= 1))
        grid = (normalized * 2 - 1).reshape(
            batch * cameras, -1, 1, 2)
        sampled = F.grid_sample(
            image_features.reshape(batch * cameras, channels, height, width),
            grid, align_corners=False).reshape(
                batch, cameras, channels, -1).permute(0, 3, 1, 2)
        mask = visible.permute(0, 2, 1)[..., None]
        visual = (sampled * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1)
        visibility = mask.any(dim=2).to(visual.dtype)
        return visual, visibility

    @staticmethod
    def covariance_inverse(scales, rotations):
        rotation = get_rotation_matrix(rotations)
        inverse_scale = torch.diag_embed(scales.clamp_min(1e-4).pow(-2))
        return rotation.transpose(-1, -2) @ inverse_scale @ rotation

    def forward(self, ego_disp, num_gaussians, time_index, context_gaussian,
                context_valid, image_features, projection_mat, image_wh):
        xyz = self.sample_frontier(ego_disp, num_gaussians)
        pooled, counts = self._pool_gaussians(
            context_gaussian['means'].detach(),
            context_gaussian['scales'].detach(),
            context_gaussian['rotations'].detach(),
            context_gaussian['opacities'].detach(),
            context_gaussian['semantics'].detach(), context_valid)
        query, gaussian_context, base = self._attend_gaussians(
            xyz, ego_disp, time_index, pooled, counts)

        # Frontier coordinates are in the future ego frame. The current code's
        # ego compensation is translational, so invert it before image projection.
        xyz_current = xyz + ego_disp[:, None]
        visual, visible = self._sample_images(
            xyz_current, image_features, projection_mat, image_wh)
        gate = self.image_gate(torch.cat(
            [query, gaussian_context, visible], dim=-1)) * visible
        fused = gaussian_context + gate * visual
        residual = self.fusion(torch.cat([query, fused], dim=-1))

        xyz = torch.cat([xyz[..., :2], base[..., :1]], dim=-1)
        lo, hi = self.pc_range[:3].to(xyz), self.pc_range[3:].to(xyz)
        xyz = torch.max(torch.min(xyz, hi - 1e-3), lo)
        base_scale = base[..., 1:4].clamp(
            float(self.scale_range[0]), float(self.scale_range[1]))
        scales = (base_scale + 0.05 * torch.tanh(residual[..., :3])).clamp(
            float(self.scale_range[0]), float(self.scale_range[1]))
        rotations = F.normalize(
            base[..., 4:8] + 0.1 * torch.tanh(residual[..., 3:7]), dim=-1)
        base_opacity = base[..., 8:9].clamp(1e-4, 1 - 1e-4)
        opacity_logit = torch.logit(base_opacity)
        opacities = torch.sigmoid(opacity_logit + residual[..., 7:8])
        base_semantics = base[..., 9:].clamp_min(1e-4)
        semantics = F.softplus(
            torch.log(base_semantics) + residual[..., 8:])
        return dict(
            means=xyz, scales=scales, rotations=rotations,
            opacities=opacities, semantics=semantics,
            cov_inv=self.covariance_inverse(scales, rotations),
            image_visible_ratio=visible.mean().detach())