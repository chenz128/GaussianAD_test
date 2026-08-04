import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.utils import get_rotation_matrix


class FrontierGenerator(nn.Module):
    """Generate gaussians for the region that newly enters the occ window.

    A future frame is rendered in the ``t+k`` ego frame: current gaussians are
    shifted by ``-ego_displacement``, so the band they vacate at the leading
    edge has no contributor at all. This module fills exactly that band, keeping
    the total gaussian count fixed.
    """

    # Low-discrepancy multipliers (Kronecker / golden-ratio sequences) so slot
    # positions are spread out deterministically without a GPU sync.
    _ALPHA_U = 0.7548776662466927
    _ALPHA_V = 0.5698402909980532
    _ALPHA_W = 0.6180339887498949

    def __init__(
        self,
        pc_range=(-30.0, -30.0, -2.0, 30.0, 30.0, 2.0),
        num_classes=17,
        hidden_dims=256,
        scale_range=(0.08, 0.64),
        max_position_delta=1.0,
        min_band=0.5,
        init_scale=0.2,
        init_opacity=0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.max_position_delta = max_position_delta
        self.min_band = min_band
        self.register_buffer(
            'pc_range', torch.tensor(pc_range, dtype=torch.float32), False)
        self.register_buffer(
            'scale_range', torch.tensor(scale_range, dtype=torch.float32), False)

        # delta_xyz(3) + scale(3) + quat(4) + opacity(1) + semantics(num_classes)
        self.out_dims = 3 + 3 + 4 + 1 + num_classes
        self.net = nn.Sequential(
            nn.Linear(7, hidden_dims),
            nn.LayerNorm(hidden_dims),
            nn.SiLU(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.LayerNorm(hidden_dims),
            nn.SiLU(),
            nn.Linear(hidden_dims, self.out_dims),
        )
        self._init_head(init_scale, init_opacity)

    def _init_head(self, init_scale, init_opacity):
        # Zero-init so the first forward emits the geometric prior verbatim.
        head = self.net[-1]
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
        lo, hi = float(self.scale_range[0]), float(self.scale_range[1])
        ratio = min(max((init_scale - lo) / (hi - lo), 1e-4), 1 - 1e-4)
        head.bias.data[3:6] = math.log(ratio / (1 - ratio))
        head.bias.data[10] = math.log(init_opacity / (1 - init_opacity))

    def sample_frontier(self, ego_disp, num_gaussians):
        """Deterministic stratified sampling inside the newly-visible band.

        ego_disp: (B, 3) cumulative ego displacement for this future step.
        returns:  (B, G, 3) candidate positions in the ``t+k`` ego frame.
        """
        device, dtype = ego_disp.device, ego_disp.dtype
        lo = self.pc_range[:3].to(device=device, dtype=dtype)
        hi = self.pc_range[3:].to(device=device, dtype=dtype)
        span = hi - lo

        idx = torch.arange(num_gaussians, device=device, dtype=dtype)
        u = torch.frac(idx * self._ALPHA_U)[None]        # (1, G)
        v = torch.frac(idx * self._ALPHA_V)[None]
        w = torch.frac(idx * self._ALPHA_W)[None]

        dx = ego_disp[:, 0:1]                            # (B, 1)
        dy = ego_disp[:, 1:2]
        band_x = dx.abs().clamp(self.min_band, span[0])
        band_y = dy.abs().clamp(0.0, span[1])

        forward_x = dx >= 0
        x_lo = torch.where(forward_x, hi[0] - band_x, lo[0])
        x_hi = torch.where(forward_x, hi[0], lo[0] + band_x)
        forward_y = dy >= 0
        y_lo = torch.where(forward_y, hi[1] - band_y, lo[1])
        y_hi = torch.where(forward_y, hi[1], lo[1] + band_y)

        # Split slots between the longitudinal and lateral bands by area.
        area_x = band_x * span[1]
        area_y = band_y * span[0]
        p_x = area_x / (area_x + area_y + 1e-6)          # (B, 1)
        use_x = w < p_x                                  # (B, G)

        px = torch.where(use_x, x_lo + u * (x_hi - x_lo), lo[0] + u * span[0])
        py = torch.where(use_x, lo[1] + v * span[1], y_lo + v * (y_hi - y_lo))
        pz = (lo[2] + w * span[2]).expand_as(px)
        return torch.stack([px, py, pz], dim=-1)

    @staticmethod
    def covariance_inverse(scales, rotations):
        rot = get_rotation_matrix(rotations)
        inv_scale_sq = torch.diag_embed(scales.clamp_min(1e-4).pow(-2))
        return rot.transpose(-1, -2) @ inv_scale_sq @ rot

    def forward(self, ego_disp, num_gaussians, time_index, num_steps=6):
        xyz = self.sample_frontier(ego_disp, num_gaussians)
        lo = self.pc_range[:3].to(xyz)
        hi = self.pc_range[3:].to(xyz)
        norm_xyz = (xyz - (hi + lo) / 2) / ((hi - lo) / 2)
        norm_ego = (ego_disp / (hi - lo))[:, None].expand_as(xyz)
        t = xyz.new_full((*xyz.shape[:2], 1), (time_index + 1) / num_steps)

        out = self.net(torch.cat([norm_xyz, norm_ego, t], dim=-1))

        means = xyz + torch.tanh(out[..., :3]) * self.max_position_delta
        means = torch.max(torch.min(means, hi), lo)
        scales = self.scale_range[0] + torch.sigmoid(out[..., 3:6]) * (
            self.scale_range[1] - self.scale_range[0])
        rotations = F.normalize(
            out[..., 6:10] + out.new_tensor([1.0, 0.0, 0.0, 0.0]), dim=-1)
        opacities = torch.sigmoid(out[..., 10:11])
        semantics = F.softplus(out[..., 11:])

        return {
            'means': means,
            'scales': scales,
            'rotations': rotations,
            'opacities': opacities,
            'semantics': semantics,
            'cov_inv': self.covariance_inverse(scales, rotations),
        }
