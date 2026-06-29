import torch, torch.nn as nn
import numpy as np
from mmseg.registry import MODELS
from .base_lifter import BaseLifter
from ..utils.safe_ops import safe_inverse_sigmoid


@MODELS.register_module()
class GaussianLifter(BaseLifter):
    def __init__(
        self,
        num_anchor,
        embed_dims,
        anchor_grad=True,
        feat_grad=True,
        phi_activation='sigmoid',
        semantics=False,
        semantic_dim=None,
        include_opa=True,
        offset=False,
        offset_dim=2*6,
        pts_init=False,
        pc_range=None,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.pts_init = pts_init

        if pts_init:
            assert pc_range is not None, "pc_range is required for pts_init"
            self.pc_range = pc_range
        else:
            xyz = torch.rand(num_anchor, 3, dtype=torch.float)
            xyz = safe_inverse_sigmoid(xyz)

        scale = torch.rand(num_anchor, 3, dtype=torch.float)
        scale = safe_inverse_sigmoid(scale)

        rots = torch.zeros(num_anchor, 4, dtype=torch.float)
        rots[:, 0] = 1

        if include_opa:
            opacity = safe_inverse_sigmoid(0.1 * torch.ones((num_anchor, 1), dtype=torch.float))
        else:
            opacity = torch.ones((num_anchor, 0), dtype=torch.float)

        if semantics:
            assert semantic_dim is not None
        else:
            semantic_dim = 0
        semantic = torch.randn(num_anchor, semantic_dim, dtype=torch.float)

        if offset:
            offsets = torch.randn(num_anchor, offset_dim, dtype=torch.float)
        else:
            offsets = torch.zeros(num_anchor, 0, dtype=torch.float)

        if pts_init:
            # non-xyz attributes: shared across all samples, learnable
            self.anchor_non_xyz = nn.Parameter(
                torch.cat([scale, rots, opacity, semantic, offsets], dim=-1),
                requires_grad=anchor_grad,
            )
        else:
            anchor = torch.cat([xyz, scale, rots, opacity, semantic, offsets], dim=-1)
            self.num_anchor = num_anchor
            self.anchor = nn.Parameter(
                torch.tensor(anchor, dtype=torch.float32),
                requires_grad=anchor_grad,
            )
            self.anchor_init = anchor

        self.num_anchor = num_anchor

    def init_weight(self):
        if not self.pts_init:
            self.anchor.data = self.anchor.data.new_tensor(self.anchor_init)

    def _sample_pts(self, init_pts, device):
        """Sample num_anchor points from backprojected 3D points for one sample.

        Args:
            init_pts: (N, 3) numpy array or tensor of 3D points in LIDAR frame.
            device: target device.

        Returns:
            xyz_normalized: (num_anchor, 3) tensor, normalized to [0, 1] within pc_range.
        """
        if isinstance(init_pts, np.ndarray):
            pts = torch.from_numpy(init_pts).float()
        else:
            pts = init_pts.float()

        pc_min = torch.tensor(self.pc_range[:3], dtype=torch.float)
        pc_max = torch.tensor(self.pc_range[3:], dtype=torch.float)

        # filter points within pc_range
        mask = ((pts[:, 0] >= pc_min[0]) & (pts[:, 0] < pc_max[0]) &
                (pts[:, 1] >= pc_min[1]) & (pts[:, 1] < pc_max[1]) &
                (pts[:, 2] >= pc_min[2]) & (pts[:, 2] < pc_max[2]))
        pts = pts[mask]

        N = pts.shape[0]
        if N == 0:
            # fallback: uniform random in pc_range
            pts = torch.rand(self.num_anchor, 3) * (pc_max - pc_min) + pc_min
        elif N < self.num_anchor:
            # repeat + jitter to fill
            repeats = (self.num_anchor // N) + 1
            pts_rep = pts.repeat(repeats, 1)[:self.num_anchor]
            # add small jitter (0.2m std)
            jitter = torch.randn_like(pts_rep) * 0.2
            pts_rep = pts_rep + jitter
            pts_rep[:, 0].clamp_(pc_min[0], pc_max[0])
            pts_rep[:, 1].clamp_(pc_min[1], pc_max[1])
            pts_rep[:, 2].clamp_(pc_min[2], pc_max[2])
            pts = pts_rep
        else:
            # random subsample
            indices = torch.randperm(N)[:self.num_anchor]
            pts = pts[indices]

        # normalize to [0, 1]
        xyz_norm = (pts - pc_min) / (pc_max - pc_min)
        xyz_norm.clamp_(1e-4, 1 - 1e-4)

        return safe_inverse_sigmoid(xyz_norm).to(device)

    def forward(self, ms_img_feats, metas=None, **kwargs):
        batch_size = ms_img_feats[0].shape[0]

        if self.pts_init and metas is not None and 'init_pts' in metas:
            # dynamic xyz from backprojected pseudo-depth points
            # init_pts is list of tuples: [(arr0,), (arr1,), ...] after collation
            init_pts_list = metas['init_pts']
            device = self.anchor_non_xyz.device
            xyz_batch = []
            for b in range(batch_size):
                pts = init_pts_list[b]
                if isinstance(pts, (tuple, list)):
                    pts = pts[0]  # unwrap tuple from collation
                xyz_b = self._sample_pts(pts, device)  # (num_anchor, 3)
                xyz_batch.append(xyz_b)
            xyz_batch = torch.stack(xyz_batch)  # (B, num_anchor, 3)

            # tile non-xyz attributes
            non_xyz = self.anchor_non_xyz[None].expand(batch_size, -1, -1)  # (B, num_anchor, D)
            anchor = torch.cat([xyz_batch, non_xyz], dim=-1)  # (B, num_anchor, 3+D)
        elif self.pts_init:
            # pts_init mode but no init_pts in metas (e.g., eval without depth)
            # fallback to random xyz
            device = self.anchor_non_xyz.device
            pc_min = torch.tensor(self.pc_range[:3], device=device)
            pc_max = torch.tensor(self.pc_range[3:], device=device)
            xyz_rand = torch.rand(batch_size, self.num_anchor, 3, device=device)
            xyz_rand = xyz_rand * (pc_max - pc_min) + pc_min
            xyz_norm = (xyz_rand - pc_min) / (pc_max - pc_min)
            xyz_norm.clamp_(1e-4, 1 - 1e-4)
            xyz_batch = safe_inverse_sigmoid(xyz_norm)
            non_xyz = self.anchor_non_xyz[None].expand(batch_size, -1, -1)
            anchor = torch.cat([xyz_batch, non_xyz], dim=-1)
        else:
            # original: fixed learnable anchor
            anchor = torch.tile(self.anchor[None], (batch_size, 1, 1))

        return {
            'representation': anchor,
        }
