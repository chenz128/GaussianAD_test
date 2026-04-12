import torch, torch.nn as nn
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
    ):
        super().__init__()
        self.embed_dims = embed_dims

        xyz = torch.rand(num_anchor, 3, dtype=torch.float)
        xyz = safe_inverse_sigmoid(xyz)

        scale = torch.rand_like(xyz)
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

        anchor = torch.cat([xyz, scale, rots, opacity, semantic, offsets], dim=-1)

        self.num_anchor = num_anchor
        self.anchor = nn.Parameter(
            torch.tensor(anchor, dtype=torch.float32),
            requires_grad=anchor_grad,
        )
        self.anchor_init = anchor

    def init_weight(self):
        self.anchor.data = self.anchor.data.new_tensor(self.anchor_init)

    def forward(self, ms_img_feats, **kwargs):
        batch_size = ms_img_feats[0].shape[0]
        anchor = torch.tile(self.anchor[None], (batch_size, 1, 1))

        return {
            'representation': anchor,
        }
