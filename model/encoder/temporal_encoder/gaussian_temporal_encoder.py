import torch, numpy as np, torch.nn as nn
import spconv.pytorch as spconv
from mmengine import MODELS
from mmengine.model import BaseModule
from functools import partial

from ...encoder.gaussian_encoder.utils import cartesian
from model.utils.safe_ops import safe_inverse_sigmoid, LOGIT_MAX


@MODELS.register_module()
class GaussianTemporalEncoder(BaseModule):

    def __init__(
        self,
        pc_range,
        num_anchor,
        anchor_encoder,
        num_encoder,
        norm_layer: dict,
        ffn: dict,
        refine_layer: dict,
        spconv_layer: dict = None,
        operation_order = None,
        init_cfg = None,
        num_frames = 4,
    ):
        super().__init__(init_cfg)
        self.get_xyz = partial(cartesian, pc_range=pc_range)
        self.num_anchor = num_anchor
        self.spatial_shape = spconv_layer['spatial_shape']
        self.register_buffer('pc_range', torch.tensor(pc_range, dtype=torch.float), False)
        self.register_buffer('grid_size', torch.tensor(spconv_layer['grid_size'], dtype=torch.float), False)

        self.num_encoder = num_encoder
        self.operation_order = operation_order
        self.num_frames = num_frames

        # =========== build modules ===========
        def build(cfg):
            if cfg is None:
                return None
            return MODELS.build(cfg)

        self.anchor_encoder = build(anchor_encoder)

        self.op_config_map = {
            "norm": norm_layer,
            "ffn": ffn,
            "refine": refine_layer,
            "spconv": spconv_layer,
        }
        self.layers = nn.ModuleList(
            [
                build(self.op_config_map.get(op, None))
                for op in self.operation_order
            ]
        )

    def init_weights(self):
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op != "refine":
                for p in self.layers[i].parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    def warp_anchor(self, anchors, instance_feature, metas):
        # TODO: metas[0]['lidar2global'] 适配数据集输入顺序
        # anchors: bs, num_frames, g, c

        # warp anchors from previous to current frame
        lidar2global = torch.tensor(metas['lidar2global'][0], dtype=anchors.dtype, device=anchors.device)
        B, F, N, _ = anchors.shape
        xyz = self.get_xyz(anchors) # bs, f, n, 3
        prev2cur = torch.matmul(torch.linalg.inv(lidar2global[0]), lidar2global[:F])[None, :, None] # bs, f, 1, 4, 4
        new_xyz = torch.matmul(prev2cur, torch.cat([xyz, torch.ones_like(xyz[..., :1])], dim=-1)[..., None])[..., :3, 0]     # bs, f, g, 3

        # get anchor indices
        # indices = ((new_xyz - self.pc_range[:3]) / self.grid_size).to(torch.int32)
        temporal_indices = torch.arange(F, device=new_xyz.device, dtype=torch.int32).reshape(1, F, 1, 1).expand(B, -1, N, -1)
        batch_indices = torch.arange(B, device=new_xyz.device, dtype=torch.int32).reshape(B, 1, 1, 1).expand(-1, F, N, -1)
        anchors_indices = torch.cat([batch_indices, temporal_indices], dim=-1)     # bs, f, g, 2

        # filter out anchors beyond the boundary
        new_xyz = (new_xyz - self.pc_range[:3]) / (self.pc_range[3:] - self.pc_range[:3])
        valid_mask = ((new_xyz > 1-LOGIT_MAX) & (new_xyz < LOGIT_MAX)).all(-1)
        anchors_warp = torch.cat([safe_inverse_sigmoid(new_xyz), anchors[..., 3:]], dim=-1)     # bs, f, g, 28
        anchors_warp = anchors_warp[valid_mask]
        instance_feature_warp = instance_feature[valid_mask]
        anchors_indices = anchors_indices[valid_mask]

        return anchors_warp, instance_feature_warp, anchors_indices

    def forward(self, anchors, instance_feature, metas, **kwargs):
        # TODO: 输入anchors shape需要对齐为 [B, F, N, C], instance_feature同理
        anchors = anchors.reshape(-1, self.num_frames, *anchors.shape[1:])
        instance_feature = instance_feature.reshape(-1, self.num_frames, *instance_feature.shape[1:])
        ### warp anchors from previous to current frame
        anchors, instance_feature, batch_indices = self.warp_anchor(anchors, instance_feature, metas)

        # ### refine
        # instance_feature = self.instance_encoder(anchors)
        anchor_embed = self.anchor_encoder(anchors)

        prediction = []
        for i, op in enumerate(self.operation_order):
            if op == 'spconv':
                instance_feature = self.layers[i](
                    instance_feature,
                    anchors,
                    batch_indices)
            elif op == "norm" or op == "ffn":
                instance_feature = self.layers[i](instance_feature)
            elif op == "identity":
                identity = instance_feature
            elif op == "add":
                instance_feature = instance_feature + identity
            elif "refine" in op:
                if i == len(self.operation_order) - 1:
                    mask = (batch_indices[:, 1] == batch_indices[:, 1].max())
                else:
                    mask = None
                anchors, gaussian, offset = self.layers[i](
                    instance_feature,
                    anchors,
                    anchor_embed,
                    mask
                )

                prediction.append({'gaussian': gaussian})
                if i != len(self.operation_order) - 1:
                    anchor_embed = self.anchor_encoder(anchors)
            else:
                raise NotImplementedError(f"{op} is not supported.")

        return {"representation_temp": prediction[-1], "rep_features": instance_feature,
                "offset": offset}
