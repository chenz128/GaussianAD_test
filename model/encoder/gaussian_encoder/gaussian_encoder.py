from typing import List, Optional
import torch, torch.nn as nn
from torch.utils.checkpoint import checkpoint as gradient_checkpoint

from mmseg.registry import MODELS
from mmengine import build_from_cfg
from ..base_encoder import BaseEncoder


@MODELS.register_module()
class GaussianOccEncoder(BaseEncoder):
    def __init__(
        self,
        anchor_encoder: dict,
        norm_layer: dict,
        ffn: dict,
        deformable_model: dict,
        refine_layer: dict,
        mid_refine_layer: dict = None,
        spconv_layer: dict = None,
        num_decoder: int = 6,
        num_single_frame_decoder: int = -1,
        operation_order: Optional[List[str]] = None,
        with_cp: bool = False,
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(init_cfg)
        self.num_decoder = num_decoder
        self.num_single_frame_decoder = num_single_frame_decoder
        self.with_cp = with_cp

        if operation_order is None:
            operation_order = [
                "spconv",
                "norm",
                "deformable",
                "norm",
                "ffn",
                "norm",
                "refine",
            ] * num_decoder
        self.operation_order = operation_order

        # =========== build modules ===========
        def build(cfg, registry):
            if cfg is None:
                return None
            return build_from_cfg(cfg, registry)

        self.anchor_encoder = build(anchor_encoder, MODELS)
        self.op_config_map = {
            "norm": [norm_layer, MODELS],
            "ffn": [ffn, MODELS],
            "deformable": [deformable_model, MODELS],
            "refine": [refine_layer, MODELS],
            "mid_refine":[mid_refine_layer, MODELS],
            "spconv": [spconv_layer, MODELS],
        }
        self.layers = nn.ModuleList(
            [
                build(*self.op_config_map.get(op, [None, None]))
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

    def forward(
        self,
        representation,
        ms_img_feats=None,
        metas=None,
        **kwargs
    ):
        feature_maps = ms_img_feats
        if isinstance(feature_maps, torch.Tensor):
            feature_maps = [feature_maps]
        anchor = representation

        anchor_embed = self.anchor_encoder(anchor)
        instance_feature = anchor_embed

        prediction = []
        for i, op in enumerate(self.operation_order):
            if op == 'spconv':
                if self.with_cp and self.training:
                    def _spconv(if_, anc, _layer=self.layers[i]):
                        return _layer(if_, anc)
                    instance_feature = gradient_checkpoint(
                        _spconv, instance_feature, anchor, use_reentrant=False)
                else:
                    instance_feature = self.layers[i](
                        instance_feature,
                        anchor)
            elif op == "norm" or op == "ffn":
                instance_feature = self.layers[i](instance_feature)
            elif op == "identity":
                identity = instance_feature
            elif op == "add":
                instance_feature = instance_feature + identity
            elif op == "deformable":
                if self.with_cp and self.training:
                    def _deform(if_, anc, ae_, _layer=self.layers[i], _fm=feature_maps, _m=metas, _ae_enc=self.anchor_encoder):
                        return _layer(if_, anc, ae_, _fm, _m, anchor_encoder=_ae_enc)
                    instance_feature = gradient_checkpoint(
                        _deform, instance_feature, anchor, anchor_embed, use_reentrant=False)
                else:
                    instance_feature = self.layers[i](
                        instance_feature,
                        anchor,
                        anchor_embed,
                        feature_maps,
                        metas,
                        anchor_encoder=self.anchor_encoder,
                    )
            elif "refine" in op:
                anchor, gaussian, offset = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                )

                prediction.append({'gaussian': gaussian})
                if i != len(self.operation_order) - 1:
                    anchor_embed = self.anchor_encoder(anchor)
                    instance_feature += anchor_embed
            else:
                raise NotImplementedError(f"{op} is not supported.")

        # TODO: 这里只返回并监督最后一层的gauss即可，instance feature同理；需要修改一下head部分随机监督一层高斯的代码

        return {"representation": prediction[-1], "instance_feature": instance_feature,
        "anchors": anchor}
