from typing import List, Optional
import torch, torch.nn as nn
from torch.utils.checkpoint import checkpoint as cp

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
        decouple_feat: bool = False,
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(init_cfg)
        self.num_decoder = num_decoder
        self.num_single_frame_decoder = num_single_frame_decoder
        self.with_cp = with_cp
        # GaussianFormer V1 full decoupling: when True, skip re-injecting
        # anchor geometry into the instance_feature (query) inside the refine
        # loop, so the query stays independent of anchor geometry across all
        # decoder layers. anchor_embed is still recomputed for the next
        # layer's deformable op. Default False preserves original behavior.
        self.decouple_feat = decouple_feat

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
        representation,#representation是高斯编码器的输出参数，表示每个锚点的状态。通过训练，模型会学习如何调整这个representation参数来更好地拟合输入数据，从而实现高斯编码器的功能。
        rep_features=None,#rep_features是lifter提供的独立可学习instance feature（query）。若为None则退回从anchor几何编码派生（向后兼容）。
        ms_img_feats=None,#ms_img_feats是多尺度图像特征的列表，每个元素对应一个尺度的特征图。通过这些特征图，模型可以提取不同尺度的信息来辅助高斯编码器的学习和预测。
        metas=None,#metas是一个包含输入数据相关信息的字典，可能包括图像的元数据、传感器信息、时间戳等。这些信息可以帮助模型更好地理解输入数据的上下文，从而提高高斯编码器的性能。
        **kwargs
    ):
        feature_maps = ms_img_feats#ms_img_feats是多尺度图像特征的列表，每个元素对应一个尺度的特征图。通过这些特征图，模型可以提取不同尺度的信息来辅助高斯编码器的学习和预测。
        if isinstance(feature_maps, torch.Tensor):
            feature_maps = [feature_maps]
        anchor = representation

        anchor_embed = self.anchor_encoder(anchor)#anchor_embed是通过anchor_encoder模块对anchor进行编码得到的特征表示。这个特征表示可以包含关于锚点的位置信息、尺度信息、旋转信息等，通过训练，模型会学习如何调整这个anchor_embed参数来更好地拟合输入数据，从而实现高斯编码器的功能。
        if rep_features is not None:
            # GaussianFormer V1: use independent learnable instance feature as
            # the initial query, decoupled from anchor geometry.
            instance_feature = rep_features
        else:
            instance_feature = anchor_embed#instance_feature是通过anchor_embed得到的特征表示，表示每个锚点的状态。通过训练，模型会学习如何调整这个instance_feature参数来更好地拟合输入数据，从而实现高斯编码器的功能。这个参数在后续的计算中会被不断更新和优化，以提高模型的性能和准确性。

        prediction = []
        for i, op in enumerate(self.operation_order):#operation_order是一个字符串列表，表示在高斯编码器中不同操作的执行顺序。每个字符串对应一个特定的操作，例如"norm"表示归一化操作，"ffn"表示前馈神经网络操作，"deformable"表示可变形卷积操作，"refine"表示细化操作等。通过定义这个operation_order，模型可以按照指定的顺序执行这些操作，从而实现高斯编码器的功能。
            if op == 'spconv':
                if self.with_cp and self.training:
                    def _spconv_forward(_feat, _anc, _layer=self.layers[i]):
                        return _layer(_feat, _anc)
                    instance_feature = cp(_spconv_forward, instance_feature, anchor, use_reentrant=False)
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
            elif op == "deformable":#deformable是一个字符串，表示在高斯编码器中使用可变形卷积操作。可变形卷积是一种增强卷积神经网络的能力的方法，它允许卷积核在输入特征图上进行灵活的采样，从而更好地捕捉输入数据的几何变形和局部结构信息。在高斯编码器中，使用deformable操作可以提高模型对输入数据的适应性和鲁棒性，从而实现更准确的特征提取和表示。
                if self.with_cp and self.training:
                    def _deform_forward(_feat, _anc, _anc_emb, _fmaps, _metas, _enc=self.anchor_encoder, _layer=self.layers[i]):
                        return _layer(_feat, _anc, _anc_emb, _fmaps, _metas, anchor_encoder=_enc)
                    instance_feature = cp(_deform_forward, instance_feature, anchor, anchor_embed, feature_maps, metas, use_reentrant=False)
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
                    if not self.decouple_feat:
                        instance_feature += anchor_embed
            else:
                raise NotImplementedError(f"{op} is not supported.")

        # TODO: 这里只返回并监督最后一层的gauss即可，instance feature同理；需要修改一下head部分随机监督一层高斯的代码

        return {"representation": prediction[-1], "instance_feature": instance_feature,
        "anchors": anchor}
