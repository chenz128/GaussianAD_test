import torch, torch.nn as nn
from mmengine import MODELS
from mmengine.model import BaseModule

import spconv.pytorch as spconv
from .utils import cartesian
from functools import partial


@MODELS.register_module()
class SparseConv3D(BaseModule):
    def __init__(
        self, 
        in_channels,
        embed_channels,
        pc_range,
        grid_size,
        use_out_proj=False,
        kernel_size=5,
        dilation=1,
        init_cfg=None
    ):
        super().__init__(init_cfg)

        self.layer = spconv.SubMConv3d(
            in_channels,
            embed_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            dilation=dilation)
        if use_out_proj:
            self.output_proj = nn.Linear(embed_channels, embed_channels)
        else:
            self.output_proj = nn.Identity()
        self.get_xyz = partial(cartesian, pc_range=pc_range)
        self.register_buffer('pc_range', torch.tensor(pc_range, dtype=torch.float), False)
        self.register_buffer('grid_size', torch.tensor(grid_size, dtype=torch.float), False)

    def forward(self, instance_feature, anchor):
        # anchor: b, g, 11
        # instance_feature: b, g, c
        bs, g, _ = instance_feature.shape

        # sparsify
        anchor_xyz = self.get_xyz(anchor).flatten(0, 1) 

        indices = anchor_xyz - self.pc_range[None, :3]
        indices = indices / self.grid_size[None, :] # bg, 3
        indices = indices.to(torch.int32)
        batched_indices = torch.cat([
            torch.arange(bs, device=indices.device, dtype=torch.int32).reshape(
                bs, 1, 1).expand(-1, g, -1).flatten(0, 1),
            indices], dim=-1)
        
        spatial_shape = indices.max(0)[0]

        input = spconv.SparseConvTensor(
            instance_feature.flatten(0, 1), # bg, c
            indices=batched_indices, # bg, 4
            spatial_shape=spatial_shape,
            batch_size=bs)

        output = self.layer(input)
        output = output.features.unflatten(0, (bs, g))

        return self.output_proj(output)


@MODELS.register_module()
class SparseConv3DBlock(BaseModule):
    def __init__(
        self, 
        in_channels,
        embed_channels,
        pc_range,
        grid_size,
        use_out_proj=False,
        kernel_size=[5],
        stride=[1],
        padding=[0],
        dilation=[1],
        spatial_shape=[256, 256, 20],
        init_cfg=None
    ):
        super().__init__(init_cfg)

        assert isinstance(kernel_size, (list, tuple))
        assert isinstance(padding, (list, tuple))
        assert len(kernel_size) == len(padding)
        # [Opt-spconv-1] 同一个 SparseConv3DBlock 内的多个 SubMConv3d 共享 indice_key。
        # SubMConv 不下采样、indices 不变，多次调用的 hash table（rulebook）可以复用，
        # 避免 forward 重复构建、backward 重复申请显存。数值结果完全一致，不影响精度。
        # 每个 block 实例使用唯一 key（基于对象 id），避免与其他 block 冲突。
        _block_indice_key = f'subm3d_block_{id(self):x}'
        # [Opt-spconv-2] large_kernel_fast_algo=True：把 algo 自动选择的 kv 上限从
        # 32 提到 128。我们 kernel_size=5 → kv=125，原本会 fallback 到 ConvAlgo.Native
        # (最慢)；现在升到 ConvAlgo.MaskImplicitGemm（融合算子，3D 大 kernel 下快很多）。
        # 数学等价（同一个卷积运算的不同实现），仅浮点累加顺序略异。
        layers = []
        for k, s, p, d in zip(kernel_size, stride, padding, dilation):
            layers.append(spconv.SubMConv3d(
                in_channels,
                embed_channels,
                kernel_size=k,
                stride=s,
                padding=p,
                dilation=d,
                indice_key=_block_indice_key,
                large_kernel_fast_algo=True))
            layers.append(nn.LayerNorm(embed_channels))
            layers.append(nn.ReLU(True))
            in_channels = embed_channels
        self.layers = nn.ModuleList(layers)
        if use_out_proj:
            self.output_proj = nn.Linear(embed_channels, embed_channels)
        else:
            self.output_proj = nn.Identity()
        self.get_xyz = partial(cartesian, pc_range=pc_range)
        self.spatial_shape = spatial_shape
        self.register_buffer('pc_range', torch.tensor(pc_range, dtype=torch.float), False)
        self.register_buffer('grid_size', torch.tensor(grid_size, dtype=torch.float), False)

    def forward(self, instance_feature, anchor):
        # anchor: b, g, 11
        # instance_feature: b, g, c
        bs, g, _ = instance_feature.shape

        # sparsify
        anchor_xyz = self.get_xyz(anchor).flatten(0, 1)

        indices = anchor_xyz - self.pc_range[None, :3]
        indices = indices / self.grid_size[None, :] # bg, 3
        indices = indices.to(torch.int32)
        batched_indices = torch.cat([
            torch.arange(bs, device=indices.device, dtype=torch.int32).reshape(
                bs, 1, 1).expand(-1, g, -1).flatten(0, 1),
            indices], dim=-1)
        x = spconv.SparseConvTensor(
            instance_feature.flatten(0, 1), # bg, c
            indices=batched_indices, # bg, 4
            spatial_shape=self.spatial_shape,
            batch_size=bs)

        for layer in self.layers:
            if isinstance(layer, spconv.SubMConv3d):
                x = layer(x)
            elif isinstance(layer, (nn.LayerNorm, nn.ReLU)):
                x = x.replace_feature(layer(x.features))
            else:
                raise NotImplementedError

        output = x.features.unflatten(0, (bs, g)) # b, g, c

        return self.output_proj(output)


@MODELS.register_module()
class SparseConv4D(BaseModule):
    def __init__(
        self, 
        in_channels,
        embed_channels,
        pc_range,
        grid_size,
        use_out_proj=False,
        kernel_size=[5],
        stride=[1],
        padding=[0],
        dilation=[1],
        spatial_shape=[3, 256, 256, 20],
        init_cfg=None
    ):
        super().__init__(init_cfg)

        assert isinstance(kernel_size, (list, tuple))
        assert isinstance(padding, (list, tuple))
        assert len(kernel_size) == len(padding)
        layers = []
        for k, s, p, d in zip(kernel_size, stride, padding, dilation):
            layers.append(spconv.SubMConv4d(
                in_channels,
                embed_channels,
                kernel_size=k,
                stride=s,
                padding=p,
                dilation=d,
                bias=False,
                indice_key='sub0'))
            layers.append(nn.LayerNorm(embed_channels))
            layers.append(nn.ReLU(True))
            in_channels = embed_channels
        self.layers = nn.ModuleList(layers)
        if use_out_proj:
            self.output_proj = nn.Linear(embed_channels, embed_channels)
        else:
            self.output_proj = nn.Identity()

        self.get_xyz = partial(cartesian, pc_range=pc_range)
        self.spatial_shape = spatial_shape
        self.register_buffer('pc_range', torch.tensor(pc_range, dtype=torch.float), False)
        self.register_buffer('grid_size', torch.tensor(grid_size, dtype=torch.float), False)

    def forward(self, instance_feature, anchors, batch_indices):
        # batch_indices: n, 2
        # instance_feature: n, c
        # anchors: n, _

        # sparsify
        anchor_xyz = self.get_xyz(anchors)
        xyz_indices = ((anchor_xyz - self.pc_range[:3]) / self.grid_size).to(torch.int32) # n, 3

        indices = torch.cat([batch_indices, xyz_indices], dim=-1) # bfg, 5

        x = spconv.SparseConvTensor(
            instance_feature, # n, c
            indices=indices, # n, 5
            spatial_shape=self.spatial_shape,
            batch_size=1)

        for layer in self.layers:
            if isinstance(layer, spconv.SubMConv4d):
                x = layer(x)
            elif isinstance(layer, (nn.LayerNorm, nn.ReLU)):
                x = x.replace_feature(layer(x.features))
            else:
                raise NotImplementedError

        output = self.output_proj(x.features) # n, c
        return output