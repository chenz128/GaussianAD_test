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
        layers = []
        for k, s, p, d in zip(kernel_size, stride, padding, dilation):
            layers.append(spconv.SubMConv3d(
                in_channels,
                embed_channels,
                kernel_size=k,
                stride=s,
                padding=p,
                dilation=d))
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