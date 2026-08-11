"""VADHeadFutGaussianTime —— VADHeadFutGaussian 的「时间位置编码」变体。

动机
----
VADHeadFutGaussian (planner_v3) 把未来所有帧高斯展平成一个 key 集合
    (fut_ts*G, B, D)
供 ego 单 query 做交叉注意力。由于未来帧被「展平」到 seq 维，且 key/value 侧
没有携带任何「这是第几帧」的信息，模型无法显式区分未来不同时刻的高斯 ——
跨帧的时序信息只能隐含在 offset 平移后的坐标里。

本模块 (VADHeadFutGaussianTime) 在 v3 基础上做最小改动：
    给未来帧高斯特征在 (fut_ts, G, D) 维度上，逐时间步加上一个「可学习的时间
    位置编码」fut_time_pos = nn.Embedding(fut_ts, embed_dims)，
再展平进注意力 key/value。这样每个 key 都明确携带「属于未来第几帧」的信息，
帮助 ego 在跨帧注意力中区分不同时刻的高斯。

与 v3 的差异（仅 fn：_build_future_gaussian_kv）
    v3:  fut_gs = fut_gaussian_fus_mlp(fut_feats)          # (fut_ts, G, D)
    v4:  fut_gs = fut_gaussian_fus_mlp(fut_feats)
         fut_gs = fut_gs + fut_time_pos.weight[:, None, :]  # ← 逐时间步 PE
其余（当前帧 3 路、第 4 路 decoder、4D 输出头）与 v3 完全一致。

新增参数
    fut_time_pos : nn.Embedding(fut_ts, embed_dims)，xavier 随机初始化。
"""
import torch
import torch.nn as nn
from mmengine.registry import MODELS

from .planner_v3 import VADHeadFutGaussian


@MODELS.register_module()
class VADHeadFutGaussianTime(VADHeadFutGaussian):
    """VADHeadFutGaussian 变体：未来帧高斯 key/value 加入逐时间步位置编码。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _init_layers(self):
        """父类初始化 + 未来帧高斯逐时间步位置编码。"""
        super()._init_layers()

        # 逐未来时间步的可学习时间位置编码
        self.fut_time_pos = nn.Embedding(self.fut_ts, self.embed_dims)

    def init_weights(self):
        """父类初始化 + 时间位置编码 xavier 初始化。"""
        super().init_weights()
        if hasattr(self, 'fut_time_pos'):
            nn.init.xavier_uniform_(self.fut_time_pos.weight)

    def _build_future_gaussian_kv(self, results):
        """由当前高斯 + offset 构造未来所有帧高斯并编码，加入逐时间步位置编码。

        返回 (fut_ts*G, B=1, D)，batch_first=False 约定，供 ego 单 query 交叉注意力。
        """
        gs_out = results['gaussian_output']          # (B=1, G, 28)
        base = gs_out[0]                             # (G, 28)
        G = base.shape[0]

        offset = results.get('offset', None)
        if offset is not None:
            # (1, G, fut_ts, 2) -> (fut_ts, G, 2)
            off = offset.reshape(1, -1, self.fut_ts, 2)[0].permute(1, 0, 2)
            off = torch.nan_to_num(off, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            off = base.new_zeros((self.fut_ts, G, 2))

        # 未来第 t 帧高斯：xy 平移 offset_t，其余 26 维不变
        means_xy = base[:, :2].unsqueeze(0) + off                    # (fut_ts, G, 2)
        rest = base[:, 2:].unsqueeze(0).expand(self.fut_ts, -1, -1)  # (fut_ts, G, 26)
        fut_feats = torch.cat([means_xy, rest], dim=-1)             # (fut_ts, G, 28)

        fut_gs = self.fut_gaussian_fus_mlp(fut_feats)              # (fut_ts, G, D)

        # ===== v4 delta：逐时间步位置编码 =====
        # fut_time_pos.weight: (fut_ts, D) -> (fut_ts, 1, D)，广播加到每个 future 帧
        fut_gs = fut_gs + self.fut_time_pos.weight[:, None, :]

        # 展平帧与高斯为一个 key 集合：(fut_ts*G, B=1, D)
        fut_gs = fut_gs.reshape(self.fut_ts * G, self.embed_dims).unsqueeze(1)
        return fut_gs
