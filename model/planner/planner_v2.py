"""VADHeadFutAttn —— 让 ego 分别与「当前帧高斯」「未来帧高斯」做注意力的 planner。

背景
----
本项目里未来帧的高斯不是重新预测的，而是把当前帧高斯按模型预测的 flow 位移
(``offset``) 平移得到：
    未来第 t 帧高斯.xy = 当前高斯.xy + offset[..., t, :]      (其余属性不变)
(与 model/head/gaussian_head.py::forward_flow 完全一致：means_fut = means + offset)

原始 VADHead 只让 ego_query 与「当前帧」的 agent / map / gaussian 交互，再用
MLP 一次性回归未来 6 帧轨迹，未来帧高斯的运动信息没有直接参与 planning。

本模块（VADHeadFutAttn）在保留当前帧交互的基础上，新增：
    ego（逐未来时间步）  <->  未来帧高斯（当前高斯平移 offset 后）  交叉注意力
即 ego 分别与当前帧高斯、未来帧高斯做注意力，让 planner 显式看到"高斯会往哪动"。

数据流
------
- results['gaussian_output']: (B, G, 28) 当前帧高斯 [means3, scales3, rot4, opa1, sem17]
- results['offset']:          (B, G, fut_ts*2) flow 位移，reshape 为 (1, G, fut_ts, 2)
  （由 temporal_encoder 产出，planner_head 运行前已在 results 中）

前向流程
--------
1) 当前帧（复用 VADHead 及其预训练权重，不改动）：
   ego_query -> ego↔agent -> ego↔map -> ego↔gaussian(当前) -> ego_feats [B,1,3D]
2) 未来帧高斯注意力（新增）：
   a. 由 offset 平移当前高斯 -> 每个未来时间步一套高斯特征 (fut_ts, G, 28)
      -> fut_gaussian_fus_mlp 编码 -> (fut_ts, G, D)
   b. 未来 ego token：由当前帧 ego_feats 投影初始化(ego_to_fut) + 逐帧位置编码(fut_pos)
      -> 先做时间维 self-attention(fut_self_decoder, 可选) 建模时序连贯性
   c. 逐未来帧 ego token 与「该帧未来高斯」做交叉注意力(ego_fut_gaussian_decoder)
3) 逐时间步回归 -> (B, ego_fut_mode, fut_ts, 2)，与 PlanLoss / use_plan_ego 兼容。

注：offset.reshape(1, -1, fut_ts, 2) 沿用 forward_flow 的 batch=1 约定。
"""
import torch
import torch.nn as nn
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmengine.registry import MODELS

from .planner import VADHead


@MODELS.register_module()
class VADHeadFutAttn(VADHead):
    """VADHead 变体：ego 分别与当前帧 / 未来帧高斯做注意力。

    额外参数：
        fut_self_decoder (dict, optional): 未来 ego token 的时间维 self-attention。
        ego_fut_gaussian_decoder (dict): ego(逐未来帧) 与未来帧高斯的交叉注意力。
    """

    def __init__(self,
                 *args,
                 fut_self_decoder=None,
                 ego_fut_gaussian_decoder=None,
                 **kwargs):
        self.fut_self_decoder = fut_self_decoder
        self.ego_fut_gaussian_decoder = ego_fut_gaussian_decoder
        super().__init__(*args, **kwargs)

    def _init_layers(self):
        """父类初始化 + 未来帧高斯注意力相关层。"""
        super()._init_layers()

        # 未来帧高斯特征编码（与当前帧 gaussian_fus_mlp 结构一致，独立权重）
        self.fut_gaussian_fus_mlp = nn.Sequential(
            nn.Linear(28, self.embed_dims, bias=True),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims, bias=True))

        # 当前帧 ego 融合特征 (3D) -> D：作为未来 ego token 的初始化，
        # 让当前帧 ego 交叉注意力分支（含预训练权重）参与未来轨迹预测。
        self.ego_to_fut = nn.Linear(self.embed_dims * 3, self.embed_dims, bias=True)

        # 逐未来时间步的位置编码 token
        self.fut_pos = nn.Embedding(self.fut_ts, self.embed_dims)

        # 未来 ego token 的时间维 self-attention（可选）
        if self.fut_self_decoder is not None:
            self.fut_self_decoder = build_transformer_layer_sequence(
                self.fut_self_decoder)

        # ego(逐未来帧) 与未来帧高斯的交叉注意力（核心新增）
        if self.ego_fut_gaussian_decoder is not None:
            self.ego_fut_gaussian_decoder = build_transformer_layer_sequence(
                self.ego_fut_gaussian_decoder)

        # 逐时间步回归：D -> D -> ego_fut_mode*2
        self.fut_out_mlp = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims, bias=True),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.ego_fut_mode * 2, bias=True))

    def init_weights(self):
        """父类初始化 + 新增 transformer 层 xavier 初始化。"""
        super().init_weights()
        for dec in [
                getattr(self, 'fut_self_decoder', None),
                getattr(self, 'ego_fut_gaussian_decoder', None),
        ]:
            if dec is not None:
                for p in dec.parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)

    def _build_future_gaussian_query(self, results):
        """由当前高斯 + offset 构造逐未来帧高斯特征并编码。

        返回 (G, fut_ts, D)：seq=G, batch=fut_ts（batch_first=False 的 key/value）。
        """
        gs_out = results['gaussian_output']          # (B, G, 28), B=1
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
        means_xy = base[:, :2].unsqueeze(0) + off                       # (fut_ts, G, 2)
        rest = base[:, 2:].unsqueeze(0).expand(self.fut_ts, -1, -1)     # (fut_ts, G, 26)
        fut_feats = torch.cat([means_xy, rest], dim=-1)                 # (fut_ts, G, 28)

        fut_gs_query = self.fut_gaussian_fus_mlp(fut_feats)            # (fut_ts, G, D)
        # -> key/value: (seq=G, batch=fut_ts, D)
        return fut_gs_query.permute(1, 0, 2)

    def forward(self, results):
        agent_query, agent_mask = self.prepare_agent_query(results)
        map_query = self.prepare_map_query(results)
        gaussian_query = self.gaussian_fus_mlp(results['gaussian_output'])

        batch = agent_query.shape[0]
        ego_query = self.ego_query.weight.unsqueeze(0).repeat(batch, 1, 1)

        # ---------- 当前帧：ego <-> agent/map/gaussian（与 VADHead 完全一致）----------
        ego_agent_query = self.ego_agent_decoder(
            query=ego_query.permute(1, 0, 2),
            key=agent_query.permute(1, 0, 2),
            value=agent_query.permute(1, 0, 2),
            key_padding_mask=agent_mask)

        ego_map_query = self.ego_map_decoder(
            query=ego_agent_query,
            key=map_query.permute(1, 0, 2),
            value=map_query.permute(1, 0, 2))

        ego_gs_query = self.ego_gaussian_decoder(
            query=ego_map_query,
            key=gaussian_query.permute(1, 0, 2),
            value=gaussian_query.permute(1, 0, 2))

        ego_feats = torch.cat(
            [ego_agent_query.permute(1, 0, 2),
             ego_map_query.permute(1, 0, 2),
             ego_gs_query.permute(1, 0, 2)],
            dim=-1)  # [B, 1, 3D]

        # ---------- 未来帧：ego(逐帧) <-> 未来帧高斯（当前高斯平移 offset）----------
        # 未来帧高斯 key/value: (G, fut_ts, D)
        fut_gs_kv = self._build_future_gaussian_query(results)

        # 未来 ego token：当前 ego 特征投影初始化 + 逐帧位置编码 -> (fut_ts, 1, D)
        fut_ego = self.ego_to_fut(ego_feats)              # (B=1, 1, D)
        fut_ego = fut_ego.reshape(1, self.embed_dims)     # (1, D)
        fut_ego = fut_ego.unsqueeze(0).expand(self.fut_ts, -1, -1)  # (fut_ts, 1, D)
        fut_ego = fut_ego + self.fut_pos.weight.unsqueeze(1)        # (fut_ts, 1, D)

        # 1) 时间维 self-attention（frames 在 seq 维，建模时序连贯性）
        if self.fut_self_decoder is not None:
            fut_ego = self.fut_self_decoder(
                query=fut_ego, key=fut_ego, value=fut_ego)

        # 2) 逐未来帧 ego token 与该帧未来高斯交叉注意力
        #    frames 折叠进 batch 维并行：query (1, fut_ts, D), key/value (G, fut_ts, D)
        fut_ego_cross = fut_ego.permute(1, 0, 2)          # (1, fut_ts, D)
        if self.ego_fut_gaussian_decoder is not None:
            fut_ego_cross = self.ego_fut_gaussian_decoder(
                query=fut_ego_cross,
                key=fut_gs_kv,
                value=fut_gs_kv)                          # (1, fut_ts, D)

        # 3) 逐时间步回归 -> (B, ego_fut_mode, fut_ts, 2)
        fut_feats = fut_ego_cross.reshape(self.fut_ts, self.embed_dims)  # (fut_ts, D)
        fut_feats = fut_feats.unsqueeze(0)                # (B=1, fut_ts, D)
        outputs_fut = self.fut_out_mlp(fut_feats)         # (1, fut_ts, ego_fut_mode*2)
        outputs_ego_trajs = outputs_fut.reshape(
            batch, self.fut_ts, self.ego_fut_mode, 2)
        outputs_ego_trajs = outputs_ego_trajs.permute(0, 2, 1, 3)

        outs = {'ego_fut_preds': outputs_ego_trajs}

        return outs
