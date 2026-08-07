"""VADHeadFutGaussian —— 把「未来帧高斯」作为与 agent/map/当前高斯完全对称的
第 4 路 stream 融合进 planner。

设计动机
--------
原 VADHead 让 ego_query 依次与「当前帧」agent / map / gaussian 交叉注意力，拼接
3 路特征 (3D) 后用 MLP 一次性回归整条未来轨迹。未来帧高斯的运动信息没有直接参与
planning。

上一版 VADHeadFutAttn 的做法存在两个问题：
  1) 信息瓶颈：先把 3D=384 压到 128 再生成未来 token，丢失了成熟的当前场景融合特征；
  2) 替换而非增广：最终轨迹完全来自新建的 ego_to_fut + self-attn + cross-attn +
     fut_out_mlp，没有保留原 VADHead 输出头作为 baseline，新分支要从零学整条轨迹映射。

本模块 (VADHeadFutGaussian) 只做一件事：让未来帧高斯像另外三路信息一样正常融合。
  - 保留当前帧 3 路交叉注意力（结构与权重完全不变）；
  - 新增一路：ego <-> 未来帧高斯（当前高斯按 offset 平移得到）交叉注意力；
  - 拼接 4 路特征 (4D)，复用原 ego_fut_decoder（仅把输入从 3D 加宽到 4D）。
因此未来帧信息是「增广」到原输出头，而不是替换它，也不存在压缩瓶颈。

数据流
------
- results['gaussian_output']: (B, G, 28) 当前帧高斯 [means3, scales3, rot4, opa1, sem17]
- results['offset']:          (B, G, fut_ts*2) flow 位移，reshape 为 (1, G, fut_ts, 2)
未来第 t 帧高斯.xy = 当前高斯.xy + offset[..., t, :]，其余属性不变
（与 model/head/gaussian_head.py::forward_flow 一致）。

前向流程
--------
1) 当前帧（与 VADHead 完全一致）：
   ego -> ego↔agent -> ego↔map -> ego↔gaussian(当前) 得 ego_agent/ego_map/ego_gs
2) 未来帧（新增，与上面对称）：
   ego_gs -> ego↔gaussian(未来所有帧展平成一个 key 集合) 得 ego_futgs
3) 拼接 [ego_agent, ego_map, ego_gs, ego_futgs] (4D) -> ego_fut_decoder 回归轨迹
   -> (B, ego_fut_mode, fut_ts, 2)，与 PlanLoss / use_plan_ego 兼容。
"""
import torch
import torch.nn as nn
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmengine.registry import MODELS

from .planner import VADHead


@MODELS.register_module()
class VADHeadFutGaussian(VADHead):
    """VADHead 变体：新增「ego <-> 未来帧高斯」第 4 路交叉注意力。

    额外参数：
        ego_fut_gaussian_decoder (dict): ego 与未来帧高斯的交叉注意力配置。
    """

    def __init__(self,
                 *args,
                 ego_fut_gaussian_decoder=None,
                 **kwargs):
        self.ego_fut_gaussian_decoder = ego_fut_gaussian_decoder
        super().__init__(*args, **kwargs)

    def _init_layers(self):
        """父类初始化 + 未来帧高斯这一路的编码/注意力/加宽后的输出头。"""
        super()._init_layers()

        # 未来帧高斯特征编码（结构同当前帧 gaussian_fus_mlp，独立权重）
        self.fut_gaussian_fus_mlp = nn.Sequential(
            nn.Linear(28, self.embed_dims, bias=True),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims, bias=True))

        # ego <-> 未来帧高斯 交叉注意力（核心新增，第 4 路）
        if self.ego_fut_gaussian_decoder is not None:
            self.ego_fut_gaussian_decoder = build_transformer_layer_sequence(
                self.ego_fut_gaussian_decoder)

        # 输出头输入从 3D 加宽到 4D（保留原 MLP 结构，只改输入维度）
        ego_fut_dec_in_dim = self.embed_dims * 4
        ego_fut_decoder = []
        for _ in range(2):
            ego_fut_decoder.append(nn.Linear(ego_fut_dec_in_dim, ego_fut_dec_in_dim))
            ego_fut_decoder.append(nn.ReLU())
        ego_fut_decoder.append(
            nn.Linear(ego_fut_dec_in_dim, self.ego_fut_mode * self.fut_ts * 2))
        self.ego_fut_decoder = nn.Sequential(*ego_fut_decoder)

    def init_weights(self):
        """父类初始化 + 新增交叉注意力层 xavier 初始化。"""
        super().init_weights()
        if self.ego_fut_gaussian_decoder is not None:
            for p in self.ego_fut_gaussian_decoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def _build_future_gaussian_kv(self, results):
        """由当前高斯 + offset 构造未来所有帧高斯并编码，展平成一个 key/value 集合。

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
        # 展平帧与高斯为一个 key 集合：(fut_ts*G, B=1, D)
        fut_gs = fut_gs.reshape(self.fut_ts * G, self.embed_dims).unsqueeze(1)
        return fut_gs

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

        # ---------- 未来帧：ego <-> 未来帧高斯（第 4 路，与上面对称）----------
        ego_futgs_query = ego_gs_query
        if self.ego_fut_gaussian_decoder is not None:
            fut_gs_kv = self._build_future_gaussian_kv(results)   # (fut_ts*G, B, D)
            ego_futgs_query = self.ego_fut_gaussian_decoder(
                query=ego_gs_query,
                key=fut_gs_kv,
                value=fut_gs_kv)

        ego_feats = torch.cat(
            [ego_agent_query.permute(1, 0, 2),
             ego_map_query.permute(1, 0, 2),
             ego_gs_query.permute(1, 0, 2),
             ego_futgs_query.permute(1, 0, 2)],
            dim=-1
        )  # [B, 1, 4D]

        # Ego prediction（复用原输出头，输入从 3D 加宽到 4D）
        outputs_ego_trajs = self.ego_fut_decoder(ego_feats)
        outputs_ego_trajs = outputs_ego_trajs.reshape(
            outputs_ego_trajs.shape[0], self.ego_fut_mode, self.fut_ts, 2)

        outs = {'ego_fut_preds': outputs_ego_trajs}

        return outs
