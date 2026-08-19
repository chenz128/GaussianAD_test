"""planner_v12 —— 融合 futattn(逐帧碰撞安全) 与 timequery_residual(全局低 L2) 的优势。

设计动机（来自 planner 变体的 L2-vs-碰撞 研究）
------------------------------------------------------------------
* futattn (VADHeadFutAttn): ego 逐未来帧只与「该帧」未来高斯做 cross-attn
  (per-frame grounding)，每个 waypoint 都锚定在本帧占据上 -> 碰撞率优秀，
  但缺少全局轨迹形状，L2 偏高（远端尤甚）。
* timequery_residual (VADHeadTimeQueryResidual): 用 ego_fut_decoder 对「所有」
  未来帧高斯的全局特征联合回归整条轨迹 -> L2 最优；但它把全局路当主干、逐帧
  路当残差 (main = global + gate*(aux - global))，主干不是逐帧接地 -> 碰撞最差。

融合策略：翻转 timequery 的融合方向
------------------------------------------------------------------
本类以 **futattn 的逐帧路为碰撞安全 base**，把 **timequery 的全局路当作门控残差**：

    main = per_frame + gate * (global - per_frame)

- ``per_frame``：完全复用 futattn 的逐帧 cross-attn 路（碰撞安全），是被保护的基座。
- ``global``   ：timequery 的全局 joint MLP（ego 对所有未来帧高斯 attn），负责 L2。
- ``gate``     ：逐样本、逐时间步、输入相关的门；末层零初始化 (tanh(0)=0)，
  故初始时 main == per_frame（碰撞安全）。门同时看「全局摘要」与「逐帧接地特征」，
  只在信任全局形状能降 L2 的帧（通常远端）开启，碰撞临界帧保持关闭。

全局分支输出为 ``ego_fut_aux_preds``，由独立的模仿损失(TimeQueryPlanLoss)监督，
使其梯度不依赖门（避免 zero-gate 饿死残差分支的 dead-gate 失效）。逐帧基座输出
额外暴露为 ``ego_fut_per_frame_preds`` 便于调试/可视化。

从 v12_fixempty/epoch_15 从头训练（planner 无预训练权重），与 futattn /
timequery_residual 同起点、同调度，便于公平对比。
"""
import math

import torch
import torch.nn as nn
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmengine.registry import MODELS

from .planner_v2 import VADHeadFutAttn


@MODELS.register_module()
class VADHeadFutAttnGlobalResidual(VADHeadFutAttn):
    """futattn 逐帧碰撞安全 base + timequery 全局低-L2 门控残差。"""

    def __init__(self,
                 *args,
                 global_fut_gaussian_decoder=None,
                 time_interval=0.5,
                 num_fourier_bands=8,
                 **kwargs):
        self.global_fut_gaussian_decoder = global_fut_gaussian_decoder
        self.time_interval = time_interval
        self.num_fourier_bands = num_fourier_bands
        super().__init__(*args, **kwargs)

    def _init_layers(self):
        # 复用 futattn 的逐帧 base 层：fut_gaussian_fus_mlp / ego_to_fut / fut_pos /
        # fut_self_decoder / ego_fut_gaussian_decoder / fut_out_mlp
        super()._init_layers()

        # ---- timequery 的全局分支：ego 对所有未来帧高斯做 cross-attn ----
        if self.global_fut_gaussian_decoder is not None:
            self.global_fut_gaussian_decoder = build_transformer_layer_sequence(
                self.global_fut_gaussian_decoder)

        # 未来帧高斯 key 的逐时间步编码（连续 Fourier + 可学习），与 timequery 一致。
        self.learned_time_pos = nn.Embedding(self.fut_ts, self.embed_dims)
        periods = torch.logspace(
            math.log10(self.time_interval),
            math.log10(self.time_interval * self.fut_ts * 2),
            steps=self.num_fourier_bands)
        self.register_buffer('time_frequencies', periods.reciprocal())
        self.time_fourier_proj = nn.Linear(
            1 + 2 * self.num_fourier_bands, self.embed_dims)
        self.time_encoding_norm = nn.LayerNorm(self.embed_dims)

        # 全局 joint 轨迹形状 MLP：4D 全局特征 -> 整条轨迹（timequery 的 ego_fut_decoder）。
        global_dim = self.embed_dims * 4
        self.global_shape_mlp = nn.Sequential(
            nn.Linear(global_dim, global_dim),
            nn.ReLU(),
            nn.Linear(global_dim, global_dim),
            nn.ReLU(),
            nn.Linear(global_dim, self.ego_fut_mode * self.fut_ts * 2))

        # 碰撞感知融合门：输入 = 全局摘要(4D, 逐帧广播) + 逐帧接地特征(D)。
        # 末层在 init_weights 里零初始化 -> tanh(0)=0 -> 初始 main == per_frame。
        self.refine_gate_mlp = nn.Sequential(
            nn.Linear(global_dim + self.embed_dims, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, 1))

    def init_weights(self):
        super().init_weights()
        if self.global_fut_gaussian_decoder is not None:
            for parameter in self.global_fut_gaussian_decoder.parameters():
                if parameter.dim() > 1:
                    nn.init.xavier_uniform_(parameter)
        nn.init.xavier_uniform_(self.learned_time_pos.weight)
        # 关闭残差：初始时融合输出精确等于碰撞安全逐帧 base。
        nn.init.zeros_(self.refine_gate_mlp[-1].weight)
        nn.init.zeros_(self.refine_gate_mlp[-1].bias)

    def _build_time_encoding(self, reference):
        times = torch.arange(
            1, self.fut_ts + 1, device=reference.device,
            dtype=reference.dtype) * self.time_interval
        phases = 2 * math.pi * times[:, None] * self.time_frequencies.to(
            device=reference.device, dtype=reference.dtype)[None, :]
        normalized_time = (times / (self.time_interval * self.fut_ts))[:, None]
        fourier_features = torch.cat(
            [normalized_time, phases.sin(), phases.cos()], dim=-1)
        continuous_encoding = self.time_fourier_proj(fourier_features)
        time_indices = torch.arange(self.fut_ts, device=reference.device)
        learned_encoding = self.learned_time_pos(time_indices)
        return self.time_encoding_norm(
            continuous_encoding + learned_encoding).to(reference.dtype)

    def _build_future_gaussians(self, results):
        """由当前高斯 + offset 平移构造逐未来帧高斯特征（batch-general）。

        返回 future_content/(B,fut_ts,G,D)、future_key(含时间编码)、time_encoding(fut_ts,D)。
        """
        gaussian_output = results['gaussian_output']            # (B, G, 28)
        batch, num_gaussians = gaussian_output.shape[:2]
        offset = results.get('offset')
        if offset is None:
            future_offset = gaussian_output.new_zeros(
                (batch, self.fut_ts, num_gaussians, 2))
        else:
            future_offset = offset.reshape(
                batch, num_gaussians, self.fut_ts, 2).permute(0, 2, 1, 3)
            future_offset = torch.nan_to_num(
                future_offset, nan=0.0, posinf=0.0, neginf=0.0)
        future_xy = gaussian_output[:, None, :, :2] + future_offset
        future_rest = gaussian_output[:, None, :, 2:].expand(
            -1, self.fut_ts, -1, -1)
        future_features = torch.cat([future_xy, future_rest], dim=-1)
        future_content = self.fut_gaussian_fus_mlp(future_features)
        time_encoding = self._build_time_encoding(future_content)
        future_key = future_content + time_encoding[None, :, None, :]
        return future_content, future_key, time_encoding

    def forward(self, results):
        agent_query, agent_mask = self.prepare_agent_query(results)
        map_query = self.prepare_map_query(results)
        gaussian_query = self.gaussian_fus_mlp(results['gaussian_output'])

        batch = agent_query.shape[0]
        ego_query = self.ego_query.weight.unsqueeze(0).expand(batch, -1, -1)
        ego_agent_query = self.ego_agent_decoder(
            query=ego_query.permute(1, 0, 2),
            key=agent_query.permute(1, 0, 2),
            value=agent_query.permute(1, 0, 2),
            key_padding_mask=agent_mask)
        ego_map_query = self.ego_map_decoder(
            query=ego_agent_query,
            key=map_query.permute(1, 0, 2),
            value=map_query.permute(1, 0, 2))
        ego_gaussian_query = self.ego_gaussian_decoder(
            query=ego_map_query,
            key=gaussian_query.permute(1, 0, 2),
            value=gaussian_query.permute(1, 0, 2))

        current_features = torch.cat([
            ego_agent_query.permute(1, 0, 2),
            ego_map_query.permute(1, 0, 2),
            ego_gaussian_query.permute(1, 0, 2),
        ], dim=-1)                                              # (B, 1, 3D)

        future_content, future_key, _ = self._build_future_gaussians(results)
        num_gaussians = future_content.shape[2]

        # ---- 碰撞安全逐帧 base（futattn）：每帧只对本帧未来高斯 cross-attn ----
        fut_ego = self.ego_to_fut(current_features)            # (B, 1, D)
        fut_ego = fut_ego.expand(-1, self.fut_ts, -1)          # (B, fut_ts, D)
        fut_ego = fut_ego + self.fut_pos.weight[None, :, :]    # (B, fut_ts, D)
        fut_query = fut_ego.permute(1, 0, 2)                   # (fut_ts, B, D)
        if self.fut_self_decoder is not None:
            fut_query = self.fut_self_decoder(
                query=fut_query, key=fut_query, value=fut_query)
        folded_query = fut_query.permute(1, 0, 2).reshape(
            batch * self.fut_ts, self.embed_dims).unsqueeze(0)  # (1, B*fut_ts, D)
        folded_kv = future_content.reshape(
            batch * self.fut_ts, num_gaussians,
            self.embed_dims).permute(1, 0, 2)                   # (G, B*fut_ts, D)
        if self.ego_fut_gaussian_decoder is not None:
            folded_query = self.ego_fut_gaussian_decoder(
                query=folded_query, key=folded_kv, value=folded_kv)
        per_frame_features = folded_query.squeeze(0).reshape(
            batch, self.fut_ts, self.embed_dims)               # (B, fut_ts, D)
        per_frame_prediction = self.fut_out_mlp(per_frame_features).reshape(
            batch, self.fut_ts, self.ego_fut_mode, 2).permute(0, 2, 1, 3)

        # ---- 全局低-L2 分支（timequery）：ego 对所有未来帧高斯 attn + joint MLP ----
        global_key = future_key.reshape(
            batch, self.fut_ts * num_gaussians,
            self.embed_dims).permute(1, 0, 2)
        global_value = future_content.reshape(
            batch, self.fut_ts * num_gaussians,
            self.embed_dims).permute(1, 0, 2)
        if self.global_fut_gaussian_decoder is not None:
            global_future_query = self.global_fut_gaussian_decoder(
                query=ego_gaussian_query, key=global_key, value=global_value)
        else:
            global_future_query = ego_gaussian_query
        global_features = torch.cat([
            current_features,
            global_future_query.permute(1, 0, 2),
        ], dim=-1)                                             # (B, 1, 4D)
        global_prediction = self.global_shape_mlp(global_features).reshape(
            batch, self.ego_fut_mode, self.fut_ts, 2)

        # ---- 碰撞感知零初始化门控融合 ----
        # 门同时看全局摘要与逐帧接地特征，只在信任全局能降 L2 的帧开启。
        gate_in = torch.cat([
            global_features.expand(-1, self.fut_ts, -1),
            per_frame_features,
        ], dim=-1)                                             # (B, fut_ts, 5D)
        gate = self.refine_gate_mlp(gate_in).tanh().reshape(
            batch, 1, self.fut_ts, 1)
        # 近端 gate 限流：近端帧碰撞最敏感且逐帧 base 的 L2 本就最优，
        # 把门在近端压低(系数~0.2)、远端放开(~1.0)，让全局残差主要在远端注入。
        # time_scale 与门相乘，门初始为 0 -> main==per_frame 仍精确成立。
        time_scale = (0.2 + 0.8 * torch.arange(
            1, self.fut_ts + 1, device=gate.device, dtype=gate.dtype)
            / self.fut_ts).reshape(1, 1, self.fut_ts, 1)
        gate = gate * time_scale
        main_prediction = per_frame_prediction + gate * (
            global_prediction - per_frame_prediction)

        return {
            'ego_fut_preds': main_prediction,
            'ego_fut_aux_preds': global_prediction,
            'ego_fut_per_frame_preds': per_frame_prediction,
        }
