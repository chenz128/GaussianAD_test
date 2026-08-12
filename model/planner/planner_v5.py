"""VADHeadCircleGaussian —— 「循环 (circle) 递进注意力」planner。

设计动机
--------
VADHeadFutGaussian（planner_v3）让 ego_query 依次与 4 路信息做一次交叉注意力：

    ego -> ego↔agent -> ego↔map -> ego↔gaussian(当前) -> ego↔gaussian(未来)

再拼接 4 路特征回归轨迹。这是「单次前向」的信息融合：ego 只看了每一路一遍，
路与路之间缺乏反复的相互修正，导致规划对场景（尤其是碰撞约束）的推理不够充分，
碰撞率 (obj_box_col) 迟迟压不过基线。

本模块把上述「4 路递进注意力」封装成一个 block，按同样的操作循环 N 次
（circle / recurrent refinement，思路同 DETR 多层 decoder、Universal Transformer
的迭代推理）：每一轮 ego 会用上一轮融合后的表征，重新再看一遍 agent / map /
当前高斯 / 未来高斯，逐轮迭代收敛，实现信息的充分融合。

    for i in range(N):
        ego_agent  = ego↔agent (ego)
        ego_map    = ego↔map   (ego_agent)
        ego_gs     = ego↔gaussian_cur (ego_map)
        ego_futgs  = ego↔gaussian_fut (ego_gs)
        ego        = ego_futgs        # 迭代传递给下一轮

    最终用「最后一轮」的 4 路中间态拼接 (4D) -> ego_fut_decoder 回归轨迹。

与 VADHeadFutGaussian 的差异
----------------------------
- 新增 num_circles(N)：递进注意力 block 的循环次数（N=1 时等价于 planner_v3）。
- share_circle_weights：True 则各轮共享同一套注意力权重（RNN 式，参数少）；
  False（默认）则每轮独立权重（stacked，容量更大，更利于压碰撞率）。
- KV 源（agent/map/当前高斯/未来高斯的 embedding）只计算一次，循环中复用；
  只有 ego_query 在逐轮被 refine。

数据流（未来高斯构造）与 planner_v3 完全一致：
- results['gaussian_output']: (B=1, G, 28) 当前帧高斯
- results['offset']:          (B=1, G, fut_ts*2) flow 位移
  未来第 t 帧高斯.xy = 当前高斯.xy + offset[..., t, :]，其余属性不变。

输出 ego_fut_preds: (B, ego_fut_mode, fut_ts, 2)，与 PlanLoss / use_plan_ego 完全
兼容，不改动 head 与其它 planner。
"""
import copy

import torch
import torch.nn as nn
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmengine.registry import MODELS

from .planner import VADHead


@MODELS.register_module()
class VADHeadCircleGaussian(VADHead):
    """VADHead 变体：把 4 路递进交叉注意力封装为 block 并循环 N 次。

    额外参数：
        num_circles (int): 递进注意力 block 的循环次数 N（>=1）。
        ego_fut_gaussian_decoder (dict): ego 与未来帧高斯的交叉注意力配置。
        share_circle_weights (bool): 各轮是否共享注意力权重。默认 False（独立权重）。
    """

    def __init__(self,
                 *args,
                 num_circles=3,
                 ego_fut_gaussian_decoder=None,
                 share_circle_weights=False,
                 **kwargs):
        assert num_circles >= 1, 'num_circles 必须 >= 1'
        self.num_circles = num_circles
        self.share_circle_weights = share_circle_weights
        # 保存 4 路 decoder 的原始 config，供循环内复制成 N 套权重
        self._agent_dec_cfg = copy.deepcopy(kwargs.get('ego_agent_decoder'))
        self._map_dec_cfg = copy.deepcopy(kwargs.get('ego_map_decoder'))
        self._gs_dec_cfg = copy.deepcopy(kwargs.get('ego_gaussian_decoder'))
        self._futgs_dec_cfg = copy.deepcopy(ego_fut_gaussian_decoder)
        super().__init__(*args, **kwargs)

    def _build_decoder_list(self, cfg, num):
        """按 config 复制构建 num 套（独立权重）transformer decoder。"""
        return nn.ModuleList(
            [build_transformer_layer_sequence(copy.deepcopy(cfg))
             for _ in range(num)])

    def _init_layers(self):
        """父类初始化（mlp/ego_query/单路 decoder）+ 循环用的 N 套 decoder。"""
        super()._init_layers()

        # 父类已把单路 decoder build 好，但循环版不使用它们，删除以免多余参数
        for name in ('ego_agent_decoder', 'ego_map_decoder',
                     'ego_gaussian_decoder'):
            if hasattr(self, name):
                delattr(self, name)

        # 未来帧高斯特征编码（结构同当前帧 gaussian_fus_mlp，独立权重）
        self.fut_gaussian_fus_mlp = nn.Sequential(
            nn.Linear(28, self.embed_dims, bias=True),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims, bias=True))

        # 循环内每一路 decoder：share 时只建 1 套并复用，否则每轮独立
        n = 1 if self.share_circle_weights else self.num_circles
        self.ego_agent_decoders = self._build_decoder_list(self._agent_dec_cfg, n)
        self.ego_map_decoders = self._build_decoder_list(self._map_dec_cfg, n)
        self.ego_gaussian_decoders = self._build_decoder_list(self._gs_dec_cfg, n)
        self.ego_fut_gaussian_decoders = self._build_decoder_list(
            self._futgs_dec_cfg, n)

        # 输出头输入从 3D 加宽到 4D（保留原 MLP 结构，只改输入维度）
        ego_fut_dec_in_dim = self.embed_dims * 4
        ego_fut_decoder = []
        for _ in range(2):
            ego_fut_decoder.append(
                nn.Linear(ego_fut_dec_in_dim, ego_fut_dec_in_dim))
            ego_fut_decoder.append(nn.ReLU())
        ego_fut_decoder.append(
            nn.Linear(ego_fut_dec_in_dim, self.ego_fut_mode * self.fut_ts * 2))
        self.ego_fut_decoder = nn.Sequential(*ego_fut_decoder)

    def init_weights(self):
        """循环内所有交叉注意力层做 xavier 初始化。"""
        for dec_list in (self.ego_agent_decoders, self.ego_map_decoders,
                         self.ego_gaussian_decoders,
                         self.ego_fut_gaussian_decoders):
            for dec in dec_list:
                for p in dec.parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)

    def _build_future_gaussian_kv(self, results):
        """由当前高斯 + offset 构造未来所有帧高斯并编码，展平成一个 key/value 集合。

        返回 (fut_ts*G, B=1, D)，batch_first=False 约定，供 ego 单 query 交叉注意力。
        与 planner_v3.VADHeadFutGaussian 一致。
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

        means_xy = base[:, :2].unsqueeze(0) + off                    # (fut_ts, G, 2)
        rest = base[:, 2:].unsqueeze(0).expand(self.fut_ts, -1, -1)  # (fut_ts, G, 26)
        fut_feats = torch.cat([means_xy, rest], dim=-1)             # (fut_ts, G, 28)

        fut_gs = self.fut_gaussian_fus_mlp(fut_feats)              # (fut_ts, G, D)
        fut_gs = fut_gs.reshape(self.fut_ts * G, self.embed_dims).unsqueeze(1)
        return fut_gs

    def forward(self, results):
        agent_query, agent_mask = self.prepare_agent_query(results)
        map_query = self.prepare_map_query(results)
        gaussian_query = self.gaussian_fus_mlp(results['gaussian_output'])

        batch = agent_query.shape[0]
        ego_query = self.ego_query.weight.unsqueeze(0).repeat(batch, 1, 1)

        # KV 源只算一次（batch_first=False: (num_key, B, D)），循环内复用
        agent_kv = agent_query.permute(1, 0, 2)
        map_kv = map_query.permute(1, 0, 2)
        gs_kv = gaussian_query.permute(1, 0, 2)
        fut_gs_kv = self._build_future_gaussian_kv(results)   # (fut_ts*G, B, D)

        ego = ego_query.permute(1, 0, 2)                      # (1, B, D)
        ego_agent = ego_map = ego_gs = ego_futgs = None

        # ---------- 循环 N 次：每轮重做 4 路递进注意力，逐轮 refine ego ----------
        for i in range(self.num_circles):
            idx = 0 if self.share_circle_weights else i

            ego_agent = self.ego_agent_decoders[idx](
                query=ego, key=agent_kv, value=agent_kv,
                key_padding_mask=agent_mask)

            ego_map = self.ego_map_decoders[idx](
                query=ego_agent, key=map_kv, value=map_kv)

            ego_gs = self.ego_gaussian_decoders[idx](
                query=ego_map, key=gs_kv, value=gs_kv)

            ego_futgs = self.ego_fut_gaussian_decoders[idx](
                query=ego_gs, key=fut_gs_kv, value=fut_gs_kv)

            ego = ego_futgs   # 迭代：本轮融合结果作为下一轮 query

        # 用最后一轮的 4 路中间态拼接 (4D) -> 复用加宽后的 ego_fut_decoder
        ego_feats = torch.cat(
            [ego_agent.permute(1, 0, 2),
             ego_map.permute(1, 0, 2),
             ego_gs.permute(1, 0, 2),
             ego_futgs.permute(1, 0, 2)],
            dim=-1
        )  # [B, 1, 4D]

        outputs_ego_trajs = self.ego_fut_decoder(ego_feats)
        outputs_ego_trajs = outputs_ego_trajs.reshape(
            outputs_ego_trajs.shape[0], self.ego_fut_mode, self.fut_ts, 2)

        return {'ego_fut_preds': outputs_ego_trajs}
