"""v13: v12 global-residual planner + explicit future-Gaussian risk gate.

设计（详见 model/planner/planner_v13.py）
--------------------------------------------------------------------
* base   = futattn 的逐帧路：ego 逐未来帧只与「该帧」未来高斯 cross-attn -> 碰撞安全。
* 残差   = timequery 的全局路：ego 对「所有」未来帧高斯 attn + joint MLP -> 低 L2。
* 风险   = 用 mean+offset、scale、opacity、semantic 构建连续未来 Gaussian 风险。
* 融合   = 三个导航模态分别预测 gate；global 风险高于 per-frame 时显式抑制残差。
* 训练   = gate oracle-ranking + hard-negative SAT，聚焦每帧最危险的 agent。

与 futattn / timequery_residual 的公平对比
--------------------------------------------------------------------
继续共享 `_base_ = ft_plan`：预训练仍为 v12_fixempty/epoch_15，max_epochs=15、
lr=2e-4、optimizer、dataset、scheduler 和所有基础参数保持不变。新增模块末层零初始化，
且所有实现均通过 custom_imports 隔离，不修改既有 v12 planner/loss 注册文件。
"""

custom_imports = dict(
    imports=[
        'model.planner.planner_v13',
        'loss.risk_aware_plan_loss',
    ],
    allow_failed_imports=False)

_base_ = [
    '../nuscenes_gs25600_gtbox_oracle_v12_ft_plan/'
    'nuscenes_gs25600_gtbox_oracle_v12_ft_plan.py'
]

import os

_planner_grad_scale = float(os.environ.get('PLANNER_GRAD_SCALE', 1.0))

# 继承 ft_plan：load_from = v12_fixempty/epoch_15、max_epochs=15、lr=2e-4、
# frozen_modules=[]、find_unused_parameters=False（与 futattn/timequery 完全一致）。
# 训练期关闭 EVAL：epoch(1..15) 永不整除 9999 -> train.py 每轮 continue 跳过。
eval_every_epochs = 9999

# 统一 decoder-layer 模板（embed_dims=128, 8 头, ffn=256），与两个对照实验一致。
_attn = dict(type='MultiheadAttention', embed_dims=128, num_heads=8, dropout=0.1)
_self_layer = dict(
    type='MyCustomBaseTransformerLayer',
    attn_cfgs=[_attn], feedforward_channels=256, ffn_dropout=0.1,
    batch_first=False, operation_order=('self_attn', 'norm', 'ffn', 'norm'))
_cross_layer = dict(
    type='MyCustomBaseTransformerLayer',
    attn_cfgs=[_attn], feedforward_channels=256, ffn_dropout=0.1,
    batch_first=False, operation_order=('cross_attn', 'norm', 'ffn', 'norm'))

model = dict(
    planner_head=dict(
        type='VADHeadFutAttnRiskAwareGlobalResidual',
        time_interval=0.5,
        num_fourier_bands=8,
        risk_topk=32,
        risk_margin=0.5,
        risk_uncertainty_growth=0.15,
        risk_safety_temperature=8.0,
        # Default 1.0 preserves the exact v12 planner->perception gradient.
        # PLANNER_GRAD_SCALE=0.1 is available only as a separate ablation.
        planner_gaussian_grad_scale=_planner_grad_scale,
        planner_offset_grad_scale=_planner_grad_scale,
        dynamic_semantic_dims=10,
        # --- futattn 逐帧碰撞安全 base ---
        fut_self_decoder=dict(
            type='CustomTransformerDecoder', num_layers=1,
            return_intermediate=False, transformerlayers=_self_layer),
        ego_fut_gaussian_decoder=dict(
            type='CustomTransformerDecoder', num_layers=1,
            return_intermediate=False, transformerlayers=_cross_layer),
        # --- timequery 全局低-L2 残差分支 ---
        global_fut_gaussian_decoder=dict(
            type='CustomTransformerDecoder', num_layers=1,
            return_intermediate=False, transformerlayers=_cross_layer),
    ),
)

# ---- Losses ---------------------------------------------------------------
# ft_plan 基础 loss（v12 6 loss + MapLoss + PlanLoss）之上追加：
#   1) 全局分支模仿损失 TimeQueryPlanLoss(weight=2.0)（对齐 timequery_residual 最优 L2）；
#   2) fused main 位置域监督 AlignedTrajectoryPositionLoss(weight=0.5)：PlanLoss 只在
#      位移域做 L1，缺累积位置监督 -> 远端 L2 漂移；此损失读 ego_fut_preds(=fused main)、
#      对 cumsum 轨迹加权（远端 1.5）监督，补齐 base 位置精度且不引入碰撞惩罚（与
#      time_aligned 保住 0.014 碰撞率的设定一致）；
#   3) PlanLoss 内 SAT 碰撞守卫（env COL_W，默认 0.1 开启）：门朝 global 打开会带来
#      global 天生偏高的碰撞，此守卫直接作用在 fused main 上做碰撞护栏；置 0 可关闭。
_aux_w = float(os.environ.get('AUX_W', 2.0))
_col_w = float(os.environ.get('COL_W', 0.1))
_gate_w = float(os.environ.get('GATE_W', 0.1))

_patched_loss_cfgs = []
for _cfg in list(_base_.loss['loss_cfgs']):
    _c = dict(_cfg)
    if _c.get('type') == 'PlanLoss':
        _c['type'] = 'RiskAwarePlanLoss'
        if _col_w > 0:
            _c['col_loss_weight'] = _col_w
            _c['col_sat'] = True
            _c['col_safe_margin'] = 0.5
            _c['col_temperature'] = 0.2
            _c['col_timestep_weights'] = (
                0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
    _patched_loss_cfgs.append(_c)

_aux_time_query_loss = dict(
    type='TimeQueryPlanLoss', weight=_aux_w, position_weight=1.0, beta=0.5)

# fused main 位置域监督（读 ego_fut_preds，base loss_input 已有键，无需额外 plumbing）。
_aligned_pos_loss = dict(
    type='AlignedTrajectoryPositionLoss', weight=0.5, beta=0.5,
    timestep_weights=(0.5, 0.75, 1.0, 1.0, 1.25, 1.5))

# 全局分支(aux)位置域约束：TimeQueryPlanLoss 只在位移域约束裸 global，且不看门控，
# 易把碰撞临界帧的 global 拉向穿障 GT。补一份低权重累积位置监督收敛全局轨迹形状。
_aligned_pos_aux = dict(
    type='AlignedTrajectoryPositionLoss', weight=0.3, beta=0.5,
    pred_key='ego_fut_aux_preds',
    timestep_weights=(0.5, 0.75, 1.0, 1.0, 1.25, 1.5))

# 逐帧碰撞安全 base(per_frame)显式位置监督：作为碰撞锚点，保 base 的位置精度，
# 让门开启的收益真正被利用（替代危险的"强制开门"）。
_aligned_pos_per_frame = dict(
    type='AlignedTrajectoryPositionLoss', weight=0.2, beta=0.5,
    pred_key='ego_fut_per_frame_preds',
    timestep_weights=(0.5, 0.75, 1.0, 1.0, 1.25, 1.5))

# 只监督命令选中的模态：当 global 的累计位置误差 + Gaussian 风险低于
# per-frame 时开门，否则保护 per-frame base。
_risk_gate_loss = dict(
    type='RiskAwareGateLoss', weight=_gate_w, risk_weight=1.0)

loss = dict(
    type='MultiLoss',
    loss_cfgs=_patched_loss_cfgs + [
        _aux_time_query_loss,
        _aligned_pos_loss,
        _aligned_pos_aux,
        _aligned_pos_per_frame,
        _risk_gate_loss,
    ],
)

# 全局分支轨迹从 planner 输出读取（dict 深合并，保留 base 的 plan/map 键）。
loss_input_convertion = dict(
    ego_fut_aux_preds='ego_fut_aux_preds',
    ego_fut_per_frame_preds='ego_fut_per_frame_preds',
    ego_fut_gate_logits='ego_fut_gate_logits',
    ego_fut_gate='ego_fut_gate',
    ego_fut_global_risk='ego_fut_global_risk',
    ego_fut_per_frame_risk='ego_fut_per_frame_risk',
)
