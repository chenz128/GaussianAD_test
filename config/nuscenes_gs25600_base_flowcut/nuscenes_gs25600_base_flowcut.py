"""
nuscenes_gs25600_base_flowcut —— 消融实验：只切断 OccFlowLoss 的 B 路梯度

【动机】
v12_fixempty vs base_gt_ego_fixempty 的实测差距为
    current mIoU  15.103 → 17.577  (+2.474)
    FutAvg         8.33  → 10.92   (+2.59)

代码层面已确认（refine_module.py / gaussian_head.py）：
  - GT box 只条件化 offset 头（_motion_offset），不进入当前帧高斯的
    means/scales/rotations/opacities/semantics；
  - decouple_offset(offset_grad_scale=0.0) / decouple_dynamic 均为 feat.detach()，
    DynamicLoss / PhysicsLoss 的梯度到不了 encoder。
=> 因此 current mIoU 的 +2.47 在数学上不可能来自 gtbox oracle，只能来自
   head.flow_grad_scale 1.0 → 0.0（切断 OccFlowLoss 回灌当前帧高斯的 B 路）。

但 v12 相对 base 同时改了 4 件事（flow 解耦 / dynamic 解耦 / offset 解耦 /
新增 Dynamic+Physics loss），上述推断仍属"由代码结构反推"，缺一次直接测量。

【本实验 = 严格单变量】
    base_gt_ego  +  head.flow_grad_scale = 0.0
其余（loss 组成仅 occ+flow+det、无 DynamicLoss/PhysicsLoss、
refine_layer.use_dynamic=False、decouple_offset=False、数据子集 3000/2000
seed=42、冻结 map_decoder+planner_head、max_epochs=15）与 base_gt_ego 完全相同。

【预期与判读】
  - 若 current mIoU 从 15.10 涨到 ~17.5 → +2.47 几乎全部由"切 B 路"贡献，
    动静分离/物理约束对当前帧的净贡献 ≈ 0（它们的价值在未来帧）；
  - 若只涨到 ~16 → 切 B 路贡献一半，另一半来自 Dynamic/Physics 带来的表征改善；
  - 若几乎不涨 → 说明 B 路污染的推断有误，需重新审视 v12 的增益来源。
  - FutAvg 预期低于 v12 的 10.92：本实验的 offset 头没有 PhysicsLoss 的 GT 轨迹
    监督，也没有 GT box 的运动条件，纯靠 OccFlowLoss 自学。该差值即为
    "物理约束 + gtbox 运动条件" 对未来帧的贡献量。

【注】empty-gaussian bugfix (cfb1356, flow_include_empty 默认 True) 已在代码中生效，
与 v12_fixempty / base_gt_ego_fixempty 同口径，三者可直接横向对比。

机器：h20-new  ssh -p 32344 root@8.130.174.55，GPU 4,5,6,7。启动见同目录 train.sh。
"""

_base_ = ['../nuscenes_gs25600_base_gt_ego/nuscenes_gs25600_base_gt_ego.py']

# 唯一自变量：未来帧分支对当前帧高斯的梯度回传比例。
# 直通混合 s*x + (1-s)*x.detach()：前向值不变，只有反向被缩放。
model = dict(
    head=dict(
        flow_grad_scale=0.0,
    ),
)

# flow_grad_scale 只改梯度缩放，每轮参与的参数集合不变，static_graph 依然成立。
static_graph = True
