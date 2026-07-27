"""
nuscenes_gs25600_gtbox_oracle_v12 实验配置
v12 = v10fix（历史最佳：current 12.84 / FutAvg 6.57）+ 切断未来帧对当前帧的梯度。

【背景：一条一直没被发现的耦合路径】
运动相关损失通往 encoder 有两条路：
  A) PhysicsLoss → offset 头 → feat → encoder
     ——v6 起已用 decouple_offset + offset_grad_scale=0.0 堵住。
  B) OccFlowLoss → forward_flow 里直接取用的 gaussian.means / opacity /
     semantics / scales → encoder
     ——**从来没堵过**。它不经过 offset，所以"offset 已解耦"并不能覆盖它。

未来 occ 的构造是 `当前高斯(+offset) - ego位移`。因此未来帧的误差会反过来
重塑当前帧的高斯（位置/语义/透明度/尺度）。在 base 里 offset 是自由的，flow
容易满足，这条耦合温和；而 oracle 用 PhysicsLoss 把 offset 钉在真实 GT 轨迹上
之后，flow 剩下的残差（新出现区域、出网格被丢弃等运动无法解释的部分）只能转嫁
到当前高斯上 → 当前帧语义被牺牲。

这与观测一致：13 个 oracle 实验没有一个 current mIoU 追平 base15(14.15)，最好的
v10fix 也差 1.31；且 per-class 里受伤最重的正是 trailer / bus / truck /
construction_vehicle 这些被 offset 与 flow 施力最重的大框可动类。
另外 v11 在 s=0（全解耦）下 current 仍掉了 1.15，正是这条 B 路径造成的——它证伪了
"s=0 ⇒ 当前帧不受影响"的旧假设。

【唯一自变量】head.flow_grad_scale：未来帧梯度回传当前高斯的比例。
  1.0 = 旧行为（完全耦合）
  0.0 = 未来分支只训练 offset，当前帧仅由 OccupancyLoss + DetectionLoss 决定
直通混合 `s*x + (1-s)*x.detach()`：**前向值不变**（未来 occ 仍用真实高斯构建，
静态背景的"复制 + ego 补偿"照常精确，loss_static 钉住静态的收益完全保留），
只有反向被缩放。

【预期】
  - current mIoU：从 12.84 回升，目标逼近 base15 的 14.15
  - FutAvg：未来帧 = f(当前高斯) ⊗ g(offset)，当前高斯变好未来帧应当搭便车
    （base15→base30 即 current +1.30 → FutAvg +0.31），故预期 ≥ 6.57
  - 若 current 回升而 FutAvg 下跌，则说明"未来帧靠牺牲当前帧换分"，需重新审视

【代价】切断 B 路径也就放弃了"运动/未来信息反哺当前帧"的正迁移可能。因此做成
可调刻度而非硬编码：将来想探正迁移时把 flow_grad_scale 调到 0.1/0.3 即可。

【不含】v11 的 MotionCrossAttention（继承 v10 → use_motion_attn 默认 False），
保证 v10fix → v12 是干净的单变量对照。

其余（GT-box oracle 动静门控、有界 CTRA offset 头、decouple_offset=True/
offset_grad_scale=0.0、decouple_dynamic=True、loss 权重、3000 子集、冻结 map/plan）
与 v10 完全相同。
GPU：h20-new 后 4 张（4,5,6,7）。
"""

_base_ = ['./nuscenes_gs25600_gtbox_oracle_v10.py']

# 唯一自变量：未来帧分支对当前帧高斯的梯度回传比例。
model = dict(
    head=dict(
        flow_grad_scale=0.0,   # 0 = 未来帧只训 offset，当前帧完全受保护
    ),
)

# 与 v10 相同：静态图 + backbone checkpoint。flow_grad_scale 只改变梯度缩放，
# 每轮参与的参数集合不变，static_graph 依然成立。
static_graph = True
