"""
nuscenes_gs25600_base_decouple -- zero-weight DynamicLoss/PhysicsLoss ablation.

Everything is inherited from v12, including the GT-box motion-conditioned CTRA
offset head, detached gradient paths, dynamic head, loss computation graph, DDP
static-graph mode, data, and schedule. Only the two loss weights under study are
set to zero, preserving the exact v12 parameter-usage graph.
"""

_base_ = ['../nuscenes_gs25600_gtbox_oracle_v12.py']

loss = _base_.loss
loss.loss_cfgs[4].weight = 0.0  # DynamicLoss
loss.loss_cfgs[5].weight = 0.0  # PhysicsLoss