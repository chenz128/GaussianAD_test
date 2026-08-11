"""
nuscenes_gs25600_gtbox_oracle_v12_ft_plan_frozen_futgau_detach_false
==================================================================
对照实验：在 futgau_detach_false 基础上，冻结「高斯预测模型」参数。

【目的】
验证：当高斯表征完全固定时，planner（VADHeadFutGaussian）单独学习「未来帧高斯
融合」是否仍能改善 future occ / plan 指标，从而判断 planner 的学习是来自高斯侧
的联合优化，还是仅凭固定高斯输入也能成立。

【与 futgau_detach_false 的唯一 delta】
  新增 frozen_modules 冻结高斯生成 + 高斯头：
      ['lifter', 'encoder', 'temporal_encoder', 'decoder', 'head']
  - lifter            : GaussianLifter  (相机特征 -> 3D 高斯)
  - encoder           : GaussianOccEncoder（当前帧高斯建模）
  - temporal_encoder  : GaussianTemporalEncoder（时序高斯聚合）
  - decoder           : VoxelNeXt（BEV 解码）
  - head              : GaussianHead（含 offset/occ 预测头）
  => 高斯的位置/形状/语义/offset 全部固定不动。

【保持不变（逐字节继承 futgau_detach_false）】
  planner_head        : VADHeadFutGaussian，保持可训练（结构不变，"planner 整体
                        不变"是指不改其结构，仅冻结它的高斯输入）。
  map_decoder         : 保持可训练。
  use_plan_ego=True / plan_ego_warmup_epochs=2 / plan_ego_detach=False：
                        仍读取 planner 预测的 ego 轨迹做 occ_flow 补偿，
                        occ_flow 一致性梯度仍回传 planner。
  load_from           : exp/nuscenes_gs25600_v12_fixempty/checkpoints/epoch_15.pth
  max_epochs          : 15（继承 base_plan），lr=2e-4（继承 base_plan）。
"""

_base_ = [
    '../nuscenes_gs25600_gtbox_oracle_v12_ft_plan_futgau_detach_false/nuscenes_gs25600_gtbox_oracle_v12_ft_plan_futgau_detach_false.py'
]

# 冻结「高斯预测模型」参数，planner_head / map_decoder 保持可训练
frozen_modules = ['lifter', 'encoder', 'temporal_encoder', 'decoder', 'head']
