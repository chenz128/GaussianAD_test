"""
nuscenes_gs25600_v1_full_decouple 实验配置（GaussianFormer V1：完全解耦）

在 nuscenes_gs25600_v1_instfeat（半解耦）基础上，进一步去掉 encoder refine
loop 里的 `instance_feature += anchor_embed`，使 query（instance_feature）在
全部 decoder 层都独立于 anchor 几何——这才是 GaussianFormer V1 的完全解耦。

与半解耦（v1_instfeat）的唯一区别：
  - encoder.decouple_feat=True （半解耦为默认 False）

其余全部继承 v1_instfeat：
  - lifter: 随机 xyz、feat_grad=False（冻结零向量独立 instance_feature）
  - occ+flow+det 三个 loss，max_epochs=15，num_samples=3000/2000 subsample_seed=42
  - 与 base / 半解耦 三方可直接对照

注意：decouple_feat=True 仅跳过 `instance_feature += anchor_embed` 这一步的
几何再注入；anchor_embed 仍照常重算供下一层 deformable 使用。
"""

_base_ = ['./nuscenes_gs25600_v1_instfeat.py']

model = dict(
    encoder=dict(
        decouple_feat=True,
    ),
)
