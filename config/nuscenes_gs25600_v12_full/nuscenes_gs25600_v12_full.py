"""
nuscenes_gs25600_v12_full —— v12（empty-gaussian bugfix 之后）的全量数据复跑

与 config/nuscenes_gs25600_gtbox_oracle_v12.py 相比，唯一改动是数据规模与轮数：
  - train : 3000 子集 → 全量（num_samples=0）
  - val   : 2000 子集 → 全量（num_samples=0）
  - epochs: 15 → 20（对齐 out/nuscenes_gs25600_4gpu_v4 那次 4 卡全量实验）

模型结构、loss 权重、GT-box oracle 门控、flow_grad_scale=0.0 等全部原样继承 v12，
不做任何改动；代码侧含 empty-gaussian bugfix（cfb1356，flow_include_empty=True）。

机器：h20-old  ssh -p 30300 root@8.130.174.55，GPU 4,5,6,7
"""

_base_ = ['../nuscenes_gs25600_gtbox_oracle_v12.py']

max_epochs = 20

# num_samples=0 -> dataset 不做子采样，使用 pkl 中的全部 keyframe
train_dataset_config = dict(num_samples=0)
val_dataset_config = dict(num_samples=0)
