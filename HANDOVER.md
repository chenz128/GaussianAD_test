# GaussianAD 项目交接文档

本文档面向第一次接手项目的同事，记录当前仓库的代码结构、默认训练路径、数据要求、实验分支和常见故障。当前文档对应 `splatting` 分支；不要把历史实验配置当成默认配置使用。

## 1. 项目定位

GaussianAD 是一个基于多摄像头图像的自动驾驶感知模型，输入 nuScenes 六相机图像和时序信息，输出：

- 3D occupancy 语义预测；
- 未来 0.5 到 3.0 秒的 occupancy flow；
- 3D 检测；
- BEV 地图矢量；
- ego 未来轨迹规划。

模型以 3D Gaussian 作为中间表示。默认配置使用 4 帧输入、25600 个 Gaussian、ResNet-101 + FPN 图像骨干和 128 维 Gaussian 特征。

## 2. 分支、环境和路径

| 用途 | Git 分支 | Python 环境 | 说明 |
| --- | --- | --- | --- |
| 默认开发分支 | `splatting` | `/data/chenz/conda_env/splatting` | 当前接手和继续实验的分支 |
| 原始 baseline | `main` | `/data/chenz/conda_env/GaussianAD` | 标准 occ/det/map 等监督 |
| 训练加速实验 | `faster` | `/data/chenz/conda_env/faster` | 基于 `splatting` 的性能实验 |

仓库远端：`https://github.com/chenz128/GaussianAD_test.git`。

H20 远端运行目录按项目约定为 `/data/chenz/GaussianAD`。本地编辑、提交和 push，远端只 pull、编译和运行。`/data/chenz/conda_env/` 下的环境不是完整 Conda 安装，不要使用 `conda activate`，直接调用绝对路径的 `python` 或 `torchrun`。

H20-new 只允许使用 GPU 4-7；训练前先检查显存和正在运行的进程，不要停止其他人的任务。

## 3. 远端同步和启动

本地修改后：

```bash
git status
git switch splatting
git add <files>
git commit -m "describe the change"
git push origin splatting
```

远端更新：

```bash
ssh -p 32344 root@8.130.174.55 \
  "cd /data/chenz/GaussianAD && git pull origin splatting && git status"
```

默认训练入口：

```bash
cd /data/chenz/GaussianAD
CUDA_VISIBLE_DEVICES=4,5,6,7 \
/data/chenz/conda_env/splatting/bin/torchrun \
  --nproc_per_node 4 --master_port 12457 \
  train.py \
  --py-config config/nuscenes_gs25600.py \
  --work-dir out/nuscenes_gs25600_splatting \
  --dataset nuscenes
```

单卡冒烟测试可使用：

```bash
CUDA_VISIBLE_DEVICES=4 /data/chenz/conda_env/splatting/bin/python train.py \
  --py-config config/nuscenes_gs25600.py \
  --work-dir out/debug_splatting --dataset nuscenes
```

`train.py` 会自动检测 `<work-dir>/latest.pth`。因此使用相同 `--work-dir` 重启即可续训，不需要额外的 resume 参数。也可以显式指定：

```bash
--resume-from out/nuscenes_gs25600_splatting/checkpoints/epoch_10.pth
```

训练每个 epoch 保存 `checkpoints/epoch_N.pth`，并把 `latest.pth` 链接到最新 checkpoint。若修改了冻结模块或 loss 参数，optimizer/scheduler 状态可能不兼容，代码会跳过对应状态恢复并继续加载模型权重。

## 4. 数据准备

默认配置依赖：

```text
data/nuscenes/
data/nuscenes_cam/nuscenes_infos_train_gaussian_ad_v4.pkl
data/nuscenes_cam/nuscenes_infos_val_gaussian_ad_v4.pkl
data/surroundocc/train_samples/
data/surroundocc/val_samples/
```

PKL 至少需要包含 `infos`、`metadata`、相机和 LIDAR 外参、occupancy 路径、检测框、地图标注、ego/agent future 标注等字段。转换和检查工具：

```bash
/data/chenz/conda_env/GaussianAD/bin/python \
  tools/data/convert_nuscenes_infos_to_gaussianad.py \
  --dataroot data/nuscenes --version v1.0-trainval \
  --surroundocc-train-dir data/surroundocc/train_samples \
  --surroundocc-val-dir data/surroundocc/val_samples

/data/chenz/conda_env/GaussianAD/bin/python \
  tools/data/stats_gaussianad_pkl.py \
  --pkl data/nuscenes_cam/nuscenes_infos_train_gaussian_ad_v6.pkl
```

转换后的 PKL 要重点检查 `scene_token`/`scene_name`、真实 `num_lidar_pts`、LIDAR 坐标系速度、ego future 命令比例和地图线数量。伪标签路径来自配置，而不是代码硬编码：

```text
/data/chenz/Gaussianflowocc_test/data/grounded_sam_nusc/
/data/chenz/Gaussianflowocc_test/data/metric_3d_nusc/
```

每个文件是 `{scene_name}/{sample_token}.npy`，六相机原始尺寸约为 `(6, 900, 1600)`。只有配置同时设置 `metric3d_root` 和 `grounded_sam_root` 时，dataset 才会加载伪标签。

## 5. 代码主链

```text
train.py
  -> mmseg.build_segmentor(cfg.model)
  -> BEVSegmentor.forward
       -> img_backbone + img_neck
       -> GaussianLifter
       -> GaussianOccEncoder
       -> GaussianTemporalEncoder
       -> Gaussian decoder
       -> map_decoder
       -> planner_head
       -> GaussianHead
  -> MultiLoss
```

主要文件：

| 文件 | 职责 |
| --- | --- |
| `train.py` | DDP 初始化、模型构建、loss、optimizer、scheduler、训练/epoch 验证、checkpoint |
| `test.py` | 完整验证，输出当前和未来 occupancy mIoU、规划指标等 |
| `eval.py` | 更轻量的 occupancy 验证入口 |
| `dataset/dataset.py` | nuScenes PKL 读取、多帧采样、图像/occupancy/地图/检测/规划 GT 组织、伪标签读取 |
| `dataset/transform_3d.py` | 多相机图像增强、格式转换和数据收集 |
| `model/segmentor/bev_segmentor.py` | 图像骨干到各个 Gaussian/地图/规划模块的总编排 |
| `model/encoder/gaussian_encoder/` | Gaussian 特征编码和位置、尺度、旋转、语义、不透明度 refinement |
| `model/encoder/temporal_encoder/` | 历史帧和当前帧 Gaussian 的时序融合 |
| `model/head/gaussian_head.py` | LocalAggregator 3D occupancy、future flow、可选 2D render 输出 |
| `model/head/localagg/` | CUDA 3D Gaussian 到 occupancy 采样点的聚合实现 |
| `model/head/gaussian_rasterizer.py` | 可选 gsplat 2D 语义/深度/动态渲染 |
| `loss/multi_loss.py` | 组合多个 loss，并可按 group 做 PCGrad |
| `loss/occupancy_loss.py` | 当前 3D occupancy 监督 |
| `loss/occupancy_loss_flow.py` | 未来 occupancy flow 监督 |
| `loss/detection_loss.py` | VoxelNeXt 风格检测分类和回归监督 |
| `loss/map_loss.py` | MapTR 风格地图矢量监督 |
| `loss/plan_loss.py` | ego future trajectory 监督 |
| `loss/render_loss.py` | 可选伪标签语义/深度/多帧深度监督 |

### Dataset 输出的关键数据

`NuScenesDataset.__getitem__` 先读取当前帧、历史帧和未来六帧，再执行 pipeline。常见关键字段：

- `img`：整理后的多帧六相机图像；
- `occ_xyz`、`occ_label`、`occ_cam_mask`：3D occupancy 采样点和标签；
- `gt_boxes`、`gt_names`、`gt_velocity`：检测 GT；
- `gt_map`：地图矢量 GT；
- `ego_fut_trajs`、`ego_fut_cmd`：规划 GT；
- `flow_info`、`future_lidar2global`：未来 occupancy flow；
- `pseudo_seg`、`pseudo_depth`、`gs_extrins`、`gs_intrins`：启用 2D gsplat 时的伪标签和相机参数。

当前 pipeline 的 `use_ego=False`，Gaussian 和 occupancy 在 LIDAR 坐标系中处理。2D 渲染外参因此是 `lidar2cam = ego2cam @ lidar2ego`，不能直接把 `ego2cam` 当成 LIDAR 到相机的变换。

## 6. 默认 loss 和配置

默认入口 `config/nuscenes_gs25600.py` 继承：

```text
config/_base_/misc.py
config/_base_/model.py
config/_base_/surroundocc.py
```

当前默认配置中的 `MultiLoss` 包含：

1. `OccupancyLoss`：18 通道，0-16 为语义类，17 为空体；
2. `OccupancyFlowLoss`：未来六个 0.5 秒时间步；
3. `DetectionLoss`：10 类 3D 检测；
4. `MapLoss`：divider、ped_crossing、boundary；
5. `PlanLoss`：权重默认 10.0。

默认配置当前没有把 `RenderLoss` 加入 loss 列表，也没有开启 `render_config`。因此默认训练不依赖 gsplat 伪标签。2D 伪标签路线使用这些配置作为起点：

- `config/nuscenes_gs25600_2D.py`：基础 2D RenderLoss；
- `config/nuscenes_gs25600_render_focus.py`：降低其他目标、强调 2D render；
- `config/nuscenes_gs25600_mf_depth.py`：历史/未来多帧深度；
- `config/nuscenes_gs25600_pcgrad.py`：把 RenderLoss 作为 aux group 做梯度投影；
- `config/nuscenes_gs25600_acc.py`、`config/nuscenes_gs25600_concentrate.py`：累积透明度和深度集中性实验；
- `config/nuscenes_gs25600_depth_init.py`：用深度反投影初始化 Gaussian，不等同于 RenderLoss。

2D render 的关键参数通常是 `pseudo_label_scale=0.44`、`pseudo_label_crop_top=140`、`max_pseudo_depth=40.0`。渲染使用 `semantics_logits`，而不是 softplus 后的 `semantics`；否则 alpha blending 后不能直接用于 CE。pseudo label 的类别标签 1-16 与 Gaussian 语义通道直接对应，不要再减 1。

## 7. 验证与结果查看

完整验证示例：

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 \
/data/chenz/conda_env/splatting/bin/torchrun \
  --nproc_per_node 4 --master_port 12458 \
  test.py \
  --py-config config/nuscenes_gs25600.py \
  --work-dir out/nuscenes_gs25600_splatting
```

`test.py` 自动优先读取相同 `work-dir/latest.pth`，并汇总：当前 occupancy mIoU、未来 0.5-3.0 秒各时间点 mIoU、平均未来 mIoU，以及规划指标。训练过程中的 epoch 验证只走 3D occupancy 路径；2D render 在 eval 模式会跳过以节约时间，所以 RenderLoss 在验证时为 0 属于预期行为。

常用日志检查：

```bash
grep -E "successfully resumed|mIoU|Future|Traceback|RuntimeError" out/<work-dir>/*.log | tail -50
```

## 8. 性能和实验注意事项

- `with_cp=True` 会节省显存但增加重计算；当前 `faster` 分支已验证 `with_cp=False` 只有约 2-3% 实测收益，主要瓶颈仍是 spconv SubMConv backward 的显存分配。
- `history_no_grad=True` 只让历史帧 backbone/encoder 不回传梯度，temporal encoder 仍接收完整时序；这是速度/显存实验开关，不要和数据帧数混淆。
- channels-last、提前转换 `lidar2global`、bool mask、DDP bucket 和 allocator 参数的收益已做过实测，整体收益很小，不建议优先继续改这些上层路径。
- `flow_grad_scale=0.0` 表示 future flow loss 不回传到当前帧 Gaussian，只训练 offset；改回 1.0 会恢复耦合行为。
- `use_plan_ego=True` 会让 future flow 使用 planner 预测的 ego 轨迹；可配 `plan_ego_warmup_epochs`，否则 planner 初期不稳定可能污染 flow 监督。
- `with_empty=True` 时会额外加入空体 Gaussian。future flow 的空体处理由 `flow_include_empty` 控制，修改前先确认空体监督是否仍有贡献。

## 9. 常见故障

### 环境激活失败

不要 `conda activate /data/chenz/conda_env/splatting`，直接使用：

```bash
/data/chenz/conda_env/splatting/bin/python -c "import torch; print(torch.__version__)"
/data/chenz/conda_env/splatting/bin/python -c "from gsplat import rasterization; print('gsplat ok')"
```

### `KeyError: ori_intrinsic` 或 `cam2ego`

pipeline 会原地修改 `input_dict`。dataset 已在 pipeline 前保存并在之后恢复这些字段。如果继续修改 dataset，必须保留这一顺序。

### `TypeError: NoneType is not callable` 出现在 RenderLoss

`BaseLoss.__init__` 会设置实例属性 `self.loss_func`，它会遮蔽 RenderLoss 的同名方法。RenderLoss 当前显式 `del self.loss_func`，不要删除这个处理。

### RenderLoss 接近 `log(17)` 或 bicycle 长期为 0

先看 `[RenderLoss Diag]`：`pred_depth_mean` 接近 0 通常是相机坐标变换错误；语义 entropy 接近 `log(17)` 表示语义尚未收敛。检查类别索引是否直接使用 1-16、相机顺序是否已重排、是否误用了 `ego2cam` 替代 `lidar2cam`。

### DDP hang 或显存异常

先确认每个进程的 `CUDA_VISIBLE_DEVICES`、`--nproc_per_node`、当前 GPU 是否属于允许范围；再确认 `find_unused_parameters=False` 与当前配置中的冻结模块一致。不要在共享服务器上直接 kill 未确认归属的进程。

## 10. 接手后的推荐顺序

1. 先在远端确认 `splatting` 分支、环境、数据 PKL 和 occupancy 文件存在。
2. 用默认配置单卡跑一个短检查，确认模型能构建、dataset 能取样、checkpoint 能保存。
3. 用已有 `work-dir` 做完整验证，记录当前 occupancy、future flow、规划指标作为接手基线。
4. 需要继续某个实验时，先阅读对应配置文件顶部说明，并核对它继承的基础配置；不要只凭目录名判断实验内容。
5. 每次实验使用独立 `work-dir`，保存实际配置、日志和 checkpoint；代码改动通过本地 commit/push 后再让远端 pull。

## 11. 代码维护边界

优先修改配置而不是复制整套模型。修改数据字段时，同时检查 `CustomCollect3D` 的 `collect_keys`、模型 `metas` 读取和 `loss_input_convertion`；修改模型输出时，同时检查 `train.py` 构造的 loss 输入。CUDA LocalAggregator 或 spconv 相关改动必须在 H20 环境编译/测试，不能只凭本地 Python 检查判断成功。
