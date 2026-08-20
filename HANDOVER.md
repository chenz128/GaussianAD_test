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

训练必须优先使用目标实验配置目录下的 `train.sh`，不要手工拼接 `torchrun` 命令。例如，启动 Frontier v2 实验：

```bash
cd /data/chenz/GaussianAD
bash config/nuscenes_gs25600_frontier_v2/train.sh
```

每个 `train.sh` 应明确封装该实验对应的 Python 环境、`CUDA_VISIBLE_DEVICES`、GPU 数量、端口、`--py-config`、`--work-dir` 与数据集参数。启动前必须先阅读脚本顶部的实验说明，并核对这些参数与当前 Git 分支和 H20-new 的 GPU 4-7 使用限制一致。

对于尚未提供 `train.sh` 的配置，先在对应配置目录补充或复制一份同类实验的启动脚本，再按脚本启动；不要长期依赖散落在命令历史中的手工启动命令。脚本只可通过 `bash <path>/train.sh` 执行，不能用 `source`，否则会直接在当前 shell 中启动训练。

完整测试也必须通过目标实验配置目录下的 `test.sh` 启动，不要手工拼接 `test.py` 命令。例如，测试已完成的 base GT ego 实验：

```bash
cd /data/chenz/GaussianAD
bash config/nuscenes_gs25600_base_gt_ego/test.sh
```

每个 `test.sh` 应明确封装该实验对应的 Python 环境、`CUDA_VISIBLE_DEVICES`、`--py-config`、`--work-dir`、`--resume-from` checkpoint 和 `--log-name`，并将启动信息及控制台输出写入该实验的 `work-dir`。测试前先阅读脚本顶部说明，确认 checkpoint、分支、环境和 H20-new 的 GPU 4-7 使用限制；脚本同样只能通过 `bash <path>/test.sh` 执行，不能使用 `source`。对于尚未提供 `test.sh` 的配置，先在对应目录补充或复制一份同类实验的测试脚本，再按脚本启动。

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

### 端到端数据流和坐标系

一次训练迭代可以按下面的顺序理解：

```text
nuScenes PKL + 六相机图像 + 多帧 LIDAR/occupancy GT
  -> NuScenesDataset 采样当前帧、历史帧和未来帧
  -> 图像 resize/crop/flip，并整理相机内外参
  -> ResNet/FPN 提取多尺度图像特征
  -> GaussianLifter 初始化当前帧 Gaussian
  -> GaussianOccEncoder + TemporalEncoder 更新 Gaussian 属性
       (位置、尺度、旋转、不透明度、语义、动态 logit、future offset)
  -> GaussianHead
       -> LocalAggregator: 3D Gaussian -> occupancy 采样点/语义
       -> forward_flow: 当前 Gaussian 按 ego motion + offset 推到未来帧
       -> 可选 gsplat: Gaussian -> 六相机 2D 语义/深度/动静图
       -> 可选 FrontierHead: 为未来视野中新进入的区域补充 Gaussian
  -> MultiLoss 汇总 occupancy、flow、det、map、plan、render、dynamic、physics
  -> DDP 反向传播、optimizer 更新和 checkpoint
```

坐标系是交接时最容易出错的地方：当前 Gaussian 和 3D occupancy 使用 LIDAR 坐标系；`cam2ego` 是相机到 ego，反转得到 `ego2cam`，再与 `lidar2ego` 组合才得到渲染需要的 `lidar2cam`。图像增强后，渲染内参必须同步 resize/crop；水平翻转时还要同步处理图像、伪标签和内参。未来 flow 的 GT 来自未来 keyframe 自己的 occupancy 标注，不是简单复制当前帧。

## 6. 监督路线详解

### 6.1 2D Gaussian Splatting 监督

2D 监督是训练期的辅助分支，目标是让 3D Gaussian 同时满足多视角图像上的语义和深度约束：

```text
Gaussian means/scales/rotations/opacities/semantics_logits
  -> gsplat rasterization（六相机）
  -> rendered_sem + rendered_depth
  -> grounded-SAM pseudo_seg 的 CE + Metric3D pseudo_depth 的 MSE
```

- `grounded_sam_nusc` 提供六相机逐像素语义标签，0 是无效/天空；`metric_3d_nusc` 提供米制深度，通常只使用有效深度和静态区域。
- 默认伪标签缩放为 `0.44`，顶部裁剪原图 140 像素，深度上限为 40 米。修改缩放或裁剪时，必须同时修改渲染分辨率、内参和伪标签尺寸。
- 渲染语义必须使用 softplus 之前的 `semantics_logits`。alpha blending 后的非负 `semantics` 不是 CE 所需的 logits。
- pseudo label 的类别 1-16 与 Gaussian 语义通道直接对应；不要再做 `target - 1`，否则会产生整体类别错位，稀有类会被错误梯度压制。
- `GaussianHead` 在训练模式才走 2D 渲染，验证模式主动跳过。因此验证日志中的 RenderLoss 为 0 是设计行为，不代表训练分支失效。

2D 监督不替代 3D occupancy、flow、检测、地图和规划监督，而是给 Gaussian 的几何位置、透明度和语义属性增加多视角约束。当前默认配置未必开启 RenderLoss，继续实验前先检查具体 config 的 `render_config`、dataset 伪标签路径和 `loss_cfgs`。

### 6.2 动静分离

动静分离为每个 Gaussian 预测 `dynamic_logits`，并通过 `DynamicLoss` 监督 2D 渲染出的动静图。离线 dynamic GT 的约定是：0=ignore、1=static、2=dynamic；训练只在大于 0 的像素上做 BCE，并用 `pos_weight` 处理动态像素稀少问题。历史帧会屏蔽动态像素，未来帧则结合 ego motion 和 Gaussian offset 生成对应监督。

当前有两种模式：

1. **伪标签模式**：`dynamic_gt_root` 提供由 LiDAR 和 ego speed gate 生成的 2D 动静 mask，输入为 `rendered_dynamic`/`pseudo_dyn`，适合真实训练链路。
2. **GT box oracle 模式**：`use_gt_box=True` 时，直接用 Gaussian 中心是否落在速度大于 `v_thresh` 的 GT box 内作为动态标签。这是干净标签的上限/诊断实验，不应当当作部署时可用的信息。还可以用 `z_margin` 排除 box 底部误包入的地面，用 `use_gt_semantic_gate` 只保留可移动语义类。

动静分离的输出主要服务 future flow 和物理约束：静态 Gaussian 应留在环境中，动态 Gaussian 才允许按目标运动。它和 2D 语义监督是不同任务，不能仅凭 RenderLoss 判断动静分支是否有效。

### 6.3 物理约束

`PhysicsLoss` 作用在 future offset 上，当前实现包含以下先验：

- **静态零位移**：静态 Gaussian 的未来 offset 应接近 0；
- **动态运动平滑**：动态 Gaussian 的加速度应小，减少逐帧抖动；
- **速度/轨迹监督**：可用 GT box 速度外推，也可用实例真实 future trajectory 监督转弯、加减速和变道；
- **刚体一致性（可选）**：同一 GT box 内的 Gaussian offset 应保持一致，近似刚体平移。

`warmup_epoch` 用来避免训练早期动静预测不稳定时过早强约束。`vel_w` 和 `traj_w` 是两条不同的动态正向监督路线，通常用真实轨迹的 `traj_w` 替代简单恒速的 `vel_w`，不要无意中同时放大两者。`use_gt_box=True` 的物理约束同样只适用于训练/上限评估；真实推理必须依赖模型自己的 dynamic logit 和 offset。

典型组合见 `config/nuscenes_gs25600_dynamic_physics.py`：RenderLoss 可以设为 0 仅做渲染诊断，DynamicLoss 提供动静标签，PhysicsLoss 再约束 offset。调整权重时先观察 static/dynamic 占比、offset 范数和 future mIoU，避免物理项把动态目标也压成静态。

### 6.4 新区域补全（Frontier Completion）

Gaussian flow 只把当前视野内的 Gaussian 推到未来 ego frame。随着 ego 前进，occupancy 窗口前沿会出现一条“新进入区域”；如果只裁剪越界 Gaussian，不会有任何 Gaussian 覆盖这条区域，未来 occupancy 必然逐步变空。Frontier 模块的核心就是回收越界 slot，并为新区域生成固定数量的 Gaussian，使未来张量长度保持不变、DDP 形状稳定。

演进过程：

- **Frontier v1**（`GaussianHeadFrontier` + `FrontierGenerator`）：在新进入的纵向/横向条带内做确定性低差异采样，用轻量 MLP 生成位置、尺度、旋转、不透明度和语义；先验证“补齐数量”本身的收益。
- **Frontier v2**（`GaussianHeadFrontierV2` + `FrontierContextGenerator`）：不再只依赖几何先验，而是从局部 3D Gaussian 上下文和当前六相机 FPN 特征取条件，预测新 Gaussian 属性。当前实现通过 BEV 局部池化、相机投影和 image gate 融合可见图像信息。
- **Frontier v2 fix**：统一 temporal encoder 和 frontier 图像条件的 current-frame index，修复历史/当前帧顺序造成的条件错配。
- **Frontier v3**（`GaussianHeadFrontierV3`）：方向转向共享 future Gaussian bank，直接生成完整的 3 秒未来 Gaussian 集合，减少属性 base/residual 的间接依赖；配置中使用 12800 个 direct Gaussian，并对当前 Gaussian 比例设置约束。

Frontier 监督仍然复用现有 `OccupancyFlowLoss`，不是额外的独立 GT loss。评估时重点看未来各时间点 mIoU，尤其是 t+2.0 秒之后的帧；同时确认 current mIoU 没有被未来分支反向污染。`flow_grad_scale=0.0` 时未来 flow 不回传当前 Gaussian，适合先做干净对照；打开耦合后需要重新检查当前帧精度和训练稳定性。

Frontier 实验的主要风险是：补全位置在边界上必须落在 `[pc_min, pc_max)`，否则 LocalAggregator 的体素索引会越界；ego 位移方向、当前帧索引和未来帧 flow GT 必须一致；v1/v2/v3 是不同模型头，不能只替换 config 名称就认为实验可比较。推荐从 `nuscenes_gs25600_frontier_v1`、`nuscenes_gs25600_frontier_v2`、`nuscenes_gs25600_froniter_v2_fix`、`nuscenes_gs25600_v3` 的 train.sh 和配置注释开始。

## 7. 默认 loss 和配置

默认入口 `config/nuscenes_gs25600.py` 继承：

```text
config/_base_/misc.py
config/_base_/model.py
config/_base_/surroundocc.py
```

训练总 loss 由 `MultiLoss` 将 `loss_cfgs` 中启用的项目直接相加：

$$
L_{total}=\sum_i L_i
$$

`weight=0` 的项目仍会执行 forward（例如保留渲染可视化/诊断），但不写入常规 loss 日志。下面列出仓库中已经注册、可以被 `MultiLoss` 使用的全部任务级 loss；“默认”仅指 `config/nuscenes_gs25600.py`，实验配置可能覆盖或新增。

### 7.1 默认五项任务 loss

| Loss | 默认状态与监督对象 | 内部组成和关键参数 |
| --- | --- | --- |
| `OccupancyLoss` | **默认启用**；当前帧 sampled occupancy，18 通道：0-16 为语义，17 为空体。 | 主项是带 `ignore_index=255` 的 voxel CE / 可选 focal CE；默认同时启用语义尺度一致性 `sem_scal_loss`、几何占用一致性 `geo_scal_loss` 和 `Lovasz-Softmax`。可用 `multi_loss_weights` 分别控制 `loss_voxel_ce`、`sem_scal`、`geo_scal`、`lovasz`；`use_dice_loss=True` 额外加入 Dice。`balance_cls_weight` 或 `manual_class_weight` 用于类别不均衡。 |
| `OccupancyFlowLoss` | **默认启用**；未来六个 $0.5\,s$ 时间步的 `occ_flow`。每一步只在 `flow_valid_flag=True` 时计入。 | 每个未来步沿用同一套 CE/focal、语义尺度、几何尺度、Lovasz、可选 Dice；六步损失求和后乘固定系数 `0.1`，再乘每步有效标志。`dynamic_class_multiplier` 可额外放大 2,3,4,5,6,7,9,10 这些动态类。它监督的是 future occupancy，不直接监督 offset 数值。 |
| `DetectionLoss` | **默认启用**；VoxelNeXt 稀疏 3D 检测，10 类目标。 | `FocalLossSparse` 监督热力图分类，权重 `loss_weights.cls_weight`；`RegLossSparse` 回归中心、z、尺寸、朝向 sin/cos、速度等 `head_order` 属性，逐维乘 `code_weights` 后再乘 `loss_weights.loc_weight`。 |
| `MapLoss` | **默认启用**；MapTR 的 divider、pedestrian crossing、boundary 向量地图。 | Hungarian matching 后的分类 Focal loss、点集 Chamfer distance、方向余弦 loss；配置中可选 bbox L1/GIoU，且可启用 BEV/PV 辅助分割 `loss_seg` / `loss_pv_seg`。解码器中间层也会生成 `d*.loss_*` 辅助项。 |
| `PlanLoss` | **默认启用**，整体 `weight=10.0`；未来 ego 轨迹及其与地图的关系。 | 三部分相加：候选命令对应轨迹的加权 L1 `loss_plan_l1`、与 lane boundary 保持距离的 `loss_plan_bound`、与地图方向一致的 `loss_plan_dir`。无效 future step 和未选中的驾驶命令由 `ego_fut_masks`、`ego_fut_cmd` 屏蔽。 |

### 7.2 2D 伪标签与时序几何 loss

| Loss | 启用条件与监督对象 | 内部组成和关键参数 |
| --- | --- | --- |
| `RenderLoss` | **默认关闭**；`render_config` 和伪标签路径同时配置后，通过 gsplat 将 Gaussian 渲染到六相机平面。监督 Grounded-SAM 语义与 Metric3D 深度。 | 总项为 $weight(L_{sem}+L_{depth}+L_{extra}+L_{acc}+L_{conc})$。`L_sem` 是按 nuScenes 类频加权的 CE，必须使用 raw `semantics_logits`，pseudo label 1-16 直接对应通道 1-16；`L_depth` 是有效静态像素上的 Huber depth（`delta=2m`）。`extra_depth_lw` 监督历史/未来额外帧深度；`acc_lw` 约束累计 opacity；`concentration_lw` 惩罚光线深度方差，减少沿光线雾化。`sem_lw`、`depth_lw` 控制前两项。 |
| `DynamicLoss` | **默认关闭**；监督每个 Gaussian 的 `dynamic_logits`。普通模式将其 2D 渲染，与离线 `pseudo_dyn`（0 ignore、1 static、2 dynamic）比较；oracle 模式 `use_gt_box=True` 直接按 GT box 内且速度超过 `v_thresh` 的 Gaussian 监督。 | 标注区域使用 `BCEWithLogits`，`pos_weight` 处理动态像素稀少；`extra_weight` 控制历史/未来额外动态图监督。`z_margin` 把移动 box 底部地面排除，`use_gt_semantic_gate` 可再用干净 occupancy GT 的 movable 类过滤。它只负责动静分类，不直接要求 offset 正确。 |
| `PhysicsLoss` | **默认关闭**；监督 flow offset 和动静掩码的物理合理性，可使用预测 dynamic logits 或 `use_gt_box=True` 的 oracle 掩码。 | `loss_static`：静态 Gaussian offset 接近 0，权重 `static_w`；`loss_smooth`：动态 Gaussian 二阶差分小，权重 `smooth_w`；可选 `loss_rigid`：同一 GT box 内 offset 方差小，`rigid_w`；可选 `loss_vel`：匹配恒速 box velocity，`vel_w`；可选 `loss_traj`：匹配真实实例未来轨迹，`traj_w`，通常代替而非叠加 `loss_vel`。`warmup_epoch` 仅延后会主动驱动运动的项，静态/平滑约束从开始生效。 |

### 7.3 Frontier / innovation 实验 loss

| Loss | 启用条件与监督对象 | 内部组成和关键参数 |
| --- | --- | --- |
| `FlowMatchingLoss` | **默认关闭**；仅在 direct future Gaussian generator 输出 `flow_matching_loss` 时有值。 | 不自行构造标签，直接读取模型提供的 matching loss 后乘 `weight`；输出缺失时返回与计算图相连的 0，不影响 DDP 静态图。用于约束生成的 future Gaussian bank 与匹配目标。 |
| `InnovationOccupancyLoss` | **默认关闭**；仅监督 `innovation_mask=True` 的 future sampled points，配合 Frontier/direct future bank 使用。 | 对每个有效 future step 的 `pred_flow` 做 CE，按步求平均再乘 `weight`。动态类 2,3,4,5,6,7,9,10 的 CE 权重由 `dynamic_multiplier` 放大；无 innovation 点或 future step 无效时返回图连接的 0。它补强新增区域，不替代完整的 `OccupancyFlowLoss`。 |

### 7.4 配置与调参边界

- 当前默认配置只启用 7.1 的五项；`RenderLoss`、`DynamicLoss`、`PhysicsLoss`、`FlowMatchingLoss`、`InnovationOccupancyLoss` 都必须显式写进 `loss.loss_cfgs` 才会参与训练。
- `loss_input_convertion` 必须同时提供某个 loss 所需的 key。新增 loss 前先对照其 `input_dict`，否则常见结果是 loss 恒为 0、得到 `None`，或仅在 DDP 某个 rank 报错。
- `flow_grad_scale` 不是独立 loss，而是 future occupancy loss 回传到当前 Gaussian 属性的梯度系数；它会改变 `OccupancyFlowLoss`、`InnovationOccupancyLoss` 和部分 physics/Frontier 实验对当前帧的影响范围。
- 多 loss 比较先记录每个子项的数值量级和 `weight`。尤其 `PlanLoss.weight=10.0`、动态类倍增、`RenderLoss` 的 `sem_lw`/`depth_lw` 以及 `PhysicsLoss` 的 `smooth_w` 会显著改变优化主导项。

默认配置当前没有把 `RenderLoss` 加入 loss 列表，也没有开启 `render_config`。因此默认训练不依赖 gsplat 伪标签。2D 伪标签路线使用这些配置作为起点：

- `config/nuscenes_gs25600_2D.py`：基础 2D RenderLoss；
- `config/nuscenes_gs25600_render_focus.py`：降低其他目标、强调 2D render；
- `config/nuscenes_gs25600_mf_depth.py`：历史/未来多帧深度；
- `config/nuscenes_gs25600_pcgrad.py`：把 RenderLoss 作为 aux group 做梯度投影；
- `config/nuscenes_gs25600_acc.py`、`config/nuscenes_gs25600_concentrate.py`：累积透明度和深度集中性实验；
- `config/nuscenes_gs25600_depth_init.py`：用深度反投影初始化 Gaussian，不等同于 RenderLoss。

2D render 的关键参数通常是 `pseudo_label_scale=0.44`、`pseudo_label_crop_top=140`、`max_pseudo_depth=40.0`。渲染使用 `semantics_logits`，而不是 softplus 后的 `semantics`；否则 alpha blending 后不能直接用于 CE。pseudo label 的类别标签 1-16 与 Gaussian 语义通道直接对应，不要再减 1。

## 8. 验证与结果查看

完整验证应执行目标实验配置目录下的 `test.sh`；脚本会固定对应的环境、GPU、checkpoint 和日志名。例如：

```bash
cd /data/chenz/GaussianAD
bash config/nuscenes_gs25600_base_gt_ego/test.sh
```

`test.py` 自动优先读取相同 `work-dir/latest.pth`，并汇总：当前 occupancy mIoU、未来 0.5-3.0 秒各时间点 mIoU、平均未来 mIoU，以及规划指标。训练过程中的 epoch 验证只走 3D occupancy 路径；2D render 在 eval 模式会跳过以节约时间，所以 RenderLoss 在验证时为 0 属于预期行为。

常用日志检查：

```bash
grep -E "successfully resumed|mIoU|Future|Traceback|RuntimeError" out/<work-dir>/*.log | tail -50
```

## 9. 性能和实验注意事项

- `with_cp=True` 会节省显存但增加重计算；当前 `faster` 分支已验证 `with_cp=False` 只有约 2-3% 实测收益，主要瓶颈仍是 spconv SubMConv backward 的显存分配。
- `history_no_grad=True` 只让历史帧 backbone/encoder 不回传梯度，temporal encoder 仍接收完整时序；这是速度/显存实验开关，不要和数据帧数混淆。
- channels-last、提前转换 `lidar2global`、bool mask、DDP bucket 和 allocator 参数的收益已做过实测，整体收益很小，不建议优先继续改这些上层路径。
- `flow_grad_scale=0.0` 表示 future flow loss 不回传到当前帧 Gaussian，只训练 offset；改回 1.0 会恢复耦合行为。
- `use_plan_ego=True` 会让 future flow 使用 planner 预测的 ego 轨迹；可配 `plan_ego_warmup_epochs`，否则 planner 初期不稳定可能污染 flow 监督。
- `with_empty=True` 时会额外加入空体 Gaussian。future flow 的空体处理由 `flow_include_empty` 控制，修改前先确认空体监督是否仍有贡献。

## 10. 相对 Baseline 改动中的重大 Bug 复盘

本节记录的是新增 2D 监督、动静/物理、future flow 与 Frontier 后实际影响训练或评估结论的高风险问题，而不是一般运行环境问题。继续修改相关路径前，先确认这些修复没有被配置继承、脚本复制或可视化辅助代码意外回退。

| 问题 | 现象与根因 | 当前修复与回归检查 |
| --- | --- | --- |
| **时序检索帧索引错位** | Frontier v2 默认 `current_frame_index=-1`，会选取多帧输入的最后一帧；而当前时序数据约定当前帧在 index 0。temporal encoder、Frontier 图像条件和当前 Gaussian bank 若引用不同帧，生成器看到的图像/几何上下文就错位，且裁剪后的 bank 可能被误当作当前帧。 | `nuscenes_gs25600_froniter_v2_fix` 将 temporal encoder 和 head 的 `current_frame_index` 都设为 `0`，并以 `min_current_gaussian_ratio=0.99` 断言当前 bank 未被错误裁剪。新增时序模块必须统一这三个引用点：temporal encoder、head、future generator。 |
| **future Gaussian 漏掉 empty 语义** | `forward_flow` 原来按有效 real Gaussian 的索引筛选 `gs`，但没有重新附加最后一个 empty Gaussian；未来渲染没有 empty label 的贡献者。由于未来 GT 中空体占比极高，模型会通过病态梯度压低当前高斯表达，历史上出现 future empty IoU 为 0、FutAvg 接近 0 的情况。 | `flow_include_empty=True` 时，future 渲染显式追加原始位置的 `empty_mean` 和 `gs` 的最后一个 empty 属性。Frontier v1/v2/v3 都保留同一逻辑。修改筛选/拼接代码后检查 `num_render_gaussians` 是否等于 real count 加 1。 |
| **测试/可视化中的未来位移未累计** | `ego_fut_trajs` 表示每个时间步的增量位移；如果测试、可视化或未来帧导出把第 $k$ 步增量直接当作从当前帧到第 $k$ 帧的位移，远期帧的 ego 补偿不足，显示和评估会与训练路径不一致。 | future 坐标构造必须使用 `ego_fut_trajs.cumsum(dim=1)`；SE(3) 路径则使用 `inv(future_lidar2global) @ current_lidar2global`。检查 `gaussian_head.py`、`visualize.py` 及新增 test/导出脚本：第 2-6 步不得使用未累计的单步 ego 位移。 |
| **2D 渲染外参用了 `ego2cam` 而不是 `lidar2cam`** | Gaussian 在 LIDAR 坐标系，而 `ego2cam` 忽略了 LIDAR 到 ego 的固定平移/旋转，导致所有相机投影产生系统性偏差，典型表现为 `pred_depth_mean` 接近 0 或渲染结构整体错位。 | dataset 在 pipeline 前保存原始 `cam2ego` 和内参，并计算 $T_{lidar\rightarrow cam}=T_{ego\rightarrow cam}T_{lidar\rightarrow ego}$。改变坐标系或增强流程后，应先检查六相机 render 可视化和 `[RenderLoss Diag]`。 |
| **2D 语义渲染用了激活后特征，且类别索引减一** | 将 softplus 后的 `semantics` 做 alpha blending 后不再是 logits，CE 梯度弱；历史代码还曾对 pseudo label 做 `target - 1`，使 channel 1-16 全体错位，稀有类如 bicycle 被错误梯度压制。 | refinement 保存 raw `semantics_logits`，gsplat 渲染该 raw logits；RenderLoss 直接以 pseudo label 1-16 作为 CE target，不再减一。任何新 render loss 都要保持与 OccupancyLoss 相同的类别通道映射。 |
| **`BaseLoss.loss_func` 实例属性遮蔽子类方法** | `BaseLoss.__init__` 在实例上设置 `self.loss_func`，会遮蔽 RenderLoss、DynamicLoss、PhysicsLoss 中同名的类方法，造成 `NoneType is not callable` 或错误地走默认空 loss。 | 三个新增 loss 的构造函数都显式删除该实例属性。新增继承 `BaseLoss` 且自定义 `loss_func` 的 loss 时必须重复这一处理，或改用不冲突的方法名。 |
| **future OccFlowLoss 反向污染当前帧 Gaussian** | 即使 offset head 已 `decouple_offset`，future loss 仍能通过 `means`、`semantics`、`opacity`、`scales` 直接回传 encoder。oracle/物理约束把 offset 钉住后，无法解释的新区域或出界残差会迫使当前帧语义让步，出现 current mIoU 长期低于 baseline 的现象。 | `flow_grad_scale` 采用直通混合：前向 future occupancy 不变，只缩放 future 到当前 Gaussian 的梯度。`0.0` 时只训练 offset 等未来分支；比较新 motion/Frontier 方法时，应先明确该值，不能混淆“未来涨点”与“牺牲当前帧”。 |
| **Frontier / SE(3) 下边界高斯被错误丢弃** | 纯中心点或整数 voxel mask 在 ego 转弯、完整 SE(3) 对齐时会将 footprint 仍覆盖 future ROI 的边缘 Gaussian 丢掉，产生历史区域空洞；坐标恰好落在 `pc_max` 还会触发 LocalAggregator 越界。 | `v3_se3` 使用 `strict_range_mask=True` 与 `range_mask_sigma=3.0`，按 scale margin 判断 footprint；新生成点必须 clamp 到 $[pc_{min}, pc_{max})$ 而非上边界。修改 range mask 后应检查 retained/generated Gaussian 数和转弯样本可视化。 |

除上述模型逻辑外，环境仍需直接调用 `/data/chenz/conda_env/<env>/bin/python` 或 `torchrun`，不能 `conda activate`；DDP 出现 hang 时先核对 `find_unused_parameters`、`static_graph` 与实际条件分支是否一致，但这类问题不属于相对 baseline 的算法改动。

## 11. 接手后的推荐顺序

1. 先在远端确认 `splatting` 分支、环境、数据 PKL 和 occupancy 文件存在。
2. 用默认配置单卡跑一个短检查，确认模型能构建、dataset 能取样、checkpoint 能保存。
3. 用已有 `work-dir` 做完整验证，记录当前 occupancy、future flow、规划指标作为接手基线。
4. 需要继续某个实验时，先阅读对应配置文件顶部说明，并核对它继承的基础配置；不要只凭目录名判断实验内容。
5. 每次实验使用独立 `work-dir`，保存实际配置、日志和 checkpoint；代码改动通过本地 commit/push 后再让远端 pull。

### 11.1 代码维护边界

优先修改配置而不是复制整套模型。修改数据字段时，同时检查 `CustomCollect3D` 的 `collect_keys`、模型 `metas` 读取和 `loss_input_convertion`；修改模型输出时，同时检查 `train.py` 构造的 loss 输入。CUDA LocalAggregator 或 spconv 相关改动必须在 H20 环境编译/测试，不能只凭本地 Python 检查判断成功。

## 12. 后续实验开关速查

后续实验新增了不少行为开关。除非在做单变量对照，不要同时改多个组；每次实验应在对应配置文件顶部记录启用值和目的。下面的“关闭/默认”表示模块代码的默认行为，实际配置会继承和覆盖。

### 12.0 所有开关的实际位置与修改方法

**不要直接改作为基线的配置。** 新实验先新建一个配置，例如 `config/nuscenes_gs25600_my_exp.py`，选择最接近的实验作为 `_base_`，然后只写需覆盖的字段。MMEngine 会把以下嵌套 `dict` 深合并；同名字段即覆盖父配置值。训练时把目标配置路径写入该实验目录的 `train.sh` 的 `--py-config`，并使用新的 `--work-dir`。

```python
# config/nuscenes_gs25600_my_exp.py
_base_ = ['./nuscenes_gs25600_2D.py']  # 按实验目标替换为最接近的父配置

# 只写本实验真正改变的项；不要复制整份父配置。
train_dataset_config = dict(
  pseudo_label_scale=0.44,
)
model = dict(
  head=dict(
    flow_grad_scale=0.0,
  ),
)
```

下面的路径均从该子配置文件的顶层开始；`loss_cfgs` 是列表，**按 `type` 找到对应的 `dict(...)` 后修改字段，不要按列表序号假设位置**。下表覆盖本节所有开关的精确落点。

| 组别 | 字段 | 精确位置 | 应如何修改 |
| --- | --- | --- | --- |
| 2D 数据 | `metric3d_root`、`grounded_sam_root`、`pseudo_label_scale`、`pseudo_label_crop_top`、`max_pseudo_depth` | `train_dataset_config`。参考 `config/nuscenes_gs25600_2D.py`。若要验证也读取伪标签，再同步写入 `val_dataset_config`。 | 在子配置新增 `train_dataset_config = dict(...)`。关闭加载时将两个 root 都设为 `None`；改 scale/crop 时同时改下一行的 `render_h/render_w`，并用一个 batch 检查 pseudo label 与 render 输出高宽相同。 |
| 2D 初始化 | `depth_init_root` | `train_dataset_config.depth_init_root`，通常也要写 `val_dataset_config.depth_init_root`。参考 `config/nuscenes_gs25600_depth_init.py`。 | 两个 dataset dict 指向同一深度目录；关闭时两个都设 `None`。它只控制初始化数据，不能代替 RenderLoss。 |
| 2D 渲染器 | `render_config`、`detach_shape` | `model.head.render_config`。基础 render 配置见 `config/nuscenes_gs25600_2D.py`；`detach_shape=True` 示例见 `config/nuscenes_gs25600_concentrate_new.py`。 | 需要渲染时保留/新增 `render_config=dict(render_h=..., render_w=..., detach_shape=...)`；完全关渲染时写 `model=dict(head=dict(render_config=None))`。`detach_shape` 只放在此处，不能写进 RenderLoss。 |
| 2D 多帧数据 | `num_hist_depth_frames`、`num_fut_depth_frames` | `train_dataset_config`。参考 `config/nuscenes_gs25600_mf_depth.py`。 | `0` 关闭；例如 `2` 表示取两帧。启用后必须确保 PKL 有对应历史/未来相机和深度文件。 |
| RenderLoss | `weight`、`sem_lw`、`depth_lw`、`extra_depth_lw`、`acc_lw`、`concentration_lw` | `loss = dict(type='MultiLoss', loss_cfgs=[dict(type='RenderLoss', ...)])`。基础样式见 `config/nuscenes_gs25600_2D.py`；acc/concentration 示例见 `config/nuscenes_gs25600_acc.py`、`config/nuscenes_gs25600_concentrate.py`。 | 在 `loss_cfgs` 中找到 `type='RenderLoss'`；`weight=0.0` 保留渲染诊断但不反传，删除整条 RenderLoss 才是不构建该 loss。`extra_depth_lw` 仅多帧数大于 0 时有效；`acc_lw=0.0`、`concentration_lw=0.0` 关闭各自附加项。 |
| 动静数据 | `dynamic_gt_root`、`num_hist_dyn_frames`、`num_fut_dyn_frames` | `train_dataset_config`。参考 `config/nuscenes_gs25600_dynamic_physics.py`。 | 真实 2D 动静监督写 `dynamic_gt_root='data/dynamic_gt_nusc'`；关闭读盘写 `None`。多帧数设 `0` 可关闭对应额外帧。 |
| DynamicLoss | `weight`、`pos_weight`、`extra_weight` | `loss.loss_cfgs` 内 `dict(type='DynamicLoss', ...)`。参考 `config/nuscenes_gs25600_dynamic_physics.py`。 | 修改该条字段；`weight=0.0` 关闭其梯度。`extra_weight` 仅 dataset 已提供额外动静帧时有效。 |
| 动静 oracle | `use_gt_box`、`v_thresh`、`z_margin`、`use_gt_semantic_gate`、`movable_classes`、`sem_gate_max_dist` | **同时**位于 `loss_cfgs` 的 `DynamicLoss` 和 `PhysicsLoss` 两条配置。完整示例见 `config/nuscenes_gs25600_gtbox_oracle_v5.py`。 | 要做 oracle，两个 loss 条目都写同一组值；要回到真实伪标签，两处都设 `use_gt_box=False`，并启用 `dynamic_gt_root`。不要只改其中一个，否则 DynamicLoss 与 PhysicsLoss 的动静定义不一致。 |
| motion head | `decouple_dynamic`、`decouple_offset`、`offset_grad_scale` | `model.encoder.refine_layer`。`decouple_offset` 示例见 `config/nuscenes_gs25600_gtbox_oracle_v6.py`，`decouple_dynamic` 见 v10。 | 例如 `model=dict(encoder=dict(refine_layer=dict(decouple_offset=True)))`。此处控制 head 读取的特征梯度；它不等于下面的 `flow_grad_scale`。 |
| PhysicsLoss | `weight`、`static_w`、`smooth_w`、`vel_w`、`traj_w`、`traj_smooth_beta`、`rigid_w`、`warmup_epoch` | `loss.loss_cfgs` 内 `dict(type='PhysicsLoss', ...)`。基础配置见 `config/nuscenes_gs25600_dynamic_physics.py`，轨迹/oracle 配置见 `config/nuscenes_gs25600_gtbox_oracle_v7.py`。 | `weight=0.0` 关闭物理 loss；`vel_w` 与 `traj_w` 通常一次只保留一个正值。修改 oracle 字段时遵循上一行，和 DynamicLoss 同步。 |
| future flow | `flow_grad_scale`、`flow_include_empty`、`use_plan_ego`、`plan_ego_warmup_epochs`、`plan_ego_detach`、`dynamic_class_multiplier` | `model.head`。基础实现参数见 `model/head/gaussian_head.py`；`flow_grad_scale` 覆盖例见 `config/nuscenes_gs25600_gtbox_oracle_v12.py`，planner 例见 `config/nuscenes_gs25600_base_plan_new/nuscenes_gs25600_base_plan_new.py`。 | 写 `model=dict(head=dict(flow_grad_scale=0.0))` 即可切断 future 到当前 Gaussian 的梯度，前向不变。除复现旧 bug 外保持 `flow_include_empty=True`。启用 `use_plan_ego=True` 时同时给 warmup，必要时以 `plan_ego_detach=True` 做不反传对照。 |
| Frontier/v3 head | `head.type`、`target_num_gaussians`、`current_frame_index`、`future_pose_mode`、`strict_range_mask`、`range_mask_sigma`、`center_only_mask` | `model.head`；另一个 `current_frame_index` 位于 `model.temporal_encoder`。v3 基础见 `config/nuscenes_gs25600_v3/nuscenes_gs25600_v3.py`，SE(3) 覆盖见 `config/nuscenes_gs25600_v3_se3/nuscenes_gs25600_v3_se3.py`。 | 更换 `head.type` 时从对应实验配置继承，不要在标准 head 上零散拼字段。每次均把 `model.temporal_encoder.current_frame_index`、`model.head.current_frame_index` 和生成器的 index 一并设为 `0`。SE(3) 实验成组写 `future_pose_mode='se3'`、`strict_range_mask=True`、`range_mask_sigma=3.0`。 |
| v3 direct generator | `num_gaussians`、`current_frame_index`、`front_fraction`、`min_band`、`responsibility_size`、`initial_opacity`、`detach_context` | `model.head.direct_generator`。完整模板见 `config/nuscenes_gs25600_v3/nuscenes_gs25600_v3.py`。 | 以 `model=dict(head=dict(direct_generator=dict(...)))` 覆盖；该层的 `num_gaussians` 对应文档中的 `direct_generator.num_gaussians`。不要误写成 `target_num_gaussians`，两者属于不同 head。 |
| Innovation Flow generator | 同名生成参数（含 `num_gaussians`、`current_frame_index`、`front_fraction`、`min_band`、`responsibility_size`、`initial_opacity`、`detach_context`） | `model.head.innovation_flow`，不是 `direct_generator`。参考 `config/nuscenes_gs25600_innovation_flow_new/nuscenes_gs25600_innovation_flow_new.py`。 | 只在 `GaussianHeadInnovationFlow` 实验中修改此字典；与 v3 的 `direct_generator` 不能交叉使用。 |
| 效率 | `history_no_grad` | `model.history_no_grad`。基础配置与 `config/nuscenes_gs25600_render_focus.py` 都有实例。 | `model=dict(history_no_grad=True)`；它是时序梯度策略，必须作为独立对照，不和模型结构变量混改。 |
| 效率 | `with_cp` | `model.img_backbone.with_cp`。 | 写 `model=dict(img_backbone=dict(with_cp=False))` 关闭 checkpointing；先确认显存充足。 |
| 精度/DDP | `backbone_fp16`、`find_unused_parameters`、`static_graph` | 三者都是配置**顶层**变量，不在 `model` 内。基础位置见 `config/nuscenes_gs25600.py`。 | 直接写 `backbone_fp16 = False`、`find_unused_parameters = True` 或 `static_graph = False`。只有每轮参与反向的参数集合固定时才保留 `static_graph=True`；新增条件分支后先设 `False` 排除 DDP 图错误。 |

最小修改流程：复制最近的实验配置为新文件或以它为 `_base_`；只添加上表指定层级的覆盖；用 `grep -n "<字段>" <新配置>` 确认没有写到错误层；再修改该实验目录 `train.sh` 中的 `--py-config` 和新的 `--work-dir`。修改 shell 脚本后用 `bash -n <path>/train.sh` 检查语法，不能 `source` 脚本。

### 12.1 伪标签与 2D 渲染监督

| 开关/参数 | 作用 | 常用值或注意事项 |
| --- | --- | --- |
| `metric3d_root` + `grounded_sam_root` | 两者同时非空才启用伪标签读取和 2D 渲染输入。 | 任一为空即 `use_pseudo_label=False`。 |
| `RenderLoss.weight` | 总开关：0 时仍可渲染并诊断，但 RenderLoss 不回传梯度。 | 正常训练设为大于 0。 |
| `sem_lw` / `depth_lw` | 2D 语义 CE 和深度 Huber 的内部权重。 | 语义用 raw `semantics_logits`；深度只在有效像素计算。 |
| `pseudo_label_scale` | 伪标签和渲染内参的缩放比例。 | 常用 `0.44`；改变时要同步检查渲染高宽。 |
| `pseudo_label_crop_top` | 裁掉伪标签顶部，并同步修正内参 `cy`。 | 当前常用 `140`。 |
| `max_pseudo_depth` | 超过该距离的深度和语义均置为无效。 | 常用 `40.0` 米。 |
| `detach_shape` | 是否阻断 2D render 对尺度、旋转和 opacity 的梯度。 | `False`：2D 深度可塑形；`True`：仅保留语义/位置相关梯度。 |
| `num_hist_depth_frames` / `num_fut_depth_frames` | 额外渲染历史/未来帧深度，强化时序几何约束。 | `0` 关闭；历史帧动态区域会被屏蔽。 |
| `extra_depth_lw` | 多帧深度 loss 的权重。 | 仅在上两项大于 0 时生效。 |
| `acc_lw` | 对累计不透明度 `rendered_acc` 的弱监督，抑制天空漂浮物、鼓励前景覆盖。 | `0` 关闭。 |
| `concentration_lw` | 约束每条光线的深度方差，减少沿深度方向的雾化/拉伸。 | `0` 关闭；启用时会增加额外渲染开销。 |
| `depth_init_root` | 启用深度反投影的 Gaussian 初始化。 | 与 RenderLoss 不等价；需要独立验证初始化分布。 |

### 12.2 动静分离

| 开关/参数 | 作用 | 常用值或注意事项 |
| --- | --- | --- |
| `dynamic_gt_root` | 提供离线 `pseudo_dyn` 动静标签，启用 2D DynamicLoss 真实训练链路。 | 路径为空则 dataset 不加载该 GT。 |
| `DynamicLoss.weight` | DynamicLoss 的总权重开关。 | `0` 可关闭动静监督。 |
| `pos_weight` | 动态正样本 BCE 权重，解决动态像素稀少。 | 过大容易让动态预测过度扩张。 |
| `extra_weight` | 历史/未来额外动静监督权重。 | 多帧动静输入不存在时该项自动跳过。 |
| `use_gt_box` | 使用 GT box 内、速度超过阈值的 Gaussian 作为动静标签。 | 仅训练期 oracle/上限诊断，不能当作真实推理方案。 |
| `v_thresh` | 认定 GT box 为运动目标的速度阈值。 | 影响 dynamic/static 的标签比例。 |
| `z_margin` | 将运动 box 底部薄层强制视为静态，避免地面被 box 误标为动态。 | `0` 关闭该地面门控。 |
| `use_gt_semantic_gate` | 用最近 occupancy GT 的可移动语义再过滤 oracle 动态 Gaussian。 | 需同时配置 `movable_classes`。 |
| `sem_gate_max_dist` | 最近 GT 点距离超过阈值即不信任该语义门控。 | 过小会丢弃过多动态 Gaussian。 |

### 12.3 PhysicsLoss 与 motion head 解耦

| 开关/参数 | 作用 | 常用值或注意事项 |
| --- | --- | --- |
| `PhysicsLoss.weight` | 物理约束总权重。 | `0` 关闭 PhysicsLoss。 |
| `static_w` | 静态 Gaussian 的 offset 接近 0。 | 太大可能压制被误分的动态目标。 |
| `smooth_w` | 动态 Gaussian 的加速度/时间平滑约束。 | 过大可能抹平急刹、转弯。 |
| `vel_w` | 用当前 GT box 速度做恒速位移监督。 | 与 `traj_w` 通常二选一，避免重复强化。 |
| `traj_w` / `traj_smooth_beta` | 用实例真实未来轨迹监督 offset，并调节 SmoothL1 的线性区间。 | 推荐优先使用轨迹监督以覆盖转弯和加减速。 |
| `rigid_w` | 同一 GT box 内 Gaussian 位移的一致性约束。 | `0` 关闭。 |
| `warmup_epoch` | 前若干 epoch 不启用或弱化 PhysicsLoss。 | 防止早期动静标签不稳定时误导 motion head。 |
| `decouple_dynamic` | 动静 head 使用 detach 后的 Gaussian 特征。 | 防止 DynamicLoss 反向重塑主 encoder。 |
| `decouple_offset` | offset head 使用独立/解耦特征路径。 | 与 `offset_grad_scale` 等实验项配合时必须检查梯度路径。 |

### 12.4 Future flow、规划耦合与空体

| 开关/参数 | 作用 | 常用值或注意事项 |
| --- | --- | --- |
| `flow_grad_scale` | future OccFlowLoss 回传到当前 Gaussian 属性的比例，前向结果不变。 | `1.0` 完全耦合；`0.0` 保护当前帧，只训练 offset 等未来分支。 |
| `flow_include_empty` | future flow 渲染时是否重新加入 empty Gaussian。 | 改动前先确认空体/自由空间监督是否仍完整。 |
| `use_plan_ego` | future flow 使用 planner 预测的 ego 轨迹，而非 GT ego 位移。 | 将 flow 与 planner 耦合，训练初期风险更高。 |
| `plan_ego_warmup_epochs` | 启用 planner ego 前，先使用 GT ego 的 epoch 数。 | 配合 `use_plan_ego=True` 使用。 |
| `dynamic_class_multiplier` | 增大 future OccupancyFlowLoss 中动态类别权重。 | v3/v3_se3 常用 `3.0`；过大可能损害静态背景。 |

### 12.5 Frontier 新区域补全与 v3_se3

| 开关/参数 | 作用 | 常用值或注意事项 |
| --- | --- | --- |
| `head.type` | 切换标准 flow、Frontier v1/v2/v3 等未来生成路径。 | 不同 head 不是可随意混用的小开关，应使用对应实验配置。 |
| `target_num_gaussians` | 未来渲染目标 Gaussian 数量。 | v3 会检查当前 bank 数量，避免时序帧顺序错误。 |
| `current_frame_index` | 指定多帧输入中哪一帧是当前参考帧。 | temporal encoder、head 和 generator 必须一致；当前实验用 `0`。 |
| `future_pose_mode` | future Gaussian 对齐使用纯平移或完整 SE(3)。 | `translation` / `se3`；`v3_se3` 使用 `se3`。 |
| `strict_range_mask` | 用真实连续坐标判断 Gaussian 是否与未来 ROI 相交。 | `True` 时可配合尺度边距；适合 SE(3) 转弯场景。 |
| `range_mask_sigma` | 范围掩码按 Gaussian 最大尺度扩张的倍数。 | `0` 无尺度边距；`v3_se3` 用 `3.0`。 |
| `center_only_mask` | 只按 Gaussian 中心是否在 ROI 内判断保留。 | 是与严格 footprint 掩码的对照项。 |
| `direct_generator.num_gaussians` | v3 共享 future bank 的直接生成数量。 | 当前 v3 配置为 `12800`。 |
| `front_fraction` / `min_band` / `responsibility_size` | 控制新进入条带的 query 分配、最小条带宽度和每个 query 的负责区域。 | 改动会直接改变新区域覆盖密度。 |
| `detach_context` | future generator 是否阻断时序/图像上下文向主干回传梯度。 | `True` 隔离 future bank 训练；`False` 允许其共同塑形主干。 |
| `initial_opacity` | 新生成 Gaussian 的初始不透明度。 | 过高会使新区域早期遮挡过强，过低则难以获得深度/occupancy 梯度。 |

### 12.6 训练效率、显存与 DDP

| 开关/参数 | 作用 | 常用值或注意事项 |
| --- | --- | --- |
| `history_no_grad` | 历史帧 backbone/encoder 只前向、不存激活、不回传梯度；temporal encoder 仍接收全部帧。 | 可省显存和部分反向时间，但要作为独立精度对照。 |
| `with_cp` | 对部分网络启用 gradient checkpointing。 | 省显存但增加重计算；显存够时可设 `False`。 |
| `backbone_fp16` | backbone 使用 FP16。 | 需观察数值稳定性和显存收益。 |
| `find_unused_parameters` | DDP 是否查找未参与反向的参数。 | 动态 head、冻结模块或条件分支导致参数不稳定参与时可能需要 `True`。 |
| `static_graph` | 告知 DDP 每个 iteration 的反向图固定。 | 仅在分支和参与参数集合确实稳定时设 `True`。 |

### 12.7 实验前最小核对清单

1. 先确认 `head.type`、`current_frame_index`、`flow_grad_scale`，这三项决定未来分支的基本语义和梯度边界。
2. 启用伪标签时，同时核对伪标签根目录、`pseudo_label_scale`、裁剪、渲染内参与 `RenderLoss.weight`；只配数据而未注册 RenderLoss 不会产生监督。
3. 启用动静/物理时，明确使用真实 2D dynamic GT 还是 `use_gt_box=True` oracle；两者的实验结论不能直接横比。
4. 启用 Frontier v3_se3 时，必须成组检查 `future_pose_mode='se3'`、`strict_range_mask=True`、`range_mask_sigma`、`current_frame_index=0`。
5. 训练加速项与模型创新项分开对照；每次只改一组，并用对应目录的 `train.sh` 固化启动参数。
