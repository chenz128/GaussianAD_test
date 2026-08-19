# GaussianAD 项目文件结构说明

本文档用于快速了解 GaussianAD 仓库中各个目录和主要文件的职责。它只描述模块边界和整体作用，具体训练命令、实验分支、数据准备和故障排查请参考 [HANDOVER.md](HANDOVER.md)。

## 1. 项目整体结构

```text
GaussianAD/
├── train.py                  # 训练入口
├── test.py                   # 完整验证入口
├── eval.py                   # 轻量评估入口
├── vis.py                    # 结果可视化入口
├── visualize.py              # 预测结果可视化脚本
├── config/                   # 模型、数据和实验配置
├── dataset/                  # nuScenes 数据集和数据增强
├── model/                    # GaussianAD 模型主体
├── loss/                     # 各类训练损失
├── tools/                    # 数据处理、可视化、调试和分析工具
├── docs/                     # 项目补充文档
├── data/                     # 数据集、标注和初始化文件
├── ckpts/                    # 预训练模型和实验 checkpoint
└── out/                      # 训练日志、checkpoint 和可视化输出
```

## 2. 顶层入口文件

| 文件 | 作用 |
| --- | --- |
| `train.py` | 初始化 DDP、构建 dataset/model/loss、执行训练循环、保存 checkpoint。 |
| `test.py` | 加载 checkpoint，执行完整验证，统计 occupancy、future flow、规划等指标。 |
| `eval.py` | 提供相对轻量的评估入口，主要用于快速检查 occupancy 结果。 |
| `vis.py` | 可视化模型输出或评估结果。 |
| `visualize.py` | 将 occupancy、Gaussian 或渲染结果整理成图片/动画等可视化结果。 |
| `train.sh` | 默认训练启动脚本，包含环境、GPU 和配置参数。 |
| `ddp_train.sh` | 多 GPU/DDP 训练启动脚本模板。 |
| `env_gaussianad.sh` | 环境变量和运行环境相关设置。 |
| `HANDOVER.md` | 面向接手人的完整工作流、方法说明和实验注意事项。 |
| `claude.md` | 伪标签监督、2D Splatting、动静分离、物理约束和 Frontier 实验记录。 |

## 3. `config/`：配置和实验管理

配置文件集中管理模型结构、数据路径、输入帧数、loss、优化器、冻结模块和实验开关。通常从一个基础配置继承，再通过局部字段覆盖实验变量。

### 基础配置

- `config/_base_/misc.py`：通用训练参数、类别定义和基础运行设置。
- `config/_base_/model.py`：默认模型结构、Gaussian 参数和各个 head 的基础配置。
- `config/_base_/surroundocc.py`：occupancy、地图、检测、规划等数据和任务配置。

### 常用实验配置

- `config/nuscenes_gs25600.py`：默认 25600 Gaussian 的主配置。
- `config/nuscenes_gs25600_2D.py`：2D Gaussian Splatting 语义/深度监督实验。
- `config/nuscenes_gs25600_dynamic_physics.py`：动静分离和 PhysicsLoss 实验。
- `config/nuscenes_gs25600_frontier_v1/`：固定槽位的 Frontier 新区域补全。
- `config/nuscenes_gs25600_frontier_v2/`：融合 3D 局部上下文和图像特征的 Frontier 补全。
- `config/nuscenes_gs25600_froniter_v2_fix/`：修复 Frontier v2 当前帧索引后的实验配置，目录名中的 `froniter` 是历史拼写，使用时不要自行改名。
- `config/nuscenes_gs25600_v3/`：共享 future Gaussian bank 的 Frontier v3 实验。
- `config/nuscenes_gs25600_gtbox_oracle_*/`：使用 GT box oracle 进行动静或运动上限分析。
- `config/nuscenes_gs25600_base_*/`：不同基础模型、规划、flow 或解耦方案的配置集合。

配置目录中的 `train.sh` 通常只负责封装该实验的启动命令；真正的模型和 loss 定义仍在 Python 配置文件及 `model/`、`loss/` 中。

## 4. `dataset/`：数据读取和预处理

`dataset/` 负责将 nuScenes PKL、图像、occupancy、检测框、地图、规划轨迹、flow 和伪标签组织成模型输入。

- `dataset.py`：核心 `NuScenesDataset`，负责多帧采样、读取标注、组织相机参数、加载 occupancy 和伪标签。
- `transform_3d.py`：图像 resize、crop、flip、格式转换和数据 pipeline。
- `data_container.py`：batch 中不同类型数据的封装和整理。
- `sampler.py`：训练/验证数据采样器。
- `nuscenes_box.py`：nuScenes 3D box 的转换和处理。
- `vad_custom_nuscenes_eval.py`：规划和 VAD 相关评估逻辑。
- `mean_ap.py`、`tpfp.py`、`tpfp_chamfer.py`：检测或地图任务的匹配、TP/FP 和评价辅助函数。
- `metric_stp3.py`：轨迹/规划相关指标计算。
- `utils.py`：数据集通用工具函数。

数据转换脚本主要位于 `tools/data/`，不会在 `dataset/` 中直接生成新数据。修改 dataset 时需要特别留意 pipeline 的原地修改行为，以及图像增强后相机内参和伪标签是否仍然对齐。

## 5. `model/`：模型主体

模型整体由图像 backbone 提取特征，再通过 Gaussian 表示完成 3D occupancy、时序 flow、检测、地图和规划任务。

### 主要子目录

- `model/segmentor/`：总模型编排，通常从这里进入 backbone、Gaussian encoder、temporal encoder、head、地图和规划模块。
- `model/backbones_2d/`：2D 图像 backbone，例如 ResNet/DCN 等。
- `model/neck/`：FPN 等多尺度图像特征融合模块。
- `model/encoder/`：Gaussian 特征编码和时序融合。
  - `gaussian_encoder/`：Gaussian 的位置、尺度、旋转、不透明度、语义和动态属性 refinement。
  - `temporal_encoder/`：历史帧与当前帧 Gaussian 的时序信息融合。
- `model/lifter/`：从图像或初始化特征生成 Gaussian 表示。
- `model/head/`：occupancy、flow、2D render、动静预测和 Frontier 补全等输出头。
- `model/planner/`：ego future trajectory 规划模块。
- `model/dense_heads/`、`model/roi_heads/`、`model/detectors/`、`model/coders/`、`model/assigners/`：检测任务使用的 head、box coder、匹配器和检测组件。
- `model/backbones_3d/`、`model/ops/`：3D 稀疏网络和 CUDA/自定义算子。
- `model/model_utils/`、`model/utils/`：模型内部的通用工具、坐标变换和辅助实现。

### `model/head/` 中的关键文件

- `gaussian_head.py`：标准 Gaussian head，负责 LocalAggregator 3D occupancy、future flow 和可选 2D 渲染输出。
- `gaussian_rasterizer.py`：封装 gsplat，将 3D Gaussian 渲染成 2D 语义/深度图。
- `gaussian_head_frontier.py`：Frontier v1，回收未来分支中越界的 Gaussian slot 并补充新区域。
- `gaussian_head_frontier_v2.py`：Frontier v2，加入局部 Gaussian 和图像上下文。
- `gaussian_head_frontier_v3.py`：Frontier v3，生成共享的 future Gaussian bank。
- `frontier_generator.py`：基于几何条带采样生成补全 Gaussian 属性。
- `frontier_context_generator.py`：融合 BEV 局部 Gaussian 上下文和相机图像特征。
- `future_gaussian_direct_generator.py`：直接生成未来 Gaussian 的组件，主要服务 v3 方向。
- `innovation_flow_generator.py`、`gaussian_head_innovation_flow.py`：创新 flow/运动建模实验组件。
- `localagg/`：3D Gaussian 到 occupancy 采样点的聚合实现，包含 Python 封装和 CUDA kernel。
- `spconv_backbone_voxelnext.py`：检测分支使用的稀疏 3D backbone。

## 6. `loss/`：训练监督

所有 loss 通常由 `MultiLoss` 统一调用，模型输出通过配置中的 `loss_input_convertion` 映射到各个 loss。

- `multi_loss.py`：组合多个任务 loss，可按 group 使用 PCGrad 等梯度处理策略。
- `base_loss.py`：loss 基类和公共接口。
- `occupancy_loss.py`：当前帧 3D occupancy 语义监督。
- `occupancy_loss_flow.py`：未来 occupancy flow 监督。
- `detection_loss.py`：3D 检测分类和回归监督。
- `map_loss.py`：divider、ped crossing、boundary 等地图矢量监督。
- `plan_loss.py`：ego future trajectory 规划监督。
- `render_loss.py`：2D Gaussian Splatting 的语义 CE、深度 MSE 以及相关辅助项。
- `dynamic_loss.py`：Gaussian 动静属性和 2D 动静渲染监督，也支持 GT box oracle 模式。
- `physics_loss.py`：静态零位移、动态速度/轨迹、平滑和可选刚体一致性约束。
- `innovation_flow_loss.py`：创新 flow 方案的运动监督。
- `utils/`：CE、Lovasz、几何/语义缩放等通用 loss 函数。

## 7. `tools/`：数据、可视化和实验工具

`tools/` 中的脚本大多是离线工具，不属于默认训练主链。

- `tools/data/`：nuScenes PKL 转换、数据统计和数据质量检查。
- `tools/frontier/`：Frontier 区域构造、检查和可视化工具。
- `tools/viz/`：预测 Gaussian、occupancy、2D GT 和深度结果的可视化。
- `tools/profiling/`：性能分析、trace 和训练瓶颈定位。
- `tools/scratch/`：临时实验和一次性验证脚本，使用前先阅读脚本顶部说明。
- `tools/compare_depth_variance.py`：比较预测/伪标签深度分布和方差。
- `tools/viz_occflow_bug.py`：排查 occupancy flow 标签或可视化异常。

## 8. 数据、权重和生成目录

- `data/nuscenes/`：原始 nuScenes 数据或其链接。
- `data/nuscenes_cam/`：相机数据索引和转换后的 GaussianAD PKL。
- `data/surroundocc/`：occupancy 训练/验证样本。
- `data/dynamic_gt_nusc/`：动静分离使用的离线 dynamic GT。
- `data/depth_anchor_init_25600.npy`：深度初始化实验使用的 Gaussian anchor。
- `ckpts/`：预训练权重和可复用 checkpoint，例如图像 backbone 权重。
- `out/`：训练输出目录，包括日志、latest checkpoint、epoch checkpoint、render 可视化和 dynamic 可视化。
- `build/`、`pcdet.egg-info/`：本地编译或 Python 包安装产生的构建产物。
- `trace.json.gz`：性能分析工具生成的 trace 文件。

`out/`、`build/`、缓存目录和临时备份通常不需要纳入代码阅读范围；接手实验时优先关注实际使用的 config、日志和 checkpoint。

## 9. 建议的阅读顺序

1. 先看 `train.py`，了解训练入口和模块如何组装。
2. 再看当前使用的 `config/*.py`，确认实际启用的模型、数据路径和 loss。
3. 沿 `dataset/dataset.py` 查看一条样本如何从 PKL 变成 batch。
4. 沿 `model/segmentor/`、`model/encoder/` 和 `model/head/gaussian_head.py` 查看 Gaussian 主链。
5. 最后查看 `loss/multi_loss.py` 及具体 loss，确认每个模型输出如何产生梯度。
6. 需要复现实验时，再阅读对应配置目录下的 `train.sh` 和 `tools/` 脚本。
