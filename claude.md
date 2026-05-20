# GaussianAD 伪标签监督方案

---

## ⚠️ 易错点：分支与 Conda 环境对应关系

**每次修改代码、运行训练前必须核查以下对应关系，如有疏忽请及时提醒！**

| 训练方案 | Git 分支 | Conda 环境 | 说明 |
|----------|----------|------------|------|
| 原始监督（occ + det + map 等标准 loss） | `main` | `/data/chenz/conda_env/GaussianAD` | 当前正在跑的 noplan 训练 |
| 伪标签监督（2D Gaussian Splatting + gsplat） | `splatting` | `/data/chenz/conda_env/splatting` | gsplat 只在 splatting 环境中安装 |
| 训练速度优化 | `faster` | `/data/chenz/conda_env/faster` | 基于 splatting 分支，专注训练加速优化 |

**规则：**
- 代码修改后 push 到对应分支，远端 pull 对应分支再训练
- ⚠️ `/data/chenz/conda_env/` 下的环境**没有** `activate` 脚本，**不能**用 `source activate` 或 `conda activate`
- 必须用完整路径调用：`/data/chenz/conda_env/splatting/bin/python` 和 `/data/chenz/conda_env/splatting/bin/torchrun`
- faster 分支同理：`/data/chenz/conda_env/faster/bin/python` 和 `/data/chenz/conda_env/faster/bin/torchrun`
- 两套方案互不干扰：本地切分支不影响远端正在跑的训练

---

## 背景

GaussianAD 是一个基于 3D Gaussian 的自动驾驶感知/规划框架，使用 nuScenes 数据集训练。
当前目标：利用现成的伪标签数据为训练过程增加额外的弱监督信号，提升模型性能。

---

## 伪标签数据来源

位于远程服务器 `/data/chenz/GaussianFlowOcc_test/data/`：

| 目录 | 来源模型 | 内容 | 格式 |
|------|----------|------|------|
| `grounded_sam_nusc/` | Grounded SAM | 图像语义分割 mask（逐像素类别标签） | (6, 900, 1600) int8，nusc 17类（0~16） |
| `metric_3d_nusc/` | Metric3D | 单目深度估计图（逐像素深度值） | (6, 900, 1600) float16，度量深度（米），范围约 2.8~168m |

**文件路径格式：** `{root}/scene-{xxxx}/{sample_token}.npy`

这两类伪标签均为离线预计算，推理时不依赖外部模型。

---

## GaussianAD 架构概览

```
图像输入 (B, F, 6, C, H, W)
  ↓
ResNet50 + FPN → ms_img_feats [B*F, 6, C, H, W]
  ↓
GaussianLifter → anchor (B, 3600, 28)
  ↓
GaussianOccEncoder:
  - AnchorEncoder: anchor → 256维特征
  - 6层迭代 (spconv + deformable + ffn + refine):
    * DeformableFeatureAggregation 聚合图像特征
    * SparseGaussian3DRefinementModule 优化 Gaussian 参数
  ↓
GaussianHead:
  - LocalAggregator: Gaussian × sampled_xyz → [1, 18, num_samples] 占用率预测
  - forward_flow: 生成 flow 预测
  ↓
Loss 计算:
  - OccupancyLoss (占用率 + 语义)
  - OccupancyFlowLoss (动态预测)
  - DetectionLoss (3D 检测)
  - [splatting 分支新增] RenderLoss (伪标签语义 + 深度)
  - MapLoss (splatting 分支禁用，省出显存给渲染)
  - PlanLoss (禁用)
```

**Gaussian 属性**（3600 个高斯）：
- `means` (B, G, 3)：3D 位置，范围 [-30,30]×[-30,30]×[-2,2]（米）
- `scales` (B, G, 3)：缩放因子，范围 [0.1, 0.6]
- `rotations` (B, G, 4)：归一化四元数
- `opacities` (B, G, 1)：不透明度 ∈ [0,1]
- `semantics` (B, G, 17)：17类语义概率（softmax）

**关键约束**：
- 当前**只有 3D→3D 渲染**（无 2D 渲染），LocalAggregator 渲染结果为 3D 占用语义
- **无显式深度输出**，深度信息隐含在高斯协方差矩阵中

---

## 路线决策：2D 高斯泼溅监督（已确定）

> **适用分支**：`splatting` / `faster`（faster 完整继承 splatting 功能）

**放弃 3D Lifting，采用 2D Splatting 方案。**

### 决策理由

| 对比维度 | 3D Lifting（❌ 放弃） | 2D Splatting（✅ 采用） |
|----------|----------------------|------------------------|
| 误差来源 | metric_3d误差 → 反投影误差 → 体素化误差（三层叠加） | 直接在 2D 空间对比，无坐标转换 |
| 梯度质量 | 体素化不可微，梯度粗糙 | 可微渲染，梯度精确传回每个高斯 |
| 多视图约束 | 无 | 六相机同时施压，天然几何一致性 |
| 已有验证 | 无 | GaussianFlowOcc 已用此方案训练成功 |
| 遮挡处理 | 用错误深度估计强行填充 | 遮挡区域不计 loss（不知道比知道错的好）|

### 核心思路

```
3D Gaussians
  ↓ 取 semantics_logits（softplus 之前的 raw logits）
  ↓ gsplat.rasterization（可微 2D 渲染，仅训练时）
  ↓
渲染语义图 (nC, H', W', 17) + 渲染深度图 (nC, H', W')
  ↓                               ↓
CE Loss vs grounded_sam      MSE Loss vs metric_3d
（跳过label=0天空/背景）      （只算depth∈[0.5, 40]m）
  ↓                               ↓
梯度回传 → 优化高斯的 means / scales / rotations / semantics
```

**关键**：gsplat 渲染使用的是 `semantics_logits`（raw logits），而非 `semantics`（softplus 激活后的值）。
这样渲染结果仍是 logits 空间，可直接用 `CrossEntropyLoss` 计算语义损失。

推理时**完全不走 2D 渲染分支**，3D 输出路径（LocalAggregator）不受影响。

---

## Splatting 分支训练策略（已确定，2026-05-15 更新）

> **适用分支**：`splatting` / `faster`

### Loss 配置（全量 loss）

| Loss | 状态 | 理由 |
|------|------|------|
| OccupancyLoss | ✅ 保留 | 核心 3D 占用语义监督 |
| OccupancyFlowLoss | ✅ 保留 | 核心动态场景监督 |
| DetectionLoss | ✅ 保留 | 核心 3D 检测监督 |
| MapLoss | ✅ 保留 | 地图结构监督（显存允许，已加回） |
| RenderLoss（伪标签语义+深度） | ✅ 新增 | gsplat 2D 渲染 vs 伪标签 |
| PlanLoss | ✅ 保留 | 规划监督（显存允许，已加回） |

> **历史变更**：Phase 1 最初去掉了 MapLoss 和 PlanLoss 以省显存。2026-05-15 实测发现全量 loss（6 个）在单卡 96GB 上仅占 67-76 GB，无需任何 gradient checkpointing，因此全部加回。

### 显存实测（2026-05-15）

| 配置 | GPU数 | with_cp | 显存占用 | 状态 |
|------|-------|---------|----------|------|
| 全量 loss（6个），无 cp | 8卡 | `False` | 67-76 GB / 96 GB | ✅ 安全运行 |
| 仅 occ+flow+det+render，无 cp | 4卡 | `False` | ~54 GB | — |

### 渲染分辨率

- **初始值 0.44×**（396×704），GaussianFlowOcc 已验证
- 做成可配置参数 `pseudo_label_scale`，VRAM 有余量可调高
- H20 96GB 单卡，全量 loss + 0.44× 渲染，显存 67-76 GB，安全

### 阶段规划

- **Phase 1**（当前）：全量 loss（occ+flow+det+map+plan+render）联合训练 → 对比 main 分支 baseline
- **Phase 2**（如 Phase 1 涨点）：去掉标准 loss，仅伪标签监督 → 探测上限

---

## 实现计划

> **适用分支**：`splatting` / `faster`（记录 2D Splatting 监督的完整实现过程）

### Phase 1 — 确认 gsplat 环境

GaussianFlowOcc 已使用 gsplat，H20 服务器上应已安装。需确认：

```bash
/opt/miniconda/envs/GaussianAD/bin/python -c "from gsplat import rasterization; print('ok')"
```

若未安装：
```bash
/opt/miniconda/envs/GaussianAD/bin/pip install gsplat
```

---

### Phase 2 — 扩展 Dataloader（`dataset/dataset.py`）

在 `__getitem__` 中加入伪标签读取，**同时做下采样**（原始 900×1600 太大，对齐 GaussianFlowOcc 的 0.44 倍缩放）：

```python
import torch.nn.functional as F

# 路径构造
scene_name = info['scene_name']         # e.g. 'scene-0001'
sample_token = info['token']
depth_path = os.path.join(self.metric3d_root, scene_name, sample_token + '.npy')
seg_path   = os.path.join(self.grounded_sam_root, scene_name, sample_token + '.npy')

# 加载原始伪标签 (6, 900, 1600)
pseudo_depth = torch.tensor(np.load(depth_path).astype(np.float32))  # float32
pseudo_seg   = torch.tensor(np.load(seg_path).astype(np.int64))       # int64

# 下采样到渲染分辨率（可配置，默认 ×0.44 → 396×704）
scale = self.pseudo_label_scale  # 0.44
pseudo_depth = F.interpolate(pseudo_depth[:, None], scale_factor=scale, mode='bilinear').squeeze(1)
pseudo_seg   = F.interpolate(pseudo_seg[:, None].float(), scale_factor=scale, mode='nearest').squeeze(1).long()

# 裁剪顶部天空区域（可选，crop_top=140 对应原始分辨率，下采样后约 62 行）
crop = int(140 * scale)
pseudo_depth = pseudo_depth[:, crop:]
pseudo_seg   = pseudo_seg[:, crop:]

# 超过 max_depth 的区域语义置为无效（0）
pseudo_seg[pseudo_depth > self.max_pseudo_depth] = 0
pseudo_depth[pseudo_depth > self.max_pseudo_depth] = 0.

results['pseudo_depth'] = pseudo_depth   # (6, H', W')
results['pseudo_seg']   = pseudo_seg     # (6, H', W')
```

**Config 新增参数：**
```python
dataset = dict(
    metric3d_root='/data/chenz/Gaussianflowocc_test/data/metric_3d_nusc',
    grounded_sam_root='/data/chenz/Gaussianflowocc_test/data/grounded_sam_nusc',
    pseudo_label_scale=0.44,   # 下采样因子，渲染分辨率 = 原始 × scale
    max_pseudo_depth=40.0,     # 超过此深度的区域不用于监督
)
```

---

### Phase 3 — 新增 2D 渲染模块（`model/head/gaussian_rasterizer.py`，新建）

参考 GaussianFlowOcc 的 rasterizer 实现：

```python
from gsplat import rasterization
import torch, torch.nn as nn, torch.nn.functional as F

class GaussianRasterizer2D(nn.Module):
    """
    只在训练时使用，将 3D Gaussian 渲染到相机平面。
    输入的相机参数需从 ego 坐标系转换到 gsplat 需要的格式。
    """
    def __init__(self, render_h, render_w, sem_lw=2.0, depth_lw=0.05,
                 dynamic_classes=(2,3,4,5,6,7,9,10)):
        super().__init__()
        self.H, self.W = render_h, render_w
        self.sem_lw = sem_lw
        self.depth_lw = depth_lw
        # 动态类别（位置不稳定，深度 loss 时屏蔽）
        self.dynamic_classes = torch.tensor(list(dynamic_classes))

        nusc_class_freq = torch.tensor([
            944004, 1897170, 152386, 2391677, 16957802, 724139,
            189027, 2074468, 413451, 2384460, 5916653, 175883646,
            4275424, 51393615, 61411620, 105975596, 116424404
        ], dtype=torch.float32)
        log_w = torch.log(nusc_class_freq.sum() / nusc_class_freq)
        self.register_buffer('class_weight', log_w / log_w.mean())

    def forward(self, gaussian, gs_extrins, gs_intrins):
        """
        gaussian:    GaussianPrediction 对象（含 means/scales/rotations/opacities/semantics）
        gs_extrins:  (B, nC, 4, 4)  ego2cam 变换矩阵
        gs_intrins:  (B, nC, 3, 3)  相机内参

        返回:
          rendered_sem:   (B, nC, H, W, 17)  语义 logits
          rendered_depth: (B, nC, H, W)      深度（米）
        """
        B = gaussian.means.shape[0]
        all_sem, all_depth = [], []

        for b in range(B):
            sem_b, depth_b = [], []
            for c in range(gs_extrins.shape[1]):
                rendered, _, _ = rasterization(
                    means=gaussian.means[b],       # (G, 3)
                    quats=gaussian.rotations[b],   # (G, 4)
                    scales=gaussian.scales[b],     # (G, 3)
                    opacities=gaussian.opacities[b, :, 0],  # (G,)
                    colors=gaussian.semantics[b],  # (G, 17)
                    viewmats=gs_extrins[b, c:c+1], # (1, 4, 4)
                    Ks=gs_intrins[b, c:c+1],       # (1, 3, 3)
                    width=self.W, height=self.H,
                    render_mode='RGB+D',
                )
                # rendered: (1, H, W, 18)  前17维语义，最后1维深度
                sem_b.append(rendered[0, ..., :17])   # (H, W, 17)
                depth_b.append(rendered[0, ..., 17])  # (H, W)
            all_sem.append(torch.stack(sem_b))    # (nC, H, W, 17)
            all_depth.append(torch.stack(depth_b))# (nC, H, W)

        return torch.stack(all_sem), torch.stack(all_depth)  # (B,nC,H,W,17), (B,nC,H,W)

    def compute_loss(self, rendered_sem, rendered_depth, pseudo_seg, pseudo_depth):
        """
        rendered_sem:   (B, nC, H, W, 17)
        rendered_depth: (B, nC, H, W)
        pseudo_seg:     (B, nC, H, W)  int，0=无效/天空
        pseudo_depth:   (B, nC, H, W)  float，0=无效
        """
        # ── 语义 loss ──────────────────────────────────────────
        pred_sem = rendered_sem.flatten(0, -2)   # (N, 17)
        target_sem = pseudo_seg.flatten()         # (N,)
        valid_sem = target_sem > 0               # 跳过天空/背景（label=0）
        pw = self.class_weight.to(pred_sem.device)[target_sem[valid_sem] - 1]
        loss_sem = self.sem_lw * (
            pw * F.cross_entropy(pred_sem[valid_sem],
                                  target_sem[valid_sem] - 1,
                                  reduction='none')
        ).mean()

        # ── 深度 loss ──────────────────────────────────────────
        pred_d = rendered_depth.flatten()
        target_d = pseudo_depth.flatten()
        # 只在静态类区域计算（动态物体深度不可靠）
        dyn_mask = torch.isin(pseudo_seg.flatten(),
                              self.dynamic_classes.to(pseudo_seg.device))
        valid_d = (target_d > 0.5) & ~dyn_mask
        loss_depth = self.depth_lw * F.mse_loss(pred_d[valid_d], target_d[valid_d])

        return loss_sem, loss_depth
```

---

### Phase 4 — 接入 GaussianHead（`model/head/gaussian_head.py`）

在 `forward()` 末尾，LocalAggregator 渲染之后，新增 2D 渲染分支：

```python
# 只在训练时触发
if self.training and self.rasterizer_2d is not None:
    # 从 metas 提取相机参数
    gs_extrins = metas['ego2cam']    # (B, nC, 4, 4)  需确认 key 名
    gs_intrins = metas['cam_intrinsic_render']  # (B, nC, 3, 3) 下采样后的内参

    rendered_sem, rendered_depth = self.rasterizer_2d(gaussian, gs_extrins, gs_intrins)
    output_dict['rendered_sem']   = rendered_sem
    output_dict['rendered_depth'] = rendered_depth
```

在 `__init__` 中初始化：
```python
self.rasterizer_2d = GaussianRasterizer2D(
    render_h=config.render_h,   # 例如 396（900 × 0.44）
    render_w=config.render_w,   # 例如 704（1600 × 0.44）
    sem_lw=config.sem_lw,
    depth_lw=config.depth_lw,
) if config.use_pseudo_label else None
```

---

### Phase 5 — 相机参数预处理（`dataset/dataset.py`）

gsplat 需要的 `viewmat` 是 **world2cam（或 ego2cam）** 的 4×4 矩阵，`K` 是 **下采样后的 3×3 内参**。

```python
# ego2cam = cam2ego 的逆
cam2ego = info['cam2ego']                        # (6, 4, 4)
ego2cam = np.linalg.inv(cam2ego)                 # (6, 4, 4)

# 内参需乘以下采样因子
K = info['cam_intrinsic'][:, :3, :3].copy()      # (6, 3, 3)
K[:, :2, :] *= pseudo_label_scale               # fx, fy, cx, cy 同比缩放
# 顶部裁剪：cy 需要减去裁剪行数
K[:, 1, 2] -= crop_pixels

results['gs_extrins'] = ego2cam   # (6, 4, 4)
results['gs_intrins_render'] = K  # (6, 3, 3)
```

---

### Phase 6 — Loss 注册（`loss/render_loss.py`，新建）

```python
from loss import OPENOCC_LOSS
from loss.base_loss import BaseLoss

@OPENOCC_LOSS.register_module()
class RenderLoss(BaseLoss):
    def __init__(self, weight=1.0, input_dict=None, **kwargs):
        super().__init__(**kwargs)
        self.weight = weight
        if input_dict is None:
            self.input_dict = {
                'rendered_sem':   'rendered_sem',
                'rendered_depth': 'rendered_depth',
                'pseudo_seg':     'pseudo_seg',
                'pseudo_depth':   'pseudo_depth',
            }

    def forward(self, rendered_sem, rendered_depth,
                pseudo_seg, pseudo_depth, **kwargs):
        # 调用 rasterizer_2d.compute_loss（或在这里重写）
        loss_sem, loss_depth = compute_render_loss(
            rendered_sem, rendered_depth, pseudo_seg, pseudo_depth)
        return (loss_sem + loss_depth) * self.weight
```

**Config 新增：**
```python
loss = dict(
    ...
    render=dict(type='RenderLoss', weight=1.0),
)
```

---

## 关键文件

> **适用分支**：`splatting` / `faster`

| 文件 | 改动类型 | 说明 |
|------|----------|------|
| `dataset/dataset.py` | 修改 | 加载伪标签、下采样、提取 ego2cam/render_K |
| `model/head/gaussian_rasterizer.py` | **新建** | GaussianRasterizer2D：gsplat 封装 + loss 计算 |
| `model/head/gaussian_head.py` | 修改 | 训练时调用 2D 渲染，输出 rendered_sem/depth |
| `loss/render_loss.py` | **新建** | RenderLoss：注册到 OPENOCC_LOSS |
| `loss/__init__.py` | 修改 | 导入 RenderLoss |
| `loss/multi_loss.py` | 修改 | 注册 RenderLoss |
| `config/nuscenes_gs25600.py` | 修改 | 伪标签路径、渲染分辨率、loss 配置 |

---

## 待确认细节

> **适用分支**：`splatting` / `faster`

- [x] H20 上 splatting 环境已安装 gsplat（训练已正常运行验证）
- [x] `cam_intrinsic` 是 **(6, 4, 4)**，dataset 中读取后取 `[:, :3, :3]` 得到 3×3 内参
- [x] `ego2cam` 在 **dataset.py** 里计算：`np.linalg.inv(cam2ego)`，保存为 `gs_extrins`
- [x] 渲染分辨率确定为 **0.44×**（396×704），已在训练中运行稳定
- [ ] 数据增强（flip）时 pseudo_seg/pseudo_depth 需同步翻转（当前 rand_flip=True，暂未处理）

---

## 待解决问题

> **适用分支**：`splatting` / `faster`

- [x] 确认伪标签文件命名/目录结构：`scene-{xxxx}/{sample_token}.npy`，每个 .npy 含 6 个相机
- [x] 确认 GaussianAD 是否输出深度图：**无显式 depth render**，深度隐含在高斯协方差矩阵中
- [x] 确认 encoder 是否有语义特征图：**无直接可监督的 2D 语义图**，最终输出为 3D occ logits
- [x] 伪标签实际路径：`/data/chenz/Gaussianflowocc_test/data/`（注意大小写）
- [x] H20 上 splatting 环境已安装 gsplat，训练正常运行
- [x] `cam_intrinsic` 维度：(6, 4, 4)，取 `[:, :3, :3]` 用于渲染内参
- [x] `ego2cam` 在 dataset 中计算，key 名为 `gs_extrins`
- [x] 渲染分辨率已确定：0.44×（396×704）
- [ ] 数据增强（flip）时伪标签需同步变换（待修复）

---

## Loss 详解

> **适用分支**：`splatting` / `faster`（DetectionLoss 三分支通用；RenderLoss 仅 splatting/faster）

### DetectionLoss

3D 目标检测 anchor-free 头的损失，由两部分构成：

| 部分 | 实现 | 权重 |
|------|------|------|
| Heatmap（分类） | `FocalLossSparse` | `cls_weight=1.0` |
| Regression（定位） | `RegLossSparse`（L1） | `loc_weight=0.25`，10属性各有 code_weight |

**10个检测属性**（`head_order`）：center(x,y), center_z, dim(l,w,h), rot(sin,cos), vel(vx,vy)
`code_weights = [1,1,1,1,1,1,0.2,0.2,1,1]`（旋转权重 0.2）

**检测目标类别**：car, truck, construction_vehicle, bus, trailer, barrier, motorcycle, bicycle, pedestrian, traffic_cone（共10类）

底层使用 VoxelNeXt 稀疏检测头，在高斯点云的稀疏体素上运行。

### RenderLoss 权重配置

```python
weight=1.0     # 整体 loss 权重（乘在语义+深度总和之上）
sem_lw=2.0     # 语义 CE loss 内部权重
depth_lw=0.05  # 深度 MSE loss 内部权重（数值通常大，故小权重平衡）
```

实际语义 loss 数值约 2.0~4.0，深度 loss 数值约 0.01~0.05（乘 lw 后），两者相加即 RenderLoss。

---

## GaussianAD 渲染系统详解

> **适用分支**：`main` / `splatting` / `faster`（共享，记录 3D occupancy 渲染系统，与 2D Splatting 无关）

### 渲染模块位置
- `model/head/gaussian_head.py` — `GaussianHead` 类，`forward()` 调用渲染
- `model/head/localagg/local_aggregate/__init__.py` — `LocalAggregator` 类
- `model/head/localagg/src/forward.cu` — `renderCUDA` 核心 CUDA 算法

### 渲染输出
- **语义 logits**：`(N, 18)` 其中 N = 采样点数，C = 18 类
- **不是** RGB 图像，也**不是** depth map
- 用于 3D occupancy 网格预测

### renderCUDA 核心算法（Mahalanobis 加权）
```cuda
对每个采样点 pts[idx]:
  d = means3D[gs_idx] - point        // 高斯中心到采样点的距离向量
  power = -0.5 * d^T * CovInv * d   // Mahalanobis 距离
  weight = opacity[gs_idx] * exp(power)
  C[ch] += semantic[gs_idx, ch] * weight  // 加权累积语义特征
output[idx] = C[]
```

### LocalAggregator 初始化参数
- `scale_multiplier=3`，体素网格 `H=200, W=200, D=16`
- `pc_min = [-40, -40, -1]`，`grid_size = 0.4`（米）

### Loss 的 input_dict 结构
**OccupancyLoss**（当前使用）：
```python
{
  'pred_occ': [pred_semantics],  # List[Tensor(1, 18, num_samples)]
  'sampled_xyz': Tensor(1, num_samples, 3),
  'sampled_label': Tensor(1, num_samples),
  'occ_mask': Tensor(1, H, W, D)  # optional
}
```

### Dataset 关键返回 Key
```python
{
  'img': [B*F*6, C, H, W],
  'cam_intrinsic': [6, 4, 4],   # 相机内参 ← 3D lifting 需要
  'cam2ego': [6, 4, 4],         # 外参 ← 3D lifting 需要
  'occ_xyz': [H*W*D, 3],        # GT 占用点坐标
  'occ_label': [H*W*D],         # GT 占用语义标签
  ...
}
```

---

## 开发工作流

> **适用分支**：`main` / `splatting` / `faster`（共享，注意推送到各自对应分支）

```bash
# 本地修改后推送
git add .
git commit -m "add pseudo label supervision"
git push origin splatting

# 远程拉取并训练（train.py 自动检测 latest.pth，无需额外参数即可接续）
ssh -p 30300 root@8.130.174.55 "cd /data/chenz/GaussianAD && git pull origin splatting"
# 之后重启训练即可，work-dir 相同则自动接续
```

### 接续训练说明

`train.py` 在启动时自动检测 `{work_dir}/latest.pth`（代码第183行），只要 `--work-dir` 不变，停训后重启**完全自动接续**，无需额外参数：
- 恢复内容：模型权重（strict=False）、optimizer 状态、scheduler 状态、epoch、global_iter
- 运行时状态（如 `_diag_counter`）从 0 重新计数，不影响权重

---

## RenderLoss 有效性诊断

> **适用分支**：`splatting` / `faster`

### 当前训练状态（2026-05-18）

- splatting 分支：8卡，max_epochs=30，从 epoch 18 接续（off-by-one 修复后）
- 修复前（epoch 1-18）：RenderLoss 均值约 **2.84**，类别索引错位，bicycle 全 0%
- 修复后（epoch 18 起）：RenderLoss 第一个 iter = **8.77**（正确 target 更难），预计逐步下降
- 预计 epoch 22-24 开始出现 bicycle IoU > 0%

### 四步诊断法

| 步骤 | 方法 | 判断标准 |
|------|------|----------|
| 1 | 看 `[RenderLoss Diag]` 日志中的 `pred_depth_mean` | 若 ≈ 0 → 相机参数错误，高斯不在视锥内；若 5~30m → 渲染结构正常 |
| 2 | 看 `sem_entropy` | 若接近 2.833（随机基准）→ 语义未收敛；若 < 2.0 → 已学到有效信息 |
| 3 | 查看可视化图片 | `out/nuscenes_gs25600_splatting/render_vis/step_*.jpg`，预测语义/深度与 GT 对比 |
| 4 | 最终：对比 main 分支 mIoU | splatting vs noplan 相同 epoch 的验证集 mIoU |

### 快速查看 main 分支 mIoU 基准

```bash
ssh -p 30300 root@8.130.174.55 "grep mIoU /data/chenz/GaussianAD/out/nuscenes_gs25600_noplan_run.log | tail -5"
```

### 可视化图片格式

每次 `vis_every=500` iter 保存一张 JPEG：
- 6 相机纵向堆叠
- 每行横向拼接：`[预测语义 | GT语义 | 预测深度图 | GT深度图]`
- 深度颜色：黑→蓝→绿→红（近→远），灰色=无效像素

---

## 调试记录（2026-05-11 Splatting 分支首次启动）

> **适用分支**：`splatting`（首次启动 Bug 记录，faster 分支同样适用，因为继承了相同代码路径）

### Bug 1：conda 环境无 activate 脚本

**现象：** tmux 中 `source activate /data/chenz/conda_env/splatting` 报错，训练进程崩溃。

**原因：** `/data/chenz/conda_env/` 路径下的环境是 conda 非标准安装，没有 `activate` 脚本。

**修复：** 直接用完整路径调用可执行文件：
```bash
# 单卡测试
CUDA_VISIBLE_DEVICES=0 /data/chenz/conda_env/splatting/bin/python train.py --py-config config/nuscenes_gs25600.py --work-dir out/nuscenes_gs25600_splatting

# 多卡训练
CUDA_VISIBLE_DEVICES=0,1,2,3 /data/chenz/conda_env/splatting/bin/torchrun \
    --nproc_per_node 4 --master_port 12457 \
    train.py --py-config config/nuscenes_gs25600.py --work-dir out/nuscenes_gs25600_splatting
```

---

### Bug 2：KeyError 'ori_intrinsic' / 'cam2ego'（dataset pipeline in-place 修改）

**现象：**
```
KeyError: 'ori_intrinsic'
KeyError: 'cam2ego'
```

**原因：** `dataset.py` 的 pipeline（`self.pipeline(input_dict)`）会 **in-place 修改** `input_dict`，导致：
- `cam_intrinsic` 被替换为 pipeline 处理后的版本（已 resize/crop）
- `cam2ego` 也可能被 pipeline 重命名或移除

因此，如果在 pipeline **之后**才保存这两个值，会出现 KeyError。

**修复：** 在 `pipeline(input_dict)` **之前**，把需要原始值的字段另存：
```python
# 在 pipeline 运行之前保存原始值
input_dict['ori_intrinsic'] = input_dict['cam_intrinsic'].copy()
input_dict['cam2ego'] = info['cams_info']['cam2ego'].copy()  # 直接从 info 取

# 之后再运行 pipeline
example = self.pipeline(input_dict)
```

已修改文件：`dataset/dataset.py`（commit c4cd8a7, 93ce9fd）

---

### Bug 3：BaseLoss.loss_func 实例属性遮蔽子类方法

**现象：**
```
TypeError: 'NoneType' object is not callable
# 或方法调用走到了 base class 的 loss_func 而非子类的
```

**原因：** `BaseLoss.__init__` 中有如下代码：
```python
self.loss_func = getattr(F, loss_type, None)
```
这会在实例上创建一个**实例属性** `loss_func`，遮蔽（shadow）了子类 `RenderLoss` 中定义的**同名方法**。Python 方法查找时实例属性优先于类方法，导致子类方法永远找不到。

**修复：** 在 `RenderLoss.__init__` 的末尾 `del self.loss_func`，删除继承来的实例属性，让方法查找回归正常 MRO：
```python
class RenderLoss(BaseLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # BaseLoss.__init__ 会设置 self.loss_func = None（实例属性），
        # 遮蔽子类的 loss_func 方法，必须删除
        if hasattr(self, 'loss_func') and not callable(type(self).loss_func):
            del self.loss_func
```

已修改文件：`loss/render_loss.py`（commit 95d88ee）

---

### 调试记录（2026-05-14 RenderLoss 有效性诊断 + 可视化模块）

#### 现象：RenderLoss 不下降，均值约 2.84

训练至 Epoch 4（约 2500 iter），RenderLoss 均值稳定在 2.84，接近随机猜测基准 $\log(17) \approx 2.833$，判断语义渲染尚未学到有效信息。

**两种可能原因：**
1. 相机参数（ego2cam / 渲染内参）有误 → 高斯没有落在相机视锥内，渲染图全为噪声
2. 训练尚早 → 高斯 semantics 还在随机初始状态，需要更多 epoch 收敛

**判断方法：** 看诊断日志中的 `pred_depth_mean`（0 → 参数有误；5~30m → 结构正常但语义未收敛）

#### 解决方案：增加诊断打印 + 可视化模块

**诊断打印**（`render_loss.py`，commit 8a9c9f4）：  
每 500 iter 打印一次 `[RenderLoss Diag]`，包含：
- `valid_sem` / `valid_depth` 像素比例
- `pred_depth_mean/std` 和 `gt_depth_mean`
- `sem_entropy`（与随机基准 2.833 对比）

**可视化模块**（`render_loss.py` + `config/nuscenes_gs25600.py`，commit 7fc218c）：  
新增 `vis_dir` / `vis_every` 配置项，每 500 iter 保存一张对比图：
```
out/nuscenes_gs25600_splatting/render_vis/step_000001.jpg
```
图片布局：6相机 × [预测语义 | GT语义 | 预测深度 | GT深度]

**接续训练：** 停训后用相同 `--work-dir` 重启即可自动 resume，无需额外参数。

---

### 调试记录（2026-05-15 语义渲染不学习的根因分析与修复）

#### 根因：softplus 激活 + alpha-blending ≠ logits

**问题描述：**
- 高斯的 `semantics` 字段经过 `softplus` 激活，输出为非负值
- gsplat 的 `rasterization` 对 `colors`（即 semantics）做 alpha-blending 混合
- `render_loss.py` 用 `CrossEntropyLoss` 对渲染结果计算 loss，期望输入为 **logits**
- 但 `softplus(x)` 经过 alpha-blend 后的值不再是 logits：
  - 始终为正数，无法表达"不是某类"（负 logit）
  - 多高斯混合后值趋于平均，梯度极弱

**本质错误：** `Σ αᵢ·softplus(logitᵢ)` ≠ logits，直接用 CE loss 无法有效学习。

#### 修复方案：Plan A — 渲染 raw logits（commit e4a4d4a）

1. `model/encoder/gaussian_encoder/refine_module.py`：
   - `GaussianPrediction` 新增 `semantics_logits` 字段（softplus 之前的 raw 值）
   - 两个 `get_gaussian` 方法都返回 `semantics_logits=raw_logits`

2. `model/head/gaussian_rasterizer.py`：
   - `forward()` 中使用 `gaussian.semantics_logits` 而非 `gaussian.semantics`
   - 渲染结果仍在 logits 空间，`CrossEntropyLoss` 直接适用

3. 不影响 3D occupancy 路径：
   - `LocalAggregator` 仍使用 `gaussian.semantics`（softplus 后）
   - 与 OccupancyLoss 的交互完全不变

**关键理解：**
- `semantics`（softplus 后）→ 3D occupancy 渲染（Mahalanobis 加权）→ OccupancyLoss
- `semantics_logits`（raw）→ 2D gsplat 渲染（alpha-blending）→ CrossEntropyLoss

---

### 全量 loss 启用记录（2026-05-15）

#### 背景

最初 splatting 分支去掉了 MapLoss 和 PlanLoss，原因是担心显存不够。
实测发现 occ+flow+det+render 仅占 ~54 GB / 96 GB，有 40+ GB 余量。

#### 改动

- 加回 MapLoss（完整 MapTRv2 配置）
- 加回 PlanLoss（weight=10.0）
- `frozen_modules = []`（解冻 map_decoder 和 planner_head）
- `loss_input_convertion` 加回 map/plan 相关 key

#### Optimizer state 兼容性

由于解冻模块导致 optimizer param groups 数量变化，`train.py` 中加了 try/except：
```python
try:
    optimizer.load_state_dict(ckpt['optimizer'])
except ValueError as e:
    logger.info(f'Optimizer state mismatch ..., skipping optimizer resume: {e}')
```
Optimizer 不 resume 仅影响前几个 iter 的动量/学习率预热，对训练质量无实质影响。

#### 训练启动方式（tmux，8卡）

```bash
# 在 tmux session train_splatting 中运行
cd /data/chenz/GaussianAD && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 /data/chenz/conda_env/splatting/bin/torchrun \
    --nproc_per_node 8 --master_port 12457 \
    train.py --py-config config/nuscenes_gs25600.py --work-dir out/nuscenes_gs25600_splatting --dataset nuscenes
```

---

### 调试记录（2026-05-18 RenderLoss 类别索引 off-by-one 根因分析与修复）

#### 现象：bicycle IoU 18 个 epoch 全部 0%

- splatting 分支跑完 18 epoch，bicycle IoU 始终 0.00%（precision=0, recall=0）
- 对比 v4 baseline：epoch 8（step 等价）时 bicycle = 2.90%，说明异常
- `total_positive` 中 bicycle 从 179 → 164 → 98，逐 epoch 下降，说明模型在主动**学会不预测** bicycle
- motorcycle（第二稀有）也从 v4 的 9.68% 降到 splatting 的 1.16%

#### 根因：CE target 错位一位

`render_loss.py` 原始代码：
```python
loss_sem = CE(pred_sem[valid_sem], target_sem[valid_sem] - 1)
```

Gaussian 17 个语义通道与 OccupancyLoss / pseudo_seg 的映射关系：
- OccupancyLoss：channel 0=noise, 1=barrier, 2=bicycle, ..., 16=vegetation
- pseudo_seg：label 0=invalid, 1=barrier, 2=bicycle, ..., 16=vegetation

`- 1` 导致每个类别错位一位：

| 通道 | OccLoss 教它（正确） | RenderLoss 教它（bug） |
|------|---------------------|----------------------|
| 0 | noise | barrier ❌ |
| 1 | barrier | bicycle ❌ |
| 2 | bicycle | bus ❌ |
| ... | ... | ... |

bicycle（44K GT voxels，最稀有）的 OccLoss 信号本就微弱，被 RenderLoss 的错误梯度完全压制 → 18 epoch 全 0%。

**同时，可视化图（render_vis）的 pred 侧也有同样的 palette 错位**，导致历史图片里预测颜色和 GT 颜色不是同一套映射，无法用于诊断。

#### 修复（commit eb138cf）

1. **CE target**：去掉 `- 1`，直接用 `target_sem[valid_sem]` 作为 CE target（1-16 对应 channel 1-16）
2. **可视化 pred 侧**：channel 0 → 灰色（noise），channel 1-16 → `palette[channel - 1]`（与 GT 侧对齐）

```python
# 修复后
loss_sem = CE(pred_sem[valid_sem], target_sem[valid_sem])  # 无 -1

# 可视化 pred 侧
pred_sem_rgb = np.where(
    (pred_cls[..., None] > 0),
    _NUSC_PALETTE[np.clip(pred_cls - 1, 0, 16)],
    np.array([128, 128, 128])   # class 0 (noise) → gray
)
```

#### 处置方案

- 已跑 18/20 epoch，剩余 2 epoch 不足以让被污染的权重恢复
- 将 `max_epochs = 20 → 30`（commit b19c429），接续 latest.pth 继续训练
- 修复后第一个 iter：`RenderLoss: 8.77`（从平均 2.2 飙升，正常——正确 target 更难拟合）
- 预计 epoch 22-24 开始看到 bicycle IoU > 0%

#### 关于 val RenderLoss = 0

**这是正常的、有意设计的行为。**
- `gaussian_head.py` 在 `self.training=False` 时跳过 gsplat 渲染，输出 `None`
- `render_loss.py` 收到 `None` 直接返回 `0.0`
- Val 评估只依赖 3D occupancy 路径（LocalAggregator），RenderLoss 对 mIoU 无任何影响
- 跳过的唯一原因：省时间（6 相机 × 4K+ val samples 开销大）

---

## PKL 数据转换脚本（tools/convert_nuscenes_infos_to_gaussianad.py）

> **适用分支**：`main` / `splatting` / `faster`（共享工具脚本，三个分支共用同一套 PKL 数据）

作者 GitHub 的 PKL 文件损坏，需要从标准 nuScenes infos PKL 自行转换。

### v6 改进（2026-05-18）

目标：使转换后 PKL 的数据分布尽量接近作者原始 PKL 的训练效果。

#### P0 修复（正确性，必须做）

| 改动 | 说明 |
|------|------|
| 写入 `info["scene_token"]` 和 `info["scene_name"]` | dataset.py L1603 pseudo-label 分支读 scene_token，不写则 KeyError 崩溃 |
| `num_lidar_pts` / `num_radar_pts` 回填真值 | 原来是 `np.ones((n,))` 占位，导致 dataset 的 `filter_min_points_in_gt` 过滤完全失效，保留了大量无效 gt_box |
| 重算 `gt_velocity` 到 LIDAR_TOP 坐标系 | 用 `nusc.box_velocity()` 返回 global 系速度，经 global→ego→lidar 两步旋转转换（由 `_velocity_global_to_lidar()` 完成），`--no-recompute-velocity` 可关闭 |

#### P1 修复（质量，提升分布匹配度）

| 改动 | 说明 |
|------|------|
| ego_fut_cmd 阈值对齐 VAD | `TURN_LATERAL_THRESHOLD` 1.0→2.0 m，`TURN_YAW_THRESHOLD` 0.20→0.0873 rad（5°），修复 STRAIGHT 命令占比过高（>0.90）的问题 |
| agent 匹配距离自适应 | `radius_cap = min(2.5 + 0.05·d_ego, 6.0)` m，`cost_cap = min(4.0 + 0.10·d_ego, 9.0)`，远处目标不再漏匹配 |
| `gt_ego_lcf_feat` 9 维全填 | [0:2] vx,vy / [2:4] ax,ay（二阶差分）/ [4] 偏航角速率 / [5:6] ego 车身尺寸(4.084, 1.730) / [7] \|v\| / [8] 转向代理量（自行车模型） |
| `min_map_line_length` 0.5→2.0 m | 对齐 VAD 配置，减少地图碎片 |
| `--mask-plan-outside-range` 默认 True | 超出 BEV 范围的 ego future step 自动掩膜，use `--no-mask-plan-outside-range` 关闭 |
| 输出文件名默认含 `_v6` 后缀 | 避免覆盖旧版 |

#### 常量

```python
MATCH_BASE_RADIUS   = 2.5    # 距离自适应匹配基础半径
MATCH_DIST_SLOPE    = 0.05   # 每米增量
MATCH_MAX_RADIUS    = 6.0    # 上限
NUSC_EGO_LENGTH     = 4.084  # Renault Zoe
NUSC_EGO_WIDTH      = 1.730
TURN_LATERAL_THRESHOLD = 2.0    # m
TURN_YAW_THRESHOLD     = 0.0873 # rad (~5°)
DEFAULT_MIN_MAP_LINE_LENGTH = 2.0  # m
```

#### 运行命令（H20）

```bash
/data/chenz/conda_env/GaussianAD/bin/python tools/convert_nuscenes_infos_to_gaussianad.py \
    --dataroot data/nuscenes --version v1.0-trainval \
    --surroundocc-train-dir data/surroundocc/train_samples \
    --surroundocc-val-dir   data/surroundocc/val_samples
# 输出: data/nuscenes_cam/nuscenes_infos_{train,val}_gaussian_ad_v6.pkl
```

### 统计分析脚本（tools/stats_gaussianad_pkl.py）

转换完成后用此脚本体检 PKL 质量，不依赖作者原始 PKL：

```bash
python tools/stats_gaussianad_pkl.py \
    --pkl data/nuscenes_cam/nuscenes_infos_train_gaussian_ad_v6.pkl
# 可选: --pkl-ref 旧版 pkl（对比 v5 vs v6 各项指标变化）
```

**9 大体检维度：**
1. 顶层结构（infos dict / metadata list 是否存在）
2. 必需 key 缺失（REQUIRED_FRAME_KEYS + PSEUDO_LABEL_KEYS）
3. ego_fut_cmd 命令分布（健康范围：STRAIGHT 55-85%，L/R 各 5-25%）
4. fut_valid_rate（≥80%）/ agent future 覆盖率（≥65%）
5. gt_boxes 分布（过滤前/后，含百分位数）
6. gt_map 三类元素数（divider/ped_crossing/boundary 均值）
7. velocity_norm 分布 / num_lidar_pts 分布（验证 P0 #2 写回效果）
8. 伪标签就绪度（scene_token/scene_name 缺失数）
9. 内置启发式健康告警（`[OK]`/`[WARN]`/`[FAIL]`）

**判读重点（v6 相较 v5 的预期变化）：**
- `num_lidar_pts.mean` 应从 ≈1 升至几十（真实点数）
- `cmd_ratio.STRAIGHT` 应从 >0.90 降至 0.65-0.80
- `ego_lcf_feat_nonzero_dim_count` dim2-6 应从 0 变为显著非 0
- `scene_token` / `scene_name` 缺失数应为 0

---

## faster 分支：GPU 训练加速优化

**创建时间**：2026-05-19  
**基础分支**：`splatting`（完整继承 splatting 所有功能）  
**Conda 环境**：`/data/chenz/conda_env/faster`（克隆自 splatting，7.1GB）  
**目标**：在不改变模型结构和 loss 的前提下，通过消除 GPU idle 时间提升训练速度。

### 性能瓶颈分析（来自 trace.json，单 iter ~10.5s profiler 时间）

| 瓶颈 | 来源 | 估计浪费 |
|------|------|---------|
| SubMConv backward 的 cudaFree/cudaMalloc | spconv 11次反向各 400ms | ~4400ms |
| backbone+encoder 历史帧重复 backward | with_cp 重算 + 3/4 帧无效梯度 | ~800ms |
| NCHW↔NHWC 格式转换 | backbone 497 个转换 kernel | ~268ms |
| lidar2global double→float 类型转换 | temporal encoder 每 iter | ~400ms |
| nonzero() GPU→CPU 同步 | gaussian_head get_filtered_lidar × 6 | ~4ms |

> **注**：trace 中 10.5s 是 profiler overhead（单卡、含仪器化开销），实际多卡训练时间更短。

### 已实现优化（history_no_grad：commit 0a3e6be；Opt-1/2/3：commit c29ca34；Opt-A/B：commit 36c229c）

#### history_no_grad：历史帧 backbone+encoder 不计算梯度（commit 0a3e6be，继承自 splatting）

**文件**：`model/segmentor/bev_segmentor.py`，`config/nuscenes_gs25600.py`

**Config**（line 229）：
```python
history_no_grad=True,
```

**原理**：GaussianAD 使用 **F=4 帧**时序输入，其中 3 帧是历史帧，只有最后 1 帧是当前帧。
在原始实现中，4 帧全部走 backbone + encoder forward，backward 也全部存储激活值，其中 3/4 的梯度实际上对模型学习没有意义（历史帧的特征只作为时序上下文，不直接决定当前预测）。

启用 `history_no_grad=True` 后：

```
backbone + encoder 的处理路径：
  历史帧（F-1=3帧）→ torch.no_grad() → 不存储激活，不接受梯度
  当前帧（最后1帧）→ 正常 autograd → 完整梯度回传

temporal_encoder：仍然接收全部 4 帧的特征（合并后输入，维度不变）
→ 时序信息完整保留，精度理论上不受影响
```

`_encoder_forward_split()` 方法将 anchor、ms_img_feats、metas 沿帧维度拆分，分别做 forward，再 merge 回 `(B*F, ...)` 格式，下游代码无需任何改动。

**收益**：
- **显存**：backbone+encoder 的激活存储量从 4 帧降为 1 帧（节省约 75%）
- **Backward 速度**：历史帧 backbone+encoder 的反向传播完全省略，节省 3/4 × (backbone_bwd + encoder_bwd)
- **精度影响**：理论上无影响，temporal_encoder 仍看到全部时序信息

#### Opt-1：channels_last 内存格式（预期 -268ms/iter）

**文件**：`model/segmentor/bev_segmentor.py`

```python
# _run_img_backbone_flat() 首行加入：
imgs_flat = imgs_flat.to(memory_format=torch.channels_last)
```

**原理**：cuDNN 内部偏好 NHWC 算法，但 PyTorch 默认 NCHW 存储，每层 conv 前后都要做格式转换。
改为 channels_last 后，转换消除。**DCNv2（stages 3,4）** 不支持 channels_last，PyTorch dispatcher
自动 fallback 到 contiguous NCHW，不会出错，stages 1,2 和 FPN 完整受益。

#### Opt-2：lidar2global 预转 float32（预期 -400ms/iter）

**文件**：`model/encoder/temporal_encoder/gaussian_temporal_encoder.py`

```python
# 原代码（numpy float64 → GPU 路径上做类型转换 → 触发 cudaStreamSynchronize）：
lidar2global = torch.tensor(metas['lidar2global'][0], dtype=anchors.dtype, device=anchors.device)

# 修改后（在 numpy 端做 float64→float32，zero-copy 送 GPU）：
lidar2global = torch.from_numpy(
    np.asarray(metas['lidar2global'][0], dtype=np.float32)
).to(anchors.device, non_blocking=True)
```

**原理**：`torch.tensor()` 在 GPU 路径上做 double→float 类型转换时，对 tiny tensor（如 [7,4,4]）
会触发同步阻塞（`cudaStreamSynchronize`）。改为 numpy 端预转换 + `non_blocking=True` 后，
类型转换在 CPU 完成，GPU 传输异步进行，消除同步点。

#### Opt-3：nonzero → bool mask（预期 -4ms/iter）

**文件**：`model/head/gaussian_head.py`

```python
# 原代码（nonzero 必须 GPU→CPU 同步确定输出大小）：
mask = torch.nonzero(mask).squeeze()
if len(mask) == 0: ...

# 修改后（bool mask 直接索引，避免同步）：
if not mask.any(): ...
return lidar[mask].unsqueeze(0).contiguous(), mask, valid
```

**原理**：`torch.nonzero()` 在返回前必须 GPU→CPU 同步（因为 output size 未知）。
换用 bool mask 后，下游 `tensor[bool_mask]` 索引方式完全兼容（调用处不需修改），
消除 `get_filtered_lidar` 的 6 次/iter 同步。

### 实测验证结果汇总（2026-05-20）

| 优化 | commit | 预期收益 | 实测结果 | 原因分析 |
|------|--------|---------|---------|---------|
| Opt-1：channels_last | c29ca34 | -268ms | ≈0 | DCNv2 stages 3,4 不支持 channels_last，自动 fallback；stages 1,2 + FPN 受益但占比小 |
| Opt-2：lidar2global float32 | f864379 | -400ms | ≈0 | Bug 修复 commit（isinstance 检查）引入一次额外 `.float()` 调用，收益被抵消 |
| Opt-3：nonzero→bool mask | bd7cbf5 | -4ms | ≈0 | 4ms 本来就太小，在 3.17s/iter 中噪声级别 |
| Opt-A：with_cp=False | 36c229c | -800ms（backbone 不重算） | 实测 ~3.10 s/iter（-2~3%） | 理论正确，但 spconv backward 4400ms 占主导，backbone 节省被掩盖 |
| Opt-B：DDP bucket=200MB | 36c229c | 减少 all-reduce 次数 | 几乎无变化 | all-reduce 在 spconv backward 之后，不是瓶颈 |
| Opt-C：max_split_size_mb | 启动命令 | 减少显存碎片 | 无可见提速 | 对 spconv 自管显存的行为影响有限 |

**结论：A+B+C 三方案合计实测 ~3% 提速（3.17→3.10 s/iter），未达预期。**

**根本原因**：真正的瓶颈是 **spconv 的 SubMConv backward 内部 cudaFree/cudaMalloc**（约 4400ms/iter，占 backward 63%）。这属于 spconv 库本身的实现问题，无法通过上层代码规避。

---

### 当前运行状态（2026-05-20）

- **训练正在运行**，tmux `train_nograd`，7卡（GPU 1-7）
- **速度**：~3.10 s/iter（稳态，iter 100+ 后）
- **启动命令**（含方案 A+B+C）：

```bash
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 /data/chenz/conda_env/faster/bin/torchrun \
    --nproc_per_node 7 --master_port 12459 \
    train.py --py-config config/nuscenes_gs25600.py \
    --work-dir out/nuscenes_gs25600_faster --dataset nuscenes
```

> **注意**：faster 环境（Python 3.8 + 旧版 PyTorch）**不支持** `expandable_segments:True`，
> 该选项会直接报错 `Unrecognized CachingAllocator option`，启动时已移除。

---

### spconv 提速实验（2026-05-20，结论：**H20 sm_90 上无解**）

> **适用分支**：`faster`（实验在 faster 分支进行，未影响 main/splatting）

#### 实验背景

baseline 3.10 s/iter 中，spconv SubMConv backward 占 ~4400ms（trace.json 实测），是最大单一瓶颈。
尝试通过 spconv 的两条优化路径攻克：

1. **Opt-spconv-1**（indice_key 共享）：同一个 `SparseConv3DBlock` 内 3 个 SubMConv3d 共享 indice_key，
   复用 hash table（rulebook），避免 forward/backward 重复构建。代码改动在
   [model/encoder/gaussian_encoder/spconv3d_module.py](model/encoder/gaussian_encoder/spconv3d_module.py)。
   commit f4e6759。**已保留**。
2. **Opt-spconv-2**（large_kernel_fast_algo=True）：把 algo 自动选择的 kv 上限从 32 提到 128。
   我们 kernel_size=5 → kv=125，原本 fallback 到 `ConvAlgo.Native`（最慢）；
   开启后会选 `ConvAlgo.MaskImplicitGemm`（融合算子，3D 大 kernel 理论更快）。
   commit a419f8d。**已回滚**（commit dfb65f6）。

#### 失败过程

| 尝试 | 结果 |
|------|------|
| spconv-cu118 2.3.6 + `large_kernel_fast_algo=True` | ❌ NVRTC 编译 cutlass kernel 失败：`this arch isn't supported`。原因：cu118 prebuilt 不含 sm_90 cutlass kernel |
| 升级 spconv-cu120 2.3.6 + cumm-cu120 0.4.11 | ✅ 安装成功；单 SubMConv 烟测 forward+backward OK，`conv.algo = MaskImplicitGemm` |
| 启动 7 卡 DDP 训练 | ❌ 卡死：GPU 100% 占用 30+ 分钟无任何 iter 输出，全程刷"Can't find algo Simt_xxx in prebuilt. compile with nvrtc..." |

#### 根因

spconv-cu120 在 H20 (sm_90) 上的 prebuilt kernel **不含 cutlass**，会 fallback 到 **Simt**（CUDA Core，非 Tensor Core）+ nvrtc 即时编译。
关键问题：
- nvrtc 编译每个 kernel 5-30 秒，每个 SubMConv 形状要编译 24-32 个变体
- 12 次 SubMConv × 多种 batch shape → 编译规模超过 1 小时
- cumm cache 在 sm_90 上可能不持久化，重复刷"Can't find algo"

**Simt 算法本质上是 CUDA Core 实现，跟 Native fallback 性能相当甚至更差。**
即使编译完成，实测速度也不会比 Native 快。

#### 回滚结果

回滚到 spconv-cu118 2.3.6 + 移除 `large_kernel_fast_algo=True`，**保留 Opt-spconv-1**（indice_key 共享）。
实测 Iter 0→50：155 秒 → **3.10 s/iter**，与原 baseline 完全一致（Opt-spconv-1 单独对总时间无可观测影响，因为 ms 级节省被 spconv backward 4400ms 淹没）。

cu118 spconv wheel 备份：`/data/chenz/spconv_backup/spconv_cu118-2.3.6-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`。

#### 经验教训

1. **H20 (sm_90 Hopper) 是 spconv 的盲区**：spconv 团队目前对 sm_90 的预编译 kernel 不完整，cu118/cu120 都不能跑 cutlass 路径
2. **不要再尝试 large_kernel_fast_algo=True**：除非 spconv 官方发布 sm_90 完整 cutlass kernel
3. **真正可行的下一步**：换路径，不再纠结 spconv 本身（见下节）

---

### 未来提速路线（已分析，按优先级排序）

#### ❶ ~~升级 spconv~~（**2026-05-20 实测失败，已删除该项**）


#### ❷ 升级 PyTorch（配合 ❶，支持 expandable_segments 内存分配策略）

当前 faster 环境是 Python 3.8 + 旧版 PyTorch，**不支持** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。
该选项在新版 PyTorch（≥2.0）中能显著减少 CUDA OOM 风险并提升显存复用率。

配合 ❶ 换 spconv 同时升级 PyTorch 到 2.0+，一次性解决两个问题。

#### ❸ torch.compile(backbone)（中等优先级，预期 5-15%）

在 with_cp=False（方案 A 已关闭）的前提下，对 ResNet+FPN backbone 做 torch.compile。

**方法**：
```python
# model/segmentor/bev_segmentor.py
self.img_backbone = torch.compile(self.img_backbone, mode='reduce-overhead')
self.img_neck = torch.compile(self.img_neck, mode='reduce-overhead')
```

**风险**：DCNv2 算子可能触发 graph break，导致 compile 收益打折；第一个 epoch 有编译开销（~5min）。
建议先在单卡跑 5 iter 验证无误差后再开多卡。

**前置条件**：需要 PyTorch ≥ 2.0（当前 Python 3.8 环境不满足，依赖 ❷）。

#### ❹ 减少 anchor 数量（低侵入性，预期 10-20%）

spconv backbone 处理的点数正比于 anchor 数量（当前 3600）。减半到 1800 可直接减少 spconv 计算量，
同时内存申请次数也减少，对 cudaFree/cudaMalloc 瓶颈也有缓解。

**代价**：精度可能下降，需要实验验证。

#### ❺ 数据 prefetch 优化（低优先级，预期 5-10%）

当前 DataLoader 在 CPU 预处理伪标签时可能存在等待，可通过：
- 增大 `num_workers`（当前值需检查）
- 对 pseudo_depth/pseudo_seg 的下采样用 GPU 做（移到 collate_fn 之后）

| 方案 | 预期提速 | 需改 conda 环境 | 风险 |
|------|---------|----------------|------|
| ❶ 升级 spconv 2.3+ | ~60% | ✅ 新建环境 | 中（API 兼容） |
| ❷ 升级 PyTorch ≥2.0 | 支撑 ❶❸ | ✅ 新建环境 | 低 |
| ❸ torch.compile | ~10% | ✅（依赖 ❷） | 中（graph break） |
| ❹ 减少 anchor 数量 | ~15% | ❌ 不需要 | 中（精度影响） |
| ❺ DataLoader prefetch | ~5% | ❌ 不需要 | 低 |

---

## 版本记录

| 日期 | 更新内容 |
|------|----------|
| 2026-05-09 | 初版，记录整体方案和实现步骤 |
| 2026-05-09 | 整合代码探索结论：确认渲染系统细节、伪标签格式、dataset key 结构，更新待解决问题清单 |
| 2026-05-09 | 路线切换：放弃 3D Lifting，确定采用 2D Splatting（gsplat 可微渲染）方案，重写实现计划 |
| 2026-05-09 | 确定 splatting 分支训练策略：occ+flow+det+render，去掉 map/plan，渲染 0.44×（可配置） |
| 2026-05-11 | 记录 splatting 分支首次启动的三个关键 Bug 及修复方法；纠正 conda activate 错误说明 |
| 2026-05-14 | 标记所有已确认细节为完成；新增 Loss 详解（DetectionLoss/RenderLoss）；新增 RenderLoss 有效性诊断四步法；新增可视化模块（commit 7fc218c）；补充接续训练说明 |
| 2026-05-15 | **全量 loss 启用**（map+plan 加回）；发现语义渲染不学习的根因（softplus→alpha-blend≠logits）；修复为 Plan A（渲染 raw logits）；训练改为 tmux `train_splatting` |
| 2026-05-15 | 训练扩展为 **8 卡**（GPU 0-7），3516 iters/epoch |
| 2026-05-18 | **发现并修复 RenderLoss off-by-one 类别索引 bug**：CE target 错位导致所有类梯度方向错误，bicycle 18 epoch 全 0%；修复 commit eb138cf；同步修复可视化 palette 映射；max_epochs 延长至 30（commit b19c429），接续 epoch 18 继续训练 |
| 2026-05-18 | **PKL 转换脚本 v6**：P0 修复（scene_token、num_lidar_pts 真值回填、velocity 坐标系重算）+ P1 质量改进（VAD 命令阈值、自适应匹配、ego_lcf_feat 全维度、map 线长过滤 2m）；新增体检脚本 `tools/stats_gaussianad_pkl.py` |
| 2026-05-19 | **创建 faster 分支**：基于 splatting，克隆 conda 环境为 `faster`；实现 Opt-1（channels_last）、Opt-2（lidar2global float32 预转换）、Opt-3（nonzero→bool mask）三项 GPU 加速优化（commit c29ca34）；分析 Opt-4/5 风险，暂不实施 |
| 2026-05-20 | **实测 A+B+C 方案**：with_cp=False（Opt-A）+ DDP bucket=200MB（Opt-B）+ max_split_size_mb（Opt-C）合计仅 ~3% 提速（3.17→3.10 s/iter）；确认真正瓶颈为 spconv SubMConv backward 的 cudaFree/cudaMalloc（4400ms/iter）；训练正在以 ~3.10 s/iter 运行；后续提速需升级 spconv 2.3+（commit 36c229c） |
| 2026-05-20 | **spconv 提速实验失败**：尝试 Opt-spconv-2（`large_kernel_fast_algo=True` → MaskImplicitGemm）；cu118 nvrtc 编译 cutlass kernel 报 "this arch isn't supported"；升级到 spconv-cu120 2.3.6 仍只有 Simt fallback，7 卡训练卡死 30+ 分钟全在 nvrtc 编译。结论：**H20 sm_90 上 spconv 没有可行的提速路径**。回滚 spconv-cu118 + 移除 large_kernel_fast_algo，保留 Opt-spconv-1（indice_key 共享，纯 Python 改动）。训练恢复 3.10 s/iter（commit dfb65f6） |

