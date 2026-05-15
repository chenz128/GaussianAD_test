# GaussianAD 伪标签监督方案

---

## ⚠️ 易错点：分支与 Conda 环境对应关系

**每次修改代码、运行训练前必须核查以下对应关系，如有疏忽请及时提醒！**

| 训练方案 | Git 分支 | Conda 环境 | 说明 |
|----------|----------|------------|------|
| 原始监督（occ + det + map 等标准 loss） | `main` | `/data/chenz/conda_env/GaussianAD` | 当前正在跑的 noplan 训练 |
| 伪标签监督（2D Gaussian Splatting + gsplat） | `splatting` | `/data/chenz/conda_env/splatting` | gsplat 只在 splatting 环境中安装 |

**规则：**
- 代码修改后 push 到对应分支，远端 pull 对应分支再训练
- ⚠️ `/data/chenz/conda_env/` 下的环境**没有** `activate` 脚本，**不能**用 `source activate` 或 `conda activate`
- 必须用完整路径调用：`/data/chenz/conda_env/splatting/bin/python` 和 `/data/chenz/conda_env/splatting/bin/torchrun`
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

| 配置 | with_cp | 显存占用 | 状态 |
|------|---------|----------|------|
| 全量 loss（6个），无 cp | `False` | 67-76 GB / 96 GB | ✅ 安全运行 |
| 仅 occ+flow+det+render，无 cp | `False` | ~54 GB | — |

### 渲染分辨率

- **初始值 0.44×**（396×704），GaussianFlowOcc 已验证
- 做成可配置参数 `pseudo_label_scale`，VRAM 有余量可调高
- H20 96GB 单卡，全量 loss + 0.44× 渲染，显存 67-76 GB，安全

### 阶段规划

- **Phase 1**（当前）：全量 loss（occ+flow+det+map+plan+render）联合训练 → 对比 main 分支 baseline
- **Phase 2**（如 Phase 1 涨点）：去掉标准 loss，仅伪标签监督 → 探测上限

---

## 实现计划

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

- [x] H20 上 splatting 环境已安装 gsplat（训练已正常运行验证）
- [x] `cam_intrinsic` 是 **(6, 4, 4)**，dataset 中读取后取 `[:, :3, :3]` 得到 3×3 内参
- [x] `ego2cam` 在 **dataset.py** 里计算：`np.linalg.inv(cam2ego)`，保存为 `gs_extrins`
- [x] 渲染分辨率确定为 **0.44×**（396×704），已在训练中运行稳定
- [ ] 数据增强（flip）时 pseudo_seg/pseudo_depth 需同步翻转（当前 rand_flip=True，暂未处理）

---

## 待解决问题

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

```bash
# 本地修改后推送
git add .
git commit -m "add pseudo label supervision"
git push origin splatting

# 远程拉取并训练（train.py 自动检测 latest.pth，无需额外参数即可接续）
ssh -p 31256 root@8.130.174.55 "cd /data/chenz/GaussianAD && git pull origin splatting"
# 之后重启训练即可，work-dir 相同则自动接续
```

### 接续训练说明

`train.py` 在启动时自动检测 `{work_dir}/latest.pth`（代码第183行），只要 `--work-dir` 不变，停训后重启**完全自动接续**，无需额外参数：
- 恢复内容：模型权重（strict=False）、optimizer 状态、scheduler 状态、epoch、global_iter
- 运行时状态（如 `_diag_counter`）从 0 重新计数，不影响权重

---

## RenderLoss 有效性诊断

### 当前训练状态（2026-05-14）

- splatting 分支正在跑 Epoch 4，已记录约 395 次 loss
- RenderLoss 均值约 **2.84**，而 $\log(17) \approx 2.833$ 是17类随机猜测时的交叉熵期望值
- ⚠️ **目前渲染语义接近随机分布，尚未收敛**

### 四步诊断法

| 步骤 | 方法 | 判断标准 |
|------|------|----------|
| 1 | 看 `[RenderLoss Diag]` 日志中的 `pred_depth_mean` | 若 ≈ 0 → 相机参数错误，高斯不在视锥内；若 5~30m → 渲染结构正常 |
| 2 | 看 `sem_entropy` | 若接近 2.833（随机基准）→ 语义未收敛；若 < 2.0 → 已学到有效信息 |
| 3 | 查看可视化图片 | `out/nuscenes_gs25600_splatting/render_vis/step_*.jpg`，预测语义/深度与 GT 对比 |
| 4 | 最终：对比 main 分支 mIoU | splatting vs noplan 相同 epoch 的验证集 mIoU |

### 快速查看 main 分支 mIoU 基准

```bash
ssh -p 31256 root@8.130.174.55 "grep mIoU /data/chenz/GaussianAD/out/nuscenes_gs25600_noplan_run.log | tail -5"
```

### 可视化图片格式

每次 `vis_every=500` iter 保存一张 JPEG：
- 6 相机纵向堆叠
- 每行横向拼接：`[预测语义 | GT语义 | 预测深度图 | GT深度图]`
- 深度颜色：黑→蓝→绿→红（近→远），灰色=无效像素

---

## 调试记录（2026-05-11 Splatting 分支首次启动）

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

#### 训练启动方式（tmux）

```bash
# 在 tmux session train_splatting 中运行
cd /data/chenz/GaussianAD && CUDA_VISIBLE_DEVICES=0,1,2,3 /data/chenz/conda_env/splatting/bin/torchrun \
    --nproc_per_node 4 --master_port 12457 \
    train.py --py-config config/nuscenes_gs25600.py --work-dir out/nuscenes_gs25600_splatting --dataset nuscenes
```

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
