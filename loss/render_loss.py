import torch
import torch.nn as nn
import torch.nn.functional as F
from . import OPENOCC_LOSS
from .base_loss import BaseLoss


@OPENOCC_LOSS.register_module()
class RenderLoss(BaseLoss):
    """
    Pseudo-label supervision loss using 2D Gaussian splatting renders.
    Computes semantic CE loss + depth MSE loss against pseudo labels.
    """

    def __init__(
        self,
        weight=1.0,
        sem_lw=2.0,
        depth_lw=0.05,
        input_dict=None,
        **kwargs,
    ):
        if input_dict is None:
            input_dict = {
                'rendered_sem': 'rendered_sem',
                'rendered_depth': 'rendered_depth',
                'pseudo_seg': 'pseudo_seg',
                'pseudo_depth': 'pseudo_depth',
            }
        super().__init__(weight=weight, input_dict=input_dict, **kwargs)
        # BaseLoss.__init__ sets self.loss_func = lambda: 0 as instance attr,
        # which shadows our loss_func method. Delete it to restore method lookup.
        del self.loss_func

        self.sem_lw = sem_lw
        self.depth_lw = depth_lw

        # dynamic classes — depth loss unreliable for moving objects
        self.dynamic_classes = torch.tensor([2, 3, 4, 5, 6, 7, 9, 10])

        # class frequency weights for balanced CE loss (nuScenes 17 classes)
        nusc_class_freq = torch.tensor([
            944004, 1897170, 152386, 2391677, 16957802, 724139,
            189027, 2074468, 413451, 2384460, 5916653, 175883646,
            4275424, 51393615, 61411620, 105975596, 116424404
        ], dtype=torch.float32)
        log_w = torch.log(nusc_class_freq.sum() / nusc_class_freq)
        self.register_buffer('class_weight', log_w / log_w.mean())

        self.loss_fn_ce = nn.CrossEntropyLoss(reduction='none')
        self.loss_fn_depth = nn.MSELoss()

    def loss_func(self, rendered_sem, rendered_depth, pseudo_seg, pseudo_depth):
        """
        Args:
            rendered_sem:   (B, nC, H, W, 17) — rendered semantic logits
            rendered_depth: (B, nC, H, W)     — rendered depth
            pseudo_seg:     (B, nC, H, W)     — pseudo semantic labels (0=invalid)
            pseudo_depth:   (B, nC, H, W)     — pseudo depth (0=invalid)
        """
        # eval mode: rendering was skipped, or val dataset has no pseudo labels
        if rendered_sem is None or rendered_depth is None or pseudo_seg is None or pseudo_depth is None:
            return torch.tensor(0.0, requires_grad=False)

        # ── semantic loss ──semantic是指渲染结果的语义分割图，pseudo_seg是指根据点云生成的伪标签语义分割图。rendered_sem和pseudo_seg都是(B, nC, H, W)的形状，其中nC是相机数量，H和W是图像的高和宽。semantic loss是计算rendered_sem和pseudo_seg之间的交叉熵损失，注意pseudo_seg中的0表示无效像素，不参与损失计算。
        pred_sem = rendered_sem.flatten(0, -2)     # (N, 17)
        target_sem = pseudo_seg.flatten().long()    # (N,)
        valid_sem = target_sem > 0
        if valid_sem.any():
            pw = self.class_weight[target_sem[valid_sem]]#这是一个权重张量，用于平衡不同类别的损失。class_weight是根据nuScenes数据集中各个类别的频率计算得到的，频率越低的类别权重越高。通过索引target_sem[valid_sem]，我们可以得到每个有效像素对应的类别权重，从而在计算交叉熵损失时给予稀有类别更大的关注。
            loss_sem = self.sem_lw * (
                pw * self.loss_fn_ce(pred_sem[valid_sem], target_sem[valid_sem] - 1)
            ).mean()#注意这里的target_sem[valid_sem] - 1是因为pseudo_seg中的类别标签是从1开始的，而CrossEntropyLoss要求类别标签从0开始，所以需要减1进行调整。
        else:
            loss_sem = pred_sem.sum() * 0.0

        # ── depth loss ──depth是指渲染结果的深度图，pseudo_depth是指根据点云生成的伪标签深度图。rendered_depth和pseudo_depth都是(B, nC, H, W)的形状，其中nC是相机数量，H和W是图像的高和宽。depth loss是计算rendered_depth和pseudo_depth之间的均方误差损失，注意pseudo_depth中的0表示无效像素，不参与损失计算。
        pred_d = rendered_depth.flatten()
        target_d = pseudo_depth.flatten()
        dyn_mask = torch.isin(
            pseudo_seg.flatten(),
            self.dynamic_classes.to(pseudo_seg.device)
        )
        valid_d = (target_d > 0.5) & ~dyn_mask
        if valid_d.any():
            loss_depth = self.depth_lw * self.loss_fn_depth(
                pred_d[valid_d], target_d[valid_d]
            )
        else:
            loss_depth = pred_d.sum() * 0.0

        # ── diagnostics（每500次iter打印一次，方便判断伪标签是否有效）──
        self._diag_counter = getattr(self, '_diag_counter', 0) + 1
        if self._diag_counter % 500 == 1:
            valid_sem_ratio = valid_sem.float().mean().item()
            valid_d_ratio   = valid_d.float().mean().item()
            pred_depth_mean = pred_d[valid_d].mean().item() if valid_d.any() else 0.0
            pred_depth_std  = pred_d[valid_d].std().item()  if valid_d.any() else 0.0
            gt_depth_mean   = target_d[valid_d].mean().item() if valid_d.any() else 0.0
            # 预测语义的熵均值（越低说明越自信，越高说明越接近随机）
            import math
            with torch.no_grad():
                prob = torch.softmax(pred_sem[valid_sem], dim=-1) if valid_sem.any() else None
                pred_entropy = (-( prob * (prob + 1e-8).log()).sum(-1)).mean().item() if prob is not None else float('nan')
                rand_entropy = math.log(17)   # 随机猜测基准 ≈ 2.833
            import logging
            logger = logging.getLogger('mmengine')
            logger.info(
                f'[RenderLoss Diag] iter={self._diag_counter} | '
                f'valid_sem={valid_sem_ratio:.2%} valid_depth={valid_d_ratio:.2%} | '
                f'pred_depth: mean={pred_depth_mean:.2f}m std={pred_depth_std:.2f}m  '
                f'gt_depth_mean={gt_depth_mean:.2f}m | '
                f'sem_entropy={pred_entropy:.3f} (rand={rand_entropy:.3f}) | '
                f'loss_sem={loss_sem.item():.4f} loss_depth={loss_depth.item():.4f}'
            )

        return loss_sem + loss_depth
