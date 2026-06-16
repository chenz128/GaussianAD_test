import torch
import torch.nn as nn
import torch.nn.functional as F
from gsplat import rasterization


class GaussianRasterizer2D(nn.Module):
    """
    2D Gaussian splatting renderer for pseudo-label supervision.
    Only used during training. Renders 3D Gaussians to camera planes
    and computes semantic + depth losses against pseudo labels.
    """

    def __init__(self, render_h, render_w, sem_lw=2.0, depth_lw=0.05):
        super().__init__()
        self.height = render_h
        self.width = render_w
        self.sem_lw = sem_lw
        self.depth_lw = depth_lw

        # dynamic classes (depth loss unreliable for moving objects)
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

    def render(self, means, quats, scales, opacities, semantics, gs_extrins, gs_intrins,
               dynamic=None):
        """
        Render 3D Gaussians to all cameras for one batch element.

        Args:
            means:      (G, 3)
            quats:      (G, 4)
            scales:     (G, 3)
            opacities:  (G,)
            semantics:  (G, 17)
            gs_extrins: (nC, 4, 4)  ego2cam
            gs_intrins: (nC, 3, 3)  render intrinsics
            dynamic:    (G, 1) or None  raw dynamic logit per gaussian

        Returns:
            rendered_sem:     (nC, H, W, 17)
            rendered_depth:   (nC, H, W)
            rendered_dynamic: (nC, H, W) or None
        """
        n_sem = semantics.shape[-1]
        if dynamic is not None:
            colors = torch.cat([semantics, dynamic], dim=-1)  # (G, 17+1)
        else:
            colors = semantics
        # rasterize all cameras at once
        rendered, _, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=gs_extrins,
            Ks=gs_intrins,
            width=self.width,
            height=self.height,
            render_mode='RGB+D',
        )
        # rendered: (nC, H, W, C+1) — first n_sem channels semantics, then optional
        # dynamic channel, last channel is depth
        rendered_sem = rendered[..., :n_sem]            # (nC, H, W, 17)
        rendered_depth = rendered[..., -1]              # (nC, H, W)
        if dynamic is not None:
            rendered_dynamic = rendered[..., n_sem]     # (nC, H, W)
        else:
            rendered_dynamic = None
        return rendered_sem, rendered_depth, rendered_dynamic

    def render_dynamic_only(self, means, quats, scales, opacities, dynamic, gs_extrins, gs_intrins):
        """Render ONLY the dynamic logit for all cameras of one batch element.

        Single dynamic color channel (depth channel from RGB+D is discarded),
        much cheaper than the full 17-ch semantic render. Used for multi-frame
        (history/future) dynamic/static supervision.

        Args:
            dynamic: (G, 1) raw per-gaussian dynamic logit
        Returns:
            rendered_dynamic: (nC, H, W)
        """
        rendered, _, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=dynamic,                 # (G, 1)
            viewmats=gs_extrins,
            Ks=gs_intrins,
            width=self.width,
            height=self.height,
            render_mode='RGB+D',
        )
        # rendered: (nC, H, W, 2) — channel 0 dynamic logit, channel 1 depth
        return rendered[..., 0]  # (nC, H, W)

    def forward(self, gaussian, gs_extrins, gs_intrins):
        """
        Args:
            gaussian:    GaussianPrediction namedtuple
            gs_extrins:  (B, nC, 4, 4)
            gs_intrins:  (B, nC, 3, 3)
        Returns:
            rendered_sem:     (B, nC, H, W, 17)
            rendered_depth:   (B, nC, H, W)
            rendered_dynamic: (B, nC, H, W) or None
        """
        B = gaussian.means.shape[0]
        has_dyn = getattr(gaussian, 'dynamic_logits', None) is not None
        all_sem, all_depth, all_dyn = [], [], []
        for b in range(B):
            dyn_b = gaussian.dynamic_logits[b] if has_dyn else None
            sem_b, depth_b, rdyn_b = self.render(
                gaussian.means[b],
                gaussian.rotations[b],
                gaussian.scales[b],
                gaussian.opacities[b, :, 0],
                gaussian.semantics_logits[b],  # use raw logits for proper CE loss
                gs_extrins[b],
                gs_intrins[b],
                dynamic=dyn_b,
            )
            all_sem.append(sem_b)
            all_depth.append(depth_b)
            if rdyn_b is not None:
                all_dyn.append(rdyn_b)
        rendered_dynamic = torch.stack(all_dyn) if all_dyn else None
        return torch.stack(all_sem), torch.stack(all_depth), rendered_dynamic

    def compute_loss(self, rendered_sem, rendered_depth, pseudo_seg, pseudo_depth):
        """
        Args:
            rendered_sem:   (B, nC, H, W, 17)
            rendered_depth: (B, nC, H, W)
            pseudo_seg:     (B, nC, H, W)  int, 0=invalid/sky
            pseudo_depth:   (B, nC, H, W)  float, 0=invalid
        Returns:
            loss_sem, loss_depth
        """
        # ── semantic loss ──
        pred_sem = rendered_sem.flatten(0, -2)    # (N, 17)
        target_sem = pseudo_seg.flatten().long()   # (N,)
        valid_sem = target_sem > 0
        if valid_sem.any():
            pw = self.class_weight[target_sem[valid_sem]]
            # NOTE: pseudo_seg labels 1-16 map directly to Gaussian channels 1-16 (no -1)
            loss_sem = self.sem_lw * (
                pw * self.loss_fn_ce(pred_sem[valid_sem], target_sem[valid_sem])
            ).mean()
        else:
            loss_sem = pred_sem.sum() * 0.0

        # ── depth loss ──
        # NOTE: this compute_loss method is dead code — training uses RenderLoss in
        # loss/render_loss.py instead. Kept for reference only.
        pred_d = rendered_depth.flatten()
        target_d = pseudo_depth.flatten()
        dyn_mask = torch.isin(
            pseudo_seg.flatten(),
            self.dynamic_classes.to(pseudo_seg.device)
        )
        # pred_d > 0: exclude pixels with no Gaussian coverage (rendered_depth=0)
        valid_d = (target_d > 0.5) & (pred_d.detach() > 0) & ~dyn_mask
        if valid_d.any():
            loss_depth = self.depth_lw * self.loss_fn_depth(pred_d[valid_d], target_d[valid_d])
        else:
            loss_depth = pred_d.sum() * 0.0

        return loss_sem, loss_depth
