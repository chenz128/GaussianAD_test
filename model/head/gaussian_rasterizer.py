import torch
import torch.nn as nn
import torch.nn.functional as F
from gsplat import  rasterization


class GaussianRasterizer2D(nn.Module):
    """
    2D Gaussian splatting renderer for pseudo-label supervision.
    Only used during training. Renders 3D Gaussians to camera planes
    and computes semantic + depth losses against pseudo labels.
    """

    def __init__(self, render_h, render_w, sem_lw=2.0, depth_lw=0.05, detach_shape=False):
        super().__init__()
        self.height = render_h
        self.width = render_w
        self.sem_lw = sem_lw
        self.depth_lw = depth_lw
        self.detach_shape = detach_shape

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

    def render(self, means, quats, scales, opacities, semantics, gs_extrins, gs_intrins):
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

        Returns:
            rendered_sem:   (nC, H, W, 17)
            rendered_depth: (nC, H, W)   accumulated depth D = sum T_i a_i z_i
            rendered_acc:   (nC, H, W)   accumulation map A = 1 - T_final
            rendered_var:   (nC, H, W)   per-ray depth variance Var[z] (concentration ①)
        """
        # rasterize all cameras at once.
        # render_mode='RGB+D' -> ACCUMULATED depth D = sum T_i a_i z_i (NOT
        # alpha-normalized). This preserves the implicit opacity floor: low
        # opacity -> small D -> depth loss pushes opacity up. (RGB+ED / A4 removed
        # this floor and collapsed opacity; reverted.)
        # 2nd return value (render_alphas) is the per-pixel accumulation map A.
        rendered, rendered_alpha, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=semantics,
            viewmats=gs_extrins,
            Ks=gs_intrins,
            width=self.width,
            height=self.height,
            render_mode='RGB+D',
        )
        # rendered: (nC, H, W, 18) — first 17 channels are semantics, last is depth
        rendered_sem = rendered[..., :17]    # (nC, H, W, 17)
        rendered_depth = rendered[..., 17]   # (nC, H, W) accumulated depth
        rendered_acc = rendered_alpha[..., 0]  # (nC, H, W)

        # ── ① depth concentration: render per-ray second moment of depth ──
        # For each camera, render z^2 (camera-space depth squared) as a single
        # color channel. With RGB+D this yields:
        #   z2_acc = sum T_i a_i z_i^2   (color channel)
        #   d_acc  = sum T_i a_i z_i     (D channel)
        # Then E[z]=d_acc/A, E[z^2]=z2_acc/A, Var[z]=E[z^2]-E[z]^2.
        # Var is per-camera (z_i depends on the camera), so we loop cameras.
        rendered_var = self._render_depth_variance(
            means, quats, scales, opacities, gs_extrins, gs_intrins, rendered_acc)
        return rendered_sem, rendered_depth, rendered_acc, rendered_var

    def _render_depth_variance(self, means, quats, scales, opacities,
                               gs_extrins, gs_intrins, rendered_acc, eps=1e-4):
        """Per-ray depth variance Var[z] for the concentration loss ①.

        Var[z] = E[z^2] - E[z]^2 where the expectations are alpha-normalized
        along each ray. A high variance means Gaussians are smeared along the
        ray (a foggy slab); minimizing it forces them to collapse onto a single
        sharp surface depth. z^2 is camera-dependent, so render per camera.
        """
        nC = gs_extrins.shape[0]
        ones = means.new_ones((means.shape[0], 1))
        means_h = torch.cat([means, ones], dim=-1)  # (G, 4)
        var_list = []
        for c in range(nC):
            # camera-space depth z_i = (ego2cam @ [x,y,z,1])_2
            cam_pts = means_h @ gs_extrins[c].transpose(0, 1)  # (G, 4)
            z = cam_pts[:, 2]                                  # (G,)
            z2 = (z * z).unsqueeze(-1)                         # (G, 1)
            out_c, alpha_c, _ = rasterization(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=z2,
                viewmats=gs_extrins[c:c + 1],
                Ks=gs_intrins[c:c + 1],
                width=self.width,
                height=self.height,
                render_mode='RGB+D',
            )
            z2_acc = out_c[0, ..., 0]            # sum T a z^2
            d_acc = out_c[0, ..., 1]             # sum T a z  (D channel)
            A_c = alpha_c[0, ..., 0].clamp_min(eps)
            E_z = d_acc / A_c
            E_z2 = z2_acc / A_c
            var_c = (E_z2 - E_z * E_z).clamp_min(0.0)  # (H, W)
            var_list.append(var_c)
        return torch.stack(var_list, dim=0)  # (nC, H, W)

    def render_depth_only(self, means, quats, scales, opacities, gs_extrins, gs_intrins):
        """Render ONLY depth for all cameras of one batch element.

        Uses a single dummy color channel (depth is color-independent in
        alpha-blending), much cheaper than the 17-channel semantic render.
        Used for multi-frame (history/future) depth supervision.

        Returns:
            rendered_depth: (nC, H, W)
        """
        dummy = means.new_zeros((means.shape[0], 1))  # (G, 1)
        # RGB+D -> accumulated depth (consistent with render(), preserves opacity floor)
        rendered, _, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=dummy,
            viewmats=gs_extrins,
            Ks=gs_intrins,
            width=self.width,
            height=self.height,
            render_mode='RGB+D',
        )
        # rendered: (nC, H, W, 2) — channel 0 dummy color, channel 1 depth
        return rendered[..., 1]  # (nC, H, W)

    def forward(self, gaussian, gs_extrins, gs_intrins):
        """
        Args:
            gaussian:    GaussianPrediction namedtuple
            gs_extrins:  (B, nC, 4, 4)
            gs_intrins:  (B, nC, 3, 3)
        Returns:
            rendered_sem:   (B, nC, H, W, 17)
            rendered_depth: (B, nC, H, W)   accumulated depth
            rendered_acc:   (B, nC, H, W)   accumulation map A
            rendered_var:   (B, nC, H, W)   per-ray depth variance (concentration ①)
        """
        B = gaussian.means.shape[0]
        all_sem, all_depth, all_acc, all_var = [], [], [], []
        for b in range(B):
            # detach_shape: block 2D gradients from flowing into scales/rot/opacity
            # so they are only optimized by 3D OccLoss
            quats_b = gaussian.rotations[b].detach() if self.detach_shape else gaussian.rotations[b]
            scales_b = gaussian.scales[b].detach() if self.detach_shape else gaussian.scales[b]
            opa_b = gaussian.opacities[b, :, 0].detach() if self.detach_shape else gaussian.opacities[b, :, 0]
            sem_b, depth_b, acc_b, var_b = self.render(
                gaussian.means[b],
                quats_b,
                scales_b,
                opa_b,
                gaussian.semantics_logits[b],  # use raw logits for proper CE loss
                gs_extrins[b],
                gs_intrins[b],
            )
            all_sem.append(sem_b)
            all_depth.append(depth_b)
            all_acc.append(acc_b)
            all_var.append(var_b)
        return (torch.stack(all_sem), torch.stack(all_depth),
                torch.stack(all_acc), torch.stack(all_var))

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
