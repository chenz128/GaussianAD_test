import logging
import torch
import torch.nn as nn
from . import OPENOCC_LOSS
from .base_loss import BaseLoss

try:
    from model.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu
except Exception:  # pragma: no cover - only needed when use_gt_box=True
    points_in_boxes_gpu = None


@OPENOCC_LOSS.register_module()
class PhysicsLoss(BaseLoss):
    """
    Physical priors on Gaussian motion via offset predictions:
    1. Static: static gaussians should have zero offset (no motion)
    2. Smoothness: dynamic gaussians should have small acceleration (constant velocity)
    3. (optional) Rigid: gaussians inside the same GT box should share the same
       offset (rigid-body / pure translation) -> penalize per-box offset variance.

    When ``use_gt_box=True``, the dynamic/static gating no longer relies on the
    (potentially noisy) predicted ``dynamic_logits``. Instead it uses an oracle
    membership derived from GT boxes: a gaussian is dynamic iff it falls inside a
    GT box whose speed |v| > ``v_thresh``. This measures the *upper bound* of the
    dynamic-separation scheme with clean labels. GT boxes are only available at
    training time, so this remains a diagnostic oracle.

    Requires:
        offset:          (B, G, 6, 2) predicted future displacements (xy)
        dynamic_logits:  (B, G, 1) raw dynamic/static logit per gaussian
        gaussian:        GaussianPrediction (uses .means (B, G, 3)) -- gt-box mode
        gt_boxes:        (B, T, >=9) [x,y,z,dx,dy,dz,heading, vx,vy, ...] -- gt-box mode
    """

    def __init__(
        self,
        static_w=5.0,
        smooth_w=50.0,
        rigid_w=0.0,
        vel_w=0.0,
        warmup_epoch=2,
        dyn_threshold=0.5,
        use_gt_box=False,
        v_thresh=0.5,
        z_margin=0.2,
        use_gt_semantic_gate=False,
        movable_classes=None,
        sem_gate_max_dist=0.5,
        num_sem_classes=18,
        weight=1.0,
        input_dict=None,
        **kwargs,
    ):
        if input_dict is None:
            input_dict = {
                'offset': 'offset',
                'dynamic_logits': 'dynamic_logits',
                'current_epoch': 'current_epoch',
            }
            if use_gt_box:
                input_dict['gaussian'] = 'gaussian'
                input_dict['gt_boxes'] = 'gt_boxes'
                input_dict['ego_fut_trajs'] = 'ego_fut_trajs'
                if use_gt_semantic_gate:
                    # per-gaussian GT occ labels for the semantic gate
                    input_dict['sampled_xyz'] = 'sampled_xyz'
                    input_dict['sampled_label'] = 'sampled_label'
        super().__init__(weight=weight, input_dict=input_dict, **kwargs)
        # BaseLoss.__init__ sets self.loss_func as instance attr, shadowing method
        if hasattr(self, 'loss_func'):
            del self.loss_func

        self.static_w = static_w
        self.smooth_w = smooth_w
        self.rigid_w = rigid_w
        # Positive motion supervision: dynamic gaussians should move at their GT
        # box velocity (offset[t] ~= v_box * (t+1) * dt). This is the only term
        # that actively DRIVES motion; the others only suppress. 0 -> disabled.
        self.vel_w = vel_w
        self.warmup_epoch = warmup_epoch
        self.dyn_threshold = dyn_threshold
        self.use_gt_box = use_gt_box
        self.v_thresh = v_thresh
        # Ground gate: a moving box's floor slice also encloses static ground
        # voxels. Only the thin bottom layer (< z_margin above the box floor) is
        # forced static; the object body is untouched. Kept small so genuine
        # dynamic gaussians are never dropped. 0 -> gate disabled.
        self.z_margin = z_margin
        # GT semantic gate: of the gaussians geometrically inside a moving box,
        # keep as dynamic only those whose nearest occ-GT label is a movable
        # class. Uses clean GT labels (sampled_label) -> no circular dependency
        # on the model's own predicted semantics.
        self.use_gt_semantic_gate = use_gt_semantic_gate
        self.sem_gate_max_dist = sem_gate_max_dist
        if use_gt_semantic_gate:
            assert movable_classes is not None, \
                'movable_classes required when use_gt_semantic_gate=True'
            lut = torch.zeros(num_sem_classes, dtype=torch.bool)
            lut[list(movable_classes)] = True
            self.register_buffer('movable_lut', lut)
        if use_gt_box:
            assert points_in_boxes_gpu is not None, \
                'points_in_boxes_gpu unavailable but use_gt_box=True'
        self._diag_counter = 0

    @torch.no_grad()
    def _gt_box_membership(self, means, gt_boxes,
                           sampled_xyz=None, sampled_label=None):
        """Assign each gaussian to a GT box and derive a dynamic mask.

        Args:
            means:     (B, G, 3) gaussian centers in LIDAR frame
            gt_boxes:  (B, T, >=9) padded GT boxes (pad rows are all-zero, i.e.
                       zero-size boxes that never contain any point)
            sampled_xyz:   (B, N, 3) occ-GT sample points (same frame as means)
            sampled_label: (B, N) occ-GT semantic labels for the GT semantic gate
        Returns:
            box_idx:  (B, G) long, index of the containing box (-1 = background)
            dyn_mask: (B, G) bool, True if inside a moving box (|v| > v_thresh),
                      after optional ground / GT-semantic gating.
        """
        means = means.detach().float().contiguous()
        gt_boxes = gt_boxes.to(means.device).float()
        boxes7 = gt_boxes[..., :7].contiguous()  # (B, T, 7)
        # points_in_boxes_gpu: points (B, M, 3), boxes (B, T, 7) -> (B, M)
        box_idx = points_in_boxes_gpu(means, boxes7).long()  # (B, G), -1 bg

        # per-box speed |v| from columns [7:9]
        speed = torch.linalg.norm(gt_boxes[..., 7:9], dim=-1)  # (B, T)
        moving_box = speed > self.v_thresh                     # (B, T) bool

        B, G = box_idx.shape
        dyn_mask = torch.zeros((B, G), dtype=torch.bool, device=means.device)
        for b in range(B):
            valid = box_idx[b] >= 0
            if valid.any():
                dyn_mask[b, valid] = moving_box[b][box_idx[b, valid]]
        # ground gate: heading is yaw-only (rotation about z), so z needs no
        # un-rotation. Force gaussians in the box's bottom slice (ground) static.
        if self.z_margin > 0:
            box_bottom = gt_boxes[..., 2] - 0.5 * gt_boxes[..., 5]  # (B, T)
            for b in range(B):
                sel = box_idx[b] >= 0
                if not sel.any():
                    continue
                bi = box_idx[b].clamp_min(0)                       # (G,)
                h_above = means[b, :, 2] - box_bottom[b][bi]       # (G,)
                ground = sel & (h_above < self.z_margin)
                dyn_mask[b, ground] = False
        # GT semantic gate: keep dynamic only where the nearest occ-GT label is
        # a movable class. Only checks currently-dynamic gaussians (cheap). A
        # gaussian is forced static if its nearest GT sample is a non-movable
        # class OR is farther than sem_gate_max_dist (label unreliable).
        if (self.use_gt_semantic_gate and sampled_xyz is not None
                and sampled_label is not None):
            sx = sampled_xyz.to(means.device).float()
            sl = sampled_label.to(means.device).long()
            if sx.dim() == 2:
                sx = sx[None]
            if sl.dim() == 1:
                sl = sl[None]
            lut = self.movable_lut.to(means.device)
            for b in range(B):
                sel = dyn_mask[b]
                if not sel.any():
                    continue
                idx = sel.nonzero(as_tuple=False).squeeze(1)      # (k,)
                dist = torch.cdist(means[b, idx], sx[b])          # (k, N)
                nn_dist, nn_idx = dist.min(dim=1)                 # (k,)
                nn_lbl = sl[b][nn_idx]                            # (k,)
                movable = torch.zeros_like(nn_lbl, dtype=torch.bool)
                ok = (nn_lbl >= 0) & (nn_lbl < lut.numel())
                movable[ok] = lut[nn_lbl[ok]]
                drop = (~movable) | (nn_dist > self.sem_gate_max_dist)
                dyn_mask[b, idx[drop]] = False
        return box_idx, dyn_mask

    def forward(self, inputs):
        actual_inputs = {}
        for input_key, input_val in self.input_dict.items():
            actual_inputs[input_key] = inputs.get(input_val)
        loss = self.loss_func(**actual_inputs)
        return self.weight * loss, {
            'PhysicsLoss': (self.weight * loss).detach().item(),
        }

    def loss_func(self, offset, dynamic_logits, current_epoch=None,
                  gaussian=None, gt_boxes=None, ego_fut_trajs=None,
                  sampled_xyz=None, sampled_label=None):
        """
        Args:
            offset:          (B, G, 6, 2) or flat tensor needing reshape
            dynamic_logits:  (B, G, 1) raw logit, or None (eval mode)
            current_epoch:   int or None
            gaussian:        GaussianPrediction (gt-box mode), provides .means
            gt_boxes:        (B, T, >=9) padded GT boxes (gt-box mode)
            ego_fut_trajs:   (B, 6, 2) GT ego per-step displacements (LIDAR frame)
            sampled_xyz:     (B, N, 3) occ-GT points (GT semantic gate)
            sampled_label:   (B, N) occ-GT labels (GT semantic gate)
        """
        if offset is None or dynamic_logits is None:
            return torch.tensor(0.0, requires_grad=False)

        # Reshape offset if needed
        if offset.dim() == 2:
            # (G, 12) -> (1, G, 6, 2)
            G = offset.shape[0]
            offset = offset.reshape(1, G, 6, 2)
        elif offset.dim() == 3:
            # (B, G, 12) -> (B, G, 6, 2)
            offset = offset.reshape(offset.shape[0], offset.shape[1], 6, 2)

        # dynamic_logits: ensure (B, G)
        if dynamic_logits.dim() == 3:
            dynamic_logits = dynamic_logits[..., 0]  # (B, G, 1) -> (B, G)

        # ====== Dynamic/static gating ======
        # gt-box mode: oracle membership from GT boxes (clean labels).
        # fallback  : predicted dynamic_logits (sigmoid soft gate).
        box_idx = None
        gt_dyn_mask = None
        if self.use_gt_box and gaussian is not None and gt_boxes is not None:
            box_idx, gt_dyn_mask = self._gt_box_membership(
                gaussian.means, gt_boxes, sampled_xyz, sampled_label)
            dyn_gate = gt_dyn_mask.float()               # (B, G) hard {0,1}
            static_gate = 1.0 - dyn_gate
        else:
            p_dyn = torch.sigmoid(dynamic_logits)        # (B, G) in [0, 1]
            dyn_gate = p_dyn
            static_gate = 1.0 - p_dyn

        # ====== Static constraint: static gaussians should not move ======
        static_mask = static_gate.unsqueeze(-1).unsqueeze(-1)  # (B, G, 1, 1)
        loss_static = self.static_w * (static_mask * offset.pow(2)).mean()

        # ====== Smoothness constraint: dynamic accel should be small ======
        velocity = offset[..., 1:, :] - offset[..., :-1, :]  # (B, G, 5, 2)
        acceleration = velocity[..., 1:, :] - velocity[..., :-1, :]  # (B, G, 4, 2)
        dyn_mask = dyn_gate.unsqueeze(-1).unsqueeze(-1)  # (B, G, 1, 1)
        loss_smooth = self.smooth_w * (dyn_mask * acceleration.pow(2)).mean()

        # ====== Rigid constraint: same-box gaussians share offset ======
        # Only meaningful with GT-box instance membership. Penalize the variance
        # of offset within each moving box (pure-translation rigid prior).
        loss_rigid = offset.new_tensor(0.0)
        if self.rigid_w > 0 and box_idx is not None:
            loss_rigid = self.rigid_w * self._rigid_variance(offset, box_idx, gt_dyn_mask)

        # ====== Velocity constraint: dynamic gaussians move at GT box velocity =
        # The only POSITIVE driver of motion: offset[t] ~= v_box * (t+1) * dt.
        # Grad flows to offset; the target is a GT-derived constant.
        # During warmup, skip loss_vel ENTIRELY (don't build it) so early epochs
        # match the vel-off baseline exactly; static_graph=False tolerates the
        # graph appearing later at epoch>=warmup_epoch.
        in_warmup = (current_epoch is not None
                     and current_epoch < self.warmup_epoch)
        loss_vel = offset.new_tensor(0.0)
        if (self.vel_w > 0 and not in_warmup and box_idx is not None
                and gt_boxes is not None and gt_dyn_mask is not None
                and gt_boxes.shape[1] > 0   # guard: empty-box batch crashes gather
                and gt_dyn_mask.any()):      # guard: no dynamic gaussians → skip
            loss_vel = self.vel_w * self._velocity_target(
                offset, box_idx, gt_dyn_mask, gt_boxes)

        total = loss_static + loss_smooth + loss_rigid + loss_vel

        # Warmup: also zero the still-active static/smooth/rigid terms.
        if in_warmup:
            total = total * 0.0

        # Diagnostics
        self._diag_counter += 1
        if self._diag_counter % 500 == 1:
            with torch.no_grad():
                if gt_dyn_mask is not None:
                    n_dyn = gt_dyn_mask.sum().item()
                    n_static = gt_dyn_mask.numel() - n_dyn
                else:
                    _p = torch.sigmoid(dynamic_logits)
                    n_static = (_p < self.dyn_threshold).sum().item()
                    n_dyn = (_p >= self.dyn_threshold).sum().item()
                off_mag = offset.pow(2).sum(-1).sqrt().mean().item()
                if gt_dyn_mask is not None and gt_dyn_mask.any():
                    off_dyn = offset.pow(2).sum(-1).sqrt()[gt_dyn_mask].mean().item()
                else:
                    off_dyn = 0.0
            _msg = (
                f'[PhysicsLoss Diag] iter={self._diag_counter} | '
                f'gt_box={self.use_gt_box} static={n_static} dyn={n_dyn} | '
                f'offset_rms={off_mag:.4f} offset_dyn_rms={off_dyn:.4f} | '
                f'loss_static={loss_static.item():.4f} '
                f'loss_smooth={loss_smooth.item():.4f} '
                f'loss_rigid={loss_rigid.item():.4f} '
                f'loss_vel={loss_vel.item():.4f} '
                f'total={total.item():.4f}'
            )
            if (not torch.distributed.is_available()
                    or not torch.distributed.is_initialized()
                    or torch.distributed.get_rank() == 0):
                print(_msg, flush=True)

        return total

    def _velocity_target(self, offset, box_idx, dyn_mask, gt_boxes,
                         ego_fut_trajs=None):
        """MSE(offset, ego-compensated GT-box trajectory), masked to dynamic gaussians.

        Corrected target per gaussian g at step t:
            target[b,g,t] = v_box*(t+1)*dt - ego_cumdisp[b,t]
        where ego_cumdisp = cumsum(ego_fut_trajs) gives the GT ego cumulative
        displacement in the LIDAR frame (same frame as offset and v_box).
        Subtracting ego displacement converts from world-relative object velocity
        to ego-relative: a static object (v_box=0) will have offset=-ego_disp.

        Uses a MULTIPLICATIVE mask over the FULL offset tensor so the autograd
        graph is identical every iteration.
        """
        with torch.no_grad():
            B, G = box_idx.shape
            dt = 0.5
            gt_boxes = gt_boxes.to(offset.device).float()
            tmul = (torch.arange(6, device=offset.device).float() + 1.0) * dt  # (6,)
            bi = box_idx.clamp(0, gt_boxes.shape[1] - 1)                      # (B, G) in-bounds
            v_box = torch.gather(
                gt_boxes[..., 7:9], 1, bi.unsqueeze(-1).expand(B, G, 2))       # (B, G, 2)
            # Object displacement target (world frame, LIDAR coords)
            target = v_box[:, :, None, :] * tmul[None, None, :, None]          # (B,G,6,2)
            # Ego compensation: subtract GT cumulative ego displacement so that
            # offset is supervised in the EGO-RELATIVE frame.
            # ego_fut_trajs: (B, 6, 2) per-step → cumsum gives cumulative disp.
            if ego_fut_trajs is not None:
                ego = ego_fut_trajs.to(offset.device).float()                  # (B, 6, 2)
                ego_cumdisp = ego.cumsum(dim=1)                                # (B, 6, 2)
                target = target - ego_cumdisp[:, None, :, :]                   # broadcast (B,G,6,2)
            w = dyn_mask.float()[:, :, None]                                   # (B,G,1) const
            denom = w.sum() * offset.shape[2] + 1e-6
        sq_err = (offset - target).pow(2).sum(-1)                             # (B,G,6) grad→offset
        return (w * sq_err).sum() / denom

    def _rigid_variance(self, offset, box_idx, dyn_mask):
        """Mean per-box variance of offset over gaussians inside each moving box.

        Args:
            offset:   (B, G, 6, 2) predicted displacements (grad flows here)
            box_idx:  (B, G) long, containing-box index (-1 background)
            dyn_mask: (B, G) bool, True for gaussians inside a moving box
        Returns:
            scalar tensor = mean over moving boxes of mean_t ||offset - mean||^2
        """
        B = offset.shape[0]
        terms = []
        for b in range(B):
            idx_b = box_idx[b]                       # (G,)
            dyn_b = dyn_mask[b] if dyn_mask is not None else (idx_b >= 0)
            valid = dyn_b & (idx_b >= 0)
            if not valid.any():
                continue
            box_ids = torch.unique(idx_b[valid])
            for bid in box_ids.tolist():
                sel = valid & (idx_b == bid)
                if sel.sum() < 2:
                    continue  # variance undefined for a single gaussian
                off_sel = offset[b][sel]             # (n, 6, 2)
                mean_off = off_sel.mean(dim=0, keepdim=True)  # (1, 6, 2)
                var = (off_sel - mean_off).pow(2).mean()
                terms.append(var)
        if len(terms) == 0:
            return offset.new_tensor(0.0)
        return torch.stack(terms).mean()
