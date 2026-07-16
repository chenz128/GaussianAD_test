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
        traj_w=0.0,
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
                if traj_w > 0:
                    # per-instance GT future trajectory (aligned to gt_boxes
                    # rows via the same mask/selected/range_mask filtering).
                    input_dict['attr_labels_planner'] = 'attr_labels_planner'
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
        # Trajectory supervision (v7): instead of extrapolating the current
        # instantaneous box velocity as a straight constant-velocity line
        # (loss_vel), regress each dynamic gaussian's offset onto the instance's
        # REAL future trajectory (gt_agent_fut_trajs, cumulative). This captures
        # acceleration / braking / turning / lane-change that loss_vel cannot.
        # loss_traj is meant to REPLACE loss_vel (set vel_w=0, traj_w>0).
        self.traj_w = traj_w
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
                # CPU cdist: k~50, N=3000. Avoids repeated ~1MB GPU allocs that
                # fragment the CUDA reserved pool and trigger OOM at iter 250.
                m_cpu = means[b, idx].detach().float().cpu()       # (k, 3)
                s_cpu = sx[b].float().cpu()                        # (N, 3)
                dist_cpu = torch.cdist(m_cpu, s_cpu)              # (k, N) on CPU
                nn_dist_cpu, nn_idx_cpu = dist_cpu.min(dim=1)     # (k,) on CPU
                nn_idx = nn_idx_cpu.to(means.device)
                nn_dist = nn_dist_cpu.to(means.device)
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
                  sampled_xyz=None, sampled_label=None,
                  attr_labels_planner=None):
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
            attr_labels_planner: (B, T, 34) per-instance planner labels aligned
                             to gt_boxes rows. [0:12]=fut_traj (6 steps x 2,
                             per-step delta), [12:18]=fut_mask (6). Used by
                             loss_traj (v7).
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
        # loss_vel is ALWAYS computed (graph constant every iter -> static_graph
        # =True compatible). smooth_l1 bounds its gradient. Empty-box / no-dynamic
        # cases handled inside _velocity_target (w=0).
        loss_vel = offset.new_tensor(0.0)
        if (self.vel_w > 0 and box_idx is not None
                and gt_boxes is not None and gt_dyn_mask is not None):
            loss_vel = self.vel_w * self._velocity_target(
                offset, box_idx, gt_dyn_mask, gt_boxes, ego_fut_trajs)

        # ====== Trajectory constraint (v7): dynamic gaussians follow the GT ===
        # instance future trajectory (real cumulative displacement), replacing
        # the constant-velocity straight-line assumption of loss_vel. Also a
        # DRIVING term (zeroed during warmup with vel/rigid). Computed with a
        # multiplicative mask over the full offset tensor -> graph constant
        # every iter (static_graph=True compatible).
        loss_traj = offset.new_tensor(0.0)
        if (self.traj_w > 0 and box_idx is not None
                and gt_dyn_mask is not None
                and attr_labels_planner is not None):
            loss_traj = self.traj_w * self._trajectory_target(
                offset, box_idx, gt_dyn_mask, attr_labels_planner)

        # Plan A selective warmup: during the first ``warmup_epoch`` epochs,
        # keep the SUPPRESSIVE terms (loss_static, loss_smooth) active from
        # epoch 0 while zeroing only the DRIVING terms (loss_rigid, loss_vel).
        #
        # Rationale: previously the ENTIRE physics loss was multiplied by 0
        # during warmup, so offset was left completely unconstrained and grew
        # (via OccFlowLoss) to ~24m by the end of warmup. When physics turned
        # on at epoch 2 as a step function, loss_static = static_w * mean(24^2)
        # ~= 700-920, a massive gradient shock into the shared encoder that
        # collapsed mIoU. Keeping loss_static/loss_smooth on from epoch 0 pins
        # the (>99%) static gaussians near zero offset throughout warmup, so it
        # never blows up and there is no shock when vel/rigid activate.
        #
        # The iter-0 crash the old warmup guarded against was caused by the
        # DRIVING terms (loss_vel with large GT-velocity targets, loss_rigid
        # variance) on randomly-placed gaussians -- exactly the terms still
        # zeroed here. loss_static/loss_smooth are bounded (offset^2, tiny at
        # iter 0: ~0.9 / ~0.1) and safe. Using a python float 0/1 scale keeps
        # the graph constant so static_graph=True + backbone with_cp=True stay
        # valid.
        in_warmup = (current_epoch is not None
                     and current_epoch < self.warmup_epoch)
        drive_scale = 0.0 if in_warmup else 1.0
        total = loss_static + loss_smooth + (
            loss_rigid + loss_vel + loss_traj) * drive_scale

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
                f'loss_traj={loss_traj.item():.4f} '
                f'total={total.item():.4f}'
            )
            if (not torch.distributed.is_available()
                    or not torch.distributed.is_initialized()
                    or torch.distributed.get_rank() == 0):
                print(_msg, flush=True)

        return total

    def _trajectory_target(self, offset, box_idx, dyn_mask, attr_labels_planner):
        """Smooth-L1(offset, GT instance future trajectory), masked to dynamic gaussians.

        Unlike ``_velocity_target`` (which extrapolates the current instantaneous
        box velocity as a constant-velocity straight line), this regresses each
        dynamic gaussian's offset onto the instance's REAL future trajectory,
        so acceleration / braking / turning / lane-change are all supervised.

        ``attr_labels_planner`` is (B, T, 34), aligned to gt_boxes rows:
            [0:12]  = fut_traj  (6 steps x 2, PER-STEP delta displacement)
            [12:18] = fut_mask  (6, 1 = step has a valid future annotation)
        The cumulative displacement d_t = cumsum(delta) is the offset target;
        it lives in the same LIDAR / world frame as offset (pure object motion,
        no ego term -- forward_flow removes ego separately, matching loss_vel).

        Per-step validity comes from the instance's fut_mask, so steps beyond
        the object's last annotation (occlusion / leaving the scene) are not
        supervised. Uses SMOOTH-L1 (bounded gradient) and a MULTIPLICATIVE mask
        over the FULL offset tensor -> graph constant every iter
        (static_graph=True compatible).

        Args:
            offset:   (B, G, 6, 2) predicted displacements (grad flows here)
            box_idx:  (B, G) long, containing-box index (-1 background)
            dyn_mask: (B, G) bool, True for gaussians inside a moving box
            attr_labels_planner: (B, T, 34) per-instance planner labels
        Returns:
            scalar tensor
        """
        with torch.no_grad():
            B, G = box_idx.shape
            attr = attr_labels_planner
            if not torch.is_tensor(attr):
                attr = torch.as_tensor(attr)
            attr = attr.to(offset.device).float()
            if attr.dim() == 2:
                attr = attr[None]                                 # (B, T, 34)
            T = attr.shape[1]
            # Empty attr (no annotations) -> no dynamic supervision.
            if T == 0:
                return offset.new_tensor(0.0)
            fut_delta = attr[..., 0:12].reshape(B, T, 6, 2)       # per-step delta
            fut_mask = attr[..., 12:18].reshape(B, T, 6)          # (B, T, 6)
            fut_delta = torch.nan_to_num(fut_delta, nan=0.0, posinf=0.0, neginf=0.0)
            fut_mask = torch.nan_to_num(fut_mask, nan=0.0, posinf=0.0, neginf=0.0)
            # cumulative displacement (offset target) in LIDAR frame
            cum_traj = torch.cumsum(fut_delta, dim=2)             # (B, T, 6, 2)
            cum_traj = cum_traj.clamp(-100.0, 100.0)
            # per-batch advanced indexing: cum_traj[b][bi[b]] -> (G, 6, 2)
            bi = box_idx.clamp(0, T - 1)                          # (B, G)
            target = torch.stack(
                [cum_traj[b][bi[b]] for b in range(B)], dim=0)    # (B, G, 6, 2)
            step_valid = torch.stack(
                [fut_mask[b][bi[b]] for b in range(B)], dim=0)    # (B, G, 6)
            # combine: gaussian is dynamic AND that future step is annotated
            w = dyn_mask.float()[:, :, None] * step_valid          # (B, G, 6)
            denom = w.sum() * offset.shape[-1] + 1e-6
        err = torch.nn.functional.smooth_l1_loss(
            offset, target, reduction='none', beta=1.0).sum(-1)   # (B, G, 6) grad→offset
        return (w * err).sum() / denom

    def _velocity_target(self, offset, box_idx, dyn_mask, gt_boxes,
                         ego_fut_trajs=None):
        """Smooth-L1(offset, ego-compensated GT-box trajectory), masked to dynamic gaussians.

        Corrected target per gaussian g at step t:
            target[b,g,t] = v_box*(t+1)*dt - ego_cumdisp[b,t]
        where ego_cumdisp = cumsum(ego_fut_trajs) gives the GT ego cumulative
        displacement in the LIDAR frame (same frame as offset and v_box).
        Subtracting ego displacement converts from world-relative object velocity
        to ego-relative: a static object (v_box=0) will have offset=-ego_disp.

        Uses SMOOTH-L1 (Huber, beta=1m) instead of MSE: the per-element gradient
        is bounded to +/-1, so a large target (a fast car 3 s out can be ~30 m)
        no longer explodes the gradient at iter 0 (MSE gave loss~24, grad~1.7e4).
        This removes the need for a warmup on loss_vel.

        Uses a MULTIPLICATIVE mask over the FULL offset tensor so the autograd
        graph is identical every iteration (static_graph=True compatible).
        """
        with torch.no_grad():
            B, G = box_idx.shape
            dt = 0.5
            gt_boxes = gt_boxes.to(offset.device).float()
            tmul = (torch.arange(6, device=offset.device).float() + 1.0) * dt  # (6,)
            # Empty gt_boxes (no annotations): v_box=0, dyn_mask is all-False
            # so w=0 and the loss is 0 regardless of target.
            if gt_boxes.shape[1] > 0:
                bi = box_idx.clamp(0, gt_boxes.shape[1] - 1)                  # (B, G)
                v_all = gt_boxes[..., 7:9]                                    # (B, T, 2)
                # Per-batch advanced indexing instead of torch.gather: v_all[b]
                # is (T, 2), bi[b] is (G,) long -> v_all[b][bi[b]] is (G, 2).
                # Avoids torch.gather with an expanded index, which triggered a
                # CUDA async error at iter 0 -> fake spconv empty-voxel at iter 1.
                v_box = torch.stack([v_all[b][bi[b]] for b in range(B)], dim=0)  # (B, G, 2)
            else:
                v_box = torch.zeros(B, G, 2, device=offset.device)
            # Sanitize GT velocity: nan/inf here -> nan target -> nan loss.
            # Even with warmup total*0, nan*0=nan poisons the gradient -> NaN
            # weights -> iter-1 gaussians become NaN -> warp_anchor valid_mask
            # all-False -> temporal_encoder gets 0 anchors -> spconv empty crash.
            v_box = torch.nan_to_num(v_box, nan=0.0, posinf=0.0, neginf=0.0)
            target = v_box[:, :, None, :] * tmul[None, None, :, None]          # (B,G,6,2)
            # offset is defined in the WORLD frame: forward_flow (gaussian_head)
            # subtracts the GT ego displacement separately to build occ_flow in
            # the future-ego frame. So the vel target is the PURE object world
            # displacement v_box*t -- NO ego term here. Adding ego would
            # double-count (forward_flow already removes it) and make offset
            # regress to an ego-relative quantity instead of pure motion.
            target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
            target = target.clamp(-100.0, 100.0)
            w = dyn_mask.float()[:, :, None]                                   # (B,G,1) const
            denom = w.sum() * offset.shape[2] + 1e-6
        # Smooth-L1 per element (beta=1.0), summed over the xy dim -> (B,G,6).
        # Bounded gradient (|grad|<=1) prevents the MSE explosion on large targets.
        err = torch.nn.functional.smooth_l1_loss(
            offset, target, reduction='none', beta=1.0).sum(-1)              # (B,G,6) grad→offset
        return (w * err).sum() / denom

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
