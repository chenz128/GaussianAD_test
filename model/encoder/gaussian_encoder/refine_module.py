from mmengine.registry import MODELS
from mmengine.model import BaseModule
from mmcv.cnn import Scale
import torch.nn as nn, torch
import torch.nn.functional as F
from .utils import linear_relu_ln, GaussianPrediction
from model.utils.safe_ops import safe_sigmoid

try:
    from model.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu
except Exception:  # pragma: no cover - only needed when motion_cond=True
    points_in_boxes_gpu = None


@MODELS.register_module()
class SparseGaussian3DRefinementModule(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        pc_range=None,
        scale_range=None,
        restrict_xyz=False,
        unit_xyz=None,
        refine_manual=None,
        phi_activation='sigmoid',
        semantics=False,
        semantic_dim=None,
        include_opa=True,
        xyz_coordinate='polar',
        semantics_activation='softmax',
        offset_dim=2*6,
        use_dynamic=False,
        decouple_offset=False,
        offset_grad_scale=0.0,
        offset_mode='free',
        kin_dt=0.5,
        motion_cond=False,
        kin_omega_max=0.5,
        kin_accel_max=3.0,
        motion_v_thresh=0.5,
    ):
        super(SparseGaussian3DRefinementModule, self).__init__()
        self.embed_dims = embed_dims
        self.xyz_coordinate = xyz_coordinate
        self.use_dynamic = use_dynamic
        # When True, the returned offset is produced by a dedicated head that
        # reads a DETACHED copy of the shared feature. This isolates the offset
        # (and therefore all PhysicsLoss / OccFlowLoss-via-offset gradients)
        # from the encoder: those gradients train only self.offset_layers and
        # never flow back into instance_feature. The occupancy semantics stay
        # clean (no kinematic pollution). Offset is still fully supervised via
        # its own head. See docs: gtbox_oracle v6 decoupling.
        self.decouple_offset = decouple_offset
        # v8: fraction of the offset-head gradient allowed to leak back into the
        # shared encoder feature. 0.0 == full detach (v7 behaviour).
        self.offset_grad_scale = offset_grad_scale
        # v9: offset parametrization.
        #   'free'      -> the head outputs offset_dim (=6*2) FREE numbers; the
        #                  6 future xy displacements are unconstrained (v7/v8).
        #   'kinematic' -> the head outputs 3 params [vx, vy, omega] per
        #                  gaussian and a differentiable constant-turn-rate
        #                  constant-speed (CTRV) rollout integrates them into
        #                  the same (6, 2) cumulative displacement. The output
        #                  is a smooth arc (omega!=0 => turning), which cannot
        #                  collapse to a degenerate straight line under sparse
        #                  supervision and needs only 3 low-variance scalars.
        #                  Requires decouple_offset=True. Downstream (forward_flow
        #                  / PhysicsLoss) is unchanged: it still receives (6, 2).
        self.offset_mode = offset_mode
        self.kin_dt = kin_dt
        # v10: condition the offset head on the object's OBSERVED motion state
        # ([vx, vy, heading], broadcast from the containing GT box). When on, the
        # kinematic head predicts only the EVOLUTION [omega, accel] and a bounded
        # CTRA rollout integrates it from the observed initial velocity v0, so
        # magnitude + initial direction are given and turning cannot blow up.
        self.motion_cond = motion_cond
        self.kin_omega_max = kin_omega_max
        self.kin_accel_max = kin_accel_max
        self.motion_v_thresh = motion_v_thresh
        # number of future steps = offset_dim // 2
        self.kin_steps = offset_dim // 2
        # kinematic head param count: motion-conditioned CTRA=4 [vx, vy, omega,
        # accel] (velocity is PREDICTED, guided by the observed conditioning, so
        # a zero-init head gives offset=0 at iter 0 -> no OccFlowLoss shock on the
        # untrained encoder); legacy CTRV=3 [vx, vy, omega].
        self.kin_param_dim = 4 if motion_cond else 3

        if semantics:
            assert semantic_dim is not None
        else:
            semantic_dim = 0

        self.offset_dim = offset_dim

        self.output_dim = 10 + int(include_opa) + semantic_dim + offset_dim
        self.semantic_start = 10 + int(include_opa)
        self.semantic_dim = semantic_dim
        self.include_opa = include_opa
        self.semantics_activation = semantics_activation

        self.pc_range = pc_range
        self.scale_range = scale_range
        self.restrict_xyz = restrict_xyz
        self.unit_xyz = unit_xyz
        self.phi_activation = phi_activation
        if restrict_xyz:
            assert unit_xyz is not None
            unit_prob = [unit_xyz[i] / (pc_range[i + 3] - pc_range[i]) for i in range(3)]
            unit_sigmoid = [4 * unit_prob[i] for i in range(3)]
            if phi_activation == 'loop':
                unit_sigmoid[2] = unit_prob[2]
            self.unit_sigmoid = unit_sigmoid

        assert isinstance(refine_manual, list)
        self.refine_state = refine_manual
        assert all([self.refine_state[i] == i for i in range(len(self.refine_state))])

        self.layers = nn.Sequential(
            *linear_relu_ln(embed_dims, 2, 2),
            nn.Linear(self.embed_dims, self.output_dim),
            Scale([1.0] * self.output_dim))

        if self.use_dynamic:
            self.dynamic_layers = nn.Sequential(
                *linear_relu_ln(embed_dims, 1, 1),
                nn.Linear(self.embed_dims, 1))

        # Dedicated offset head (decoupled). Same small-branch pattern as
        # dynamic_layers, but its input feature is detached in forward so the
        # offset gradient never reaches the encoder.
        if self.decouple_offset and self.offset_dim > 0:
            head_out = (self.kin_param_dim if self.offset_mode == 'kinematic'
                        else self.offset_dim)
            # v10: concat the 3-dim motion state ([vx, vy, heading]) onto the
            # detached feature -> the first Linear consumes embed_dims + 3.
            in_dim = self.embed_dims + (3 if self.motion_cond else 0)
            self.offset_layers = nn.Sequential(
                *linear_relu_ln(embed_dims, 1, 1, input_dims=in_dim),
                nn.Linear(self.embed_dims, head_out))
            if self.motion_cond:
                # zero-init the last Linear so initial omega=accel=0 -> the
                # offset starts as pure constant-velocity extrapolation v0*t
                # (GT-scale, no random overshoot that would shock OccFlowLoss).
                nn.init.zeros_(self.offset_layers[-1].weight)
                nn.init.zeros_(self.offset_layers[-1].bias)

    def _kinematic_rollout(self, kin, v0=None):
        """Differentiable kinematic rollout -> (..., kin_steps*2) cumulative xy.

        Two modes:
          * v10 bounded CTRA (kin=[vx, vy, omega_raw, accel_raw], 4 params): the
            head PREDICTS the initial velocity (vx, vy) (guided by the observed
            motion conditioning), plus a BOUNDED yaw rate (omega_max*tanh) and
            longitudinal accel (accel_max*tanh). A zero-init head => velocity=0,
            omega=accel=0 => offset=0 at iter 0 (no OccFlowLoss shock). Cannot
            spin into circles (bounded omega) and can brake (accel<0).
          * legacy CTRV (kin=[vx, vy, omega], 3 params): constant-speed arc (v9).
        """
        dt = self.kin_dt
        steps = self.kin_steps
        j = torch.arange(1, steps + 1, device=kin.device, dtype=kin.dtype)  # (S,)
        if kin.shape[-1] == 4:
            vx0 = kin[..., 0]
            vy0 = kin[..., 1]
            omega = self.kin_omega_max * torch.tanh(kin[..., 2])   # bounded
            accel = self.kin_accel_max * torch.tanh(kin[..., 3])   # bounded
            s0 = torch.sqrt(vx0 * vx0 + vy0 * vy0 + 1e-8)          # speed
            # guard atan2(0, 0): its gradient is NaN, which would propagate even
            # through a zero static gate (0 * NaN = NaN). Route near-zero-velocity
            # gaussians to a constant heading (no grad to vx/vy there).
            near0 = (vx0 * vx0 + vy0 * vy0) < 1e-8
            vx0g = torch.where(near0, torch.ones_like(vx0), vx0)
            vy0g = torch.where(near0, torch.zeros_like(vy0), vy0)
            head0 = torch.atan2(vy0g, vx0g)                        # heading
            t = j * dt                                             # (S,)
            hj = head0[..., None] + omega[..., None] * t           # (..., S)
            sj = torch.relu(s0[..., None] + accel[..., None] * t)  # (..., S) >=0
            vjx = sj * torch.cos(hj)
            vjy = sj * torch.sin(hj)
            dispx = torch.cumsum(vjx * dt, dim=-1)
            dispy = torch.cumsum(vjy * dt, dim=-1)
            off = torch.stack([dispx, dispy], dim=-1)              # (..., S, 2)
            return off.reshape(*off.shape[:-2], steps * 2)
        vx = kin[..., 0]
        vy = kin[..., 1]
        omega = kin[..., 2]
        ang = omega[..., None] * dt * j          # (..., S) heading after j steps
        cos_a = torch.cos(ang)
        sin_a = torch.sin(ang)
        # rotate the initial velocity vector by ang (per step)
        vjx = cos_a * vx[..., None] - sin_a * vy[..., None]   # (..., S)
        vjy = sin_a * vx[..., None] + cos_a * vy[..., None]   # (..., S)
        dispx = torch.cumsum(vjx * dt, dim=-1)               # (..., S) cumulative
        dispy = torch.cumsum(vjy * dt, dim=-1)               # (..., S)
        off = torch.stack([dispx, dispy], dim=-1)            # (..., S, 2)
        return off.reshape(*off.shape[:-2], steps * 2)       # (..., 2S)

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        mask=None,
        gt_boxes=None,
    ):
        feat = instance_feature + anchor_embed
        output = self.layers(feat)
        dynamic_logits = self.dynamic_layers(feat) if self.use_dynamic else None
        # Decoupled offset: read a DETACHED feature so its gradient stops at the
        # offset head and never pollutes the encoder. Masked at the end to match
        # the (masked) output layout, mirroring dynamic_logits. When
        # offset_grad_scale>0, leak that fraction of gradient into the encoder
        # via a straight-through blend: s*feat + (1-s)*feat.detach() has value
        # == feat but gradient scaled by s.
        if self.decouple_offset and not self.motion_cond:
            s = self.offset_grad_scale
            if s and s > 0:
                feat_off = s * feat + (1.0 - s) * feat.detach()
            else:
                feat_off = feat.detach()
            raw_off = self.offset_layers(feat_off)
            if self.offset_mode == 'kinematic':
                # CTRV rollout: 3 params -> (..., offset_dim) cumulative xy.
                offset_decoupled = self._kinematic_rollout(raw_off)
            else:
                offset_decoupled = raw_off
        else:
            # v10 motion_cond computes the offset later (needs the masked,
            # current-frame layout aligned with means); non-decoupled uses the
            # output tail slice.
            offset_decoupled = None

        if self.restrict_xyz:
            delta_xyz_sigmoid = output[..., :3]
            delta_xyz_prob = 2 * safe_sigmoid(delta_xyz_sigmoid) - 1
            delta_xyz = torch.stack([
                delta_xyz_prob[..., 0] * self.unit_sigmoid[0],
                delta_xyz_prob[..., 1] * self.unit_sigmoid[1],
                delta_xyz_prob[..., 2] * self.unit_sigmoid[2]
            ], dim=-1)
            output = torch.cat([delta_xyz, output[..., 3:]], dim=-1)

        if len(self.refine_state) > 0:
            refined_part_output = output[..., self.refine_state] + anchor[..., self.refine_state]
            output = torch.cat([refined_part_output, output[..., len(self.refine_state):]], dim=-1)
        rot = torch.nn.functional.normalize(output[..., 6:10], dim=-1)

        output = torch.cat([output[..., :6], rot, output[..., 10:]], dim=-1)

        if mask is not None:
            output = output[mask].unsqueeze(0)

        if self.phi_activation == 'sigmoid':
            xyz = safe_sigmoid(output[..., :3])
        elif self.phi_activation == 'loop':
            xy = safe_sigmoid(output[..., :2])
            z = torch.remainder(output[..., 2:3], 1.0)
            xyz = torch.cat([xy, z], dim=-1)
        else:
            raise NotImplementedError

        if self.xyz_coordinate == 'polar':
            rrr = xyz[..., 0] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
            theta = xyz[..., 1] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            phi = xyz[..., 2] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
            xxx = rrr * torch.sin(theta) * torch.cos(phi)
            yyy = rrr * torch.sin(theta) * torch.sin(phi)
            zzz = rrr * torch.cos(theta)
        else:
            xxx = xyz[..., 0] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
            yyy = xyz[..., 1] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            zzz = xyz[..., 2] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
        xyz = torch.stack([xxx, yyy, zzz], dim=-1)

        gs_scales = safe_sigmoid(output[..., 3:6])
        gs_scales = self.scale_range[0] + (self.scale_range[1] - self.scale_range[0]) * gs_scales

        semantics_logits = output[..., self.semantic_start: (self.semantic_start + self.semantic_dim)]
        if self.semantics_activation == 'softmax':
            semantics = semantics_logits.softmax(dim=-1)
        elif self.semantics_activation == 'softplus':
            semantics = F.softplus(semantics_logits)
        else:
            semantics = semantics_logits

        if dynamic_logits is not None and mask is not None:
            dynamic_logits = dynamic_logits[mask].unsqueeze(0)
        gaussian = GaussianPrediction(
            means=xyz,
            scales=gs_scales,
            rotations=output[..., 6:10],
            opacities=safe_sigmoid(output[..., 10: (10 + int(self.include_opa))]),
            semantics=semantics,
            semantics_logits=semantics_logits,
            dynamic_logits=dynamic_logits,
        )
        if self.decouple_offset:
            if self.motion_cond:
                offset = self._motion_offset(feat, mask, gaussian.means, gt_boxes)
            else:
                offset = offset_decoupled
                if mask is not None:
                    offset = offset[mask].unsqueeze(0)
        else:
            offset = output[..., -self.offset_dim:]
        return output, gaussian, offset

    @torch.no_grad()
    def _motion_state_from_boxes(self, means, gt_boxes):
        """Per-gaussian observed motion state [vx, vy, heading] from the containing
        GT box. Uses a pure-PyTorch rotated-box test (NOT the custom CUDA op
        points_in_boxes_gpu, which is unsafe to call inside the DDP forward and
        caused a 4-GPU iter-1 empty-spconv crash). Background -> zeros.
        means: (B, G, 3); gt_boxes: (B, T, >=9) [x,y,z,dx,dy,dz,heading,vx,vy,...].
        """
        if gt_boxes is None:
            return None
        m = torch.nan_to_num(means.detach().float(), nan=0.0,
                             posinf=0.0, neginf=0.0)
        gt = gt_boxes
        if not torch.is_tensor(gt):
            gt = torch.as_tensor(gt)
        gt = gt.to(m.device).float()
        if gt.dim() == 2:
            gt = gt[None]
        B, G = m.shape[0], m.shape[1]
        T = gt.shape[1]
        ms = m.new_zeros((B, G, 3))
        if T == 0:
            return ms
        cx, cy, cz = gt[..., 0], gt[..., 1], gt[..., 2]        # (B, T)
        dx, dy, dz = gt[..., 3], gt[..., 4], gt[..., 5]        # (B, T)
        yaw = gt[..., 6]                                       # (B, T)
        cos = torch.cos(-yaw)[:, None, :]                      # (B, 1, T)
        sin = torch.sin(-yaw)[:, None, :]
        px = m[..., 0:1]                                       # (B, G, 1)
        py = m[..., 1:2]
        pz = m[..., 2:3]
        ddx = px - cx[:, None, :]                             # (B, G, T)
        ddy = py - cy[:, None, :]
        ddz = pz - cz[:, None, :]
        lx = ddx * cos - ddy * sin                            # box-frame x
        ly = ddx * sin + ddy * cos                            # box-frame y
        eps = 1e-4
        inside = ((lx.abs() <= dx[:, None, :] * 0.5)
                  & (ly.abs() <= dy[:, None, :] * 0.5)
                  & (ddz.abs() <= dz[:, None, :] * 0.5)
                  & (dx[:, None, :] > eps))                    # (B, G, T)
        any_in = inside.any(dim=-1)                           # (B, G)
        box_idx = inside.float().argmax(dim=-1)               # (B, G) first match
        for b in range(B):
            sel = any_in[b]
            if sel.any():
                bi = box_idx[b, sel]
                ms[b, sel, 0] = gt[b, bi, 7]                  # vx
                ms[b, sel, 1] = gt[b, bi, 8]                  # vy
                ms[b, sel, 2] = gt[b, bi, 6]                  # heading
        return ms

    def _motion_offset(self, feat, mask, means, gt_boxes):
        """v10: motion-conditioned bounded-CTRA offset for current-frame gaussians.

        ``feat`` is the (num_valid, C) pre-mask feature; we select the current
        frame with ``mask`` so the offset aligns with the masked means/gaussian.
        The observed motion state [vx, vy, heading] is derived here from the
        (masked) ``means`` + ``gt_boxes`` (same membership as physics_loss).
        Detached so the offset gradient never reaches the encoder.
        """
        feat_c = feat[mask] if mask is not None else feat
        # means is (B, Gc, 3) after mask (unsqueezed) or (B, G, 3); flatten batch
        # to align row-for-row with feat_c (num_current, C).
        motion_state = None
        if gt_boxes is not None and mask is not None:
            ms = self._motion_state_from_boxes(means, gt_boxes)   # (B, Gc, 3)
            if ms is not None:
                motion_state = ms.reshape(-1, 3)
                motion_state = motion_state * 0.0  # DIAG: compute but zero values
        if motion_state is None:
            motion_state = feat_c.new_zeros((feat_c.shape[0], 3))
        motion_state = motion_state.to(feat_c.dtype)
        s = self.offset_grad_scale
        if s and s > 0:
            feat_off = s * feat_c + (1.0 - s) * feat_c.detach()
        else:
            feat_off = feat_c.detach()
        raw = self.offset_layers(torch.cat([feat_off, motion_state], dim=-1))
        off = self._kinematic_rollout(raw)
        # static gate: gaussians whose OBSERVED speed ~0 get exactly zero offset,
        # so the background never drifts (makes loss_static redundant).
        s0 = motion_state[..., :2].norm(dim=-1)
        off = off * (s0 > self.motion_v_thresh).to(off.dtype).unsqueeze(-1)
        return off.unsqueeze(0) if mask is not None else off

    def get_gaussian(self, output):
        if self.phi_activation == 'sigmoid':
            xyz = safe_sigmoid(output[..., :3])
        elif self.phi_activation == 'loop':
            xy = safe_sigmoid(output[..., :2])
            z = torch.remainder(output[..., 2:3], 1.0)
            xyz = torch.cat([xy, z], dim=-1)
        else:
            raise NotImplementedError

        if self.xyz_coordinate == 'polar':
            rrr = xyz[..., 0] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
            theta = xyz[..., 1] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            phi = xyz[..., 2] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
            xxx = rrr * torch.sin(theta) * torch.cos(phi)
            yyy = rrr * torch.sin(theta) * torch.sin(phi)
            zzz = rrr * torch.cos(theta)
        else:
            xxx = xyz[..., 0] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
            yyy = xyz[..., 1] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            zzz = xyz[..., 2] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]
        xyz = torch.stack([xxx, yyy, zzz], dim=-1)

        gs_scales = safe_sigmoid(output[..., 3:6])
        gs_scales = self.scale_range[0] + (self.scale_range[1] - self.scale_range[0]) * gs_scales

        semantics_logits = output[..., self.semantic_start: (self.semantic_start + self.semantic_dim)]
        if self.semantics_activation == 'softmax':
            semantics = semantics_logits.softmax(dim=-1)
        elif self.semantics_activation == 'softplus':
            semantics = F.softplus(semantics_logits)
        else:
            semantics = semantics_logits

        gaussian = GaussianPrediction(
            means=xyz,
            scales=gs_scales,
            rotations=output[..., 6:10],
            opacities=safe_sigmoid(output[..., 10: (10 + int(self.include_opa))]),
            semantics=semantics,
            semantics_logits=semantics_logits,
            dynamic_logits=None,
        )
        return gaussian
