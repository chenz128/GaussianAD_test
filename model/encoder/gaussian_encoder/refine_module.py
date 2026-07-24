from mmengine.registry import MODELS
from mmengine.model import BaseModule
from mmcv.cnn import Scale
import torch.nn as nn, torch
import torch.nn.functional as F
from .utils import linear_relu_ln, GaussianPrediction, cartesian
from model.utils.safe_ops import safe_sigmoid

try:
    from model.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu
except Exception:  # pragma: no cover - only needed when motion_cond=True
    points_in_boxes_gpu = None


class MotionCrossAttention(nn.Module):
    """v11c: current-frame gaussians attend to their historical-frame gaussians
    to gather object-motion context (position-sequence curvature -> turn rate)
    for the offset head. Association is GEOMETRIC: position is embedded into both
    Q and K, so a current gaussian focuses on the spatially-consistent history of
    the same object (the lifter tiles a SPATIAL anchor set across frames, so a
    moving object occupies different anchor indices per frame -> per-index
    temporal attention would not track it; position-keyed attention does).

    Memory-efficient via scaled_dot_product_attention (no N_q x N_k matrix
    materialized). The output projection is zero-initialized so at init the
    attention contributes 0 -> motion_feat == feat and the offset head starts
    exactly as the v10 constant-velocity CTRA rollout (no shock).
    """

    def __init__(self, embed_dims, num_heads=4):
        super().__init__()
        assert embed_dims % num_heads == 0, \
            'embed_dims must be divisible by num_heads'
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_dim = embed_dims // num_heads
        self.q_proj = nn.Linear(embed_dims, embed_dims)
        self.k_proj = nn.Linear(embed_dims, embed_dims)
        self.v_proj = nn.Linear(embed_dims, embed_dims)
        self.o_proj = nn.Linear(embed_dims, embed_dims)
        # position (xyz, normalized to ~[0,1]) and time-gap (dt in frames)
        # embeddings, added into Q/K so attention is geometric + time-aware.
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, embed_dims), nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims))
        self.dt_mlp = nn.Sequential(
            nn.Linear(1, embed_dims), nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims))
        # zero-init output: attention contributes 0 at init -> == v10 start.
        nn.init.zeros_(self.o_proj.weight)
        nn.init.zeros_(self.o_proj.bias)

    def _attn(self, q, k, v):
        # q: (Nc, C); k, v: (Nk, C) -> (Nc, C)
        Nc, Nk = q.shape[0], k.shape[0]
        H, D = self.num_heads, self.head_dim
        q = q.reshape(Nc, H, D).transpose(0, 1).unsqueeze(0)  # (1, H, Nc, D)
        k = k.reshape(Nk, H, D).transpose(0, 1).unsqueeze(0)  # (1, H, Nk, D)
        v = v.reshape(Nk, H, D).transpose(0, 1).unsqueeze(0)
        o = F.scaled_dot_product_attention(q, k, v)           # (1, H, Nc, D)
        return o.squeeze(0).transpose(0, 1).reshape(Nc, self.embed_dims)

    def forward(self, q_feat, q_pos, kv_feat, kv_pos, kv_dt,
                q_bidx=None, kv_bidx=None):
        """q_feat/kv_feat: (Nq/Nk, C); q_pos/kv_pos: (.,3) normalized; kv_dt: (Nk,)
        q_bidx/kv_bidx: (.,) real batch id so a current gaussian only attends to
        history of the SAME sample. Returns (Nq, C) additive residual."""
        q = self.q_proj(q_feat) + self.pos_mlp(q_pos)
        k = (self.k_proj(kv_feat) + self.pos_mlp(kv_pos)
             + self.dt_mlp(kv_dt[:, None]))
        v = self.v_proj(kv_feat)
        if q_bidx is None or kv_bidx is None:
            out = self._attn(q, k, v)
        else:
            out = torch.zeros_like(q)
            for b in torch.unique(q_bidx):
                qm = q_bidx == b
                km = kv_bidx == b
                if km.any():
                    out[qm] = self._attn(q[qm], k[km], v[km])
        return self.o_proj(out)


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
        decouple_dynamic=False,
        decouple_offset=False,
        offset_grad_scale=0.0,
        offset_mode='free',
        kin_dt=0.5,
        motion_cond=False,
        kin_omega_max=0.5,
        kin_accel_max=3.0,
        motion_v_thresh=0.5,
        use_motion_attn=False,
        motion_attn_heads=4,
    ):
        super(SparseGaussian3DRefinementModule, self).__init__()
        self.embed_dims = embed_dims
        self.xyz_coordinate = xyz_coordinate
        self.use_dynamic = use_dynamic
        # v10: when True, the dynamic head reads a DETACHED feature so
        # DynamicLoss trains only dynamic_layers and never leaks gradient into
        # the shared encoder -> current-frame occ stays == base (the offset head
        # is already decoupled; this closes the last encoder-touching path that
        # base did not have).
        self.decouple_dynamic = decouple_dynamic
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
        # v11c: current<-historical gaussian cross-attention feeds the offset
        # head real object-motion context (position-sequence curvature -> omega)
        # that the v10 raw motion_state concat could not provide.
        self.use_motion_attn = use_motion_attn
        self.motion_attn_heads = motion_attn_heads
        # number of future steps = offset_dim // 2
        self.kin_steps = offset_dim // 2
        # kinematic head param count: motion-conditioned CTRA=2 [omega, accel]
        # ONLY. The initial velocity v0 is the OBSERVED GT-box velocity (passed
        # into the rollout), NOT predicted. A zero-init head => omega=accel=0 =>
        # offset = v0*t (real constant-velocity extrapolation, non-zero for
        # movers, no OccFlowLoss shock). Predicting v0 instead put the head at a
        # zero-gradient fixed point (at v=0, d|v|/dv=0 AND atan2 routes to a
        # constant heading, so vx,vy never receive gradient) -> offset stuck at
        # 0 forever. legacy CTRV=3 [vx, vy, omega].
        self.kin_param_dim = 2 if motion_cond else 3

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
            # v11c: when use_motion_attn, motion context is injected as an
            # additive residual by MotionCrossAttention (not the raw concat), so
            # the head consumes just embed_dims.
            if self.use_motion_attn:
                in_dim = self.embed_dims
            else:
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
            if self.use_motion_attn:
                self.motion_attn = MotionCrossAttention(
                    self.embed_dims, num_heads=self.motion_attn_heads)

    def _kinematic_rollout(self, kin, v0=None):
        """Differentiable kinematic rollout -> (..., kin_steps*2) cumulative xy.

        Modes (selected by args):
          * v10 motion-conditioned CTRA (v0 given, kin=[omega_raw, accel_raw],
            2 params): the initial velocity v0 is the OBSERVED GT-box velocity
            (constant, no grad); the head predicts only a BOUNDED yaw rate
            (omega_max*tanh) and longitudinal accel (accel_max*tanh). Zero-init
            head => omega=accel=0 => offset=v0*t (real constant-velocity
            extrapolation, non-zero for movers). Cannot spin into circles
            (bounded omega), can brake (accel<0), and has NO zero-gradient dead
            point (v0 is provided, not predicted).
          * legacy predicted-v0 CTRA (kin=[vx, vy, omega_raw, accel_raw], 4
            params): head predicts v0 too. UNUSED -- kept for back-compat; it
            has a zero-gradient fixed point at v=0 (offset stays 0 forever).
          * legacy CTRV (kin=[vx, vy, omega], 3 params): constant-speed arc (v9).
        """
        dt = self.kin_dt
        steps = self.kin_steps
        j = torch.arange(1, steps + 1, device=kin.device, dtype=kin.dtype)  # (S,)
        if v0 is not None and kin.shape[-1] == 2:
            # v10 motion-conditioned CTRA: the initial velocity v0=[vx, vy] is
            # the OBSERVED GT-box velocity (a constant, no grad); the head
            # predicts ONLY [omega, accel]. With omega=accel=0 (zero-init) the
            # offset is exactly v0*t. Because s0=|v0| is the REAL speed (not ~0),
            # the gradient to omega/accel is well-scaled from the first iter, so
            # the head can immediately learn turning / braking. There is no dead
            # fixed point because v0 is provided, not predicted.
            vx0 = v0[..., 0]
            vy0 = v0[..., 1]
            omega = self.kin_omega_max * torch.tanh(kin[..., 0])   # bounded
            accel = self.kin_accel_max * torch.tanh(kin[..., 1])   # bounded
            s0 = torch.sqrt(vx0 * vx0 + vy0 * vy0 + 1e-8)          # observed speed
            # atan2(0,0) has NaN grad; static gaussians (v0=0) are zero-gated
            # downstream anyway, so route them to a constant heading here.
            near0 = (vx0 * vx0 + vy0 * vy0) < 1e-8
            vx0g = torch.where(near0, torch.ones_like(vx0), vx0)
            vy0g = torch.where(near0, torch.zeros_like(vy0), vy0)
            head0 = torch.atan2(vy0g, vx0g)                        # observed heading
            t = j * dt                                             # (S,)
            hj = head0[..., None] + omega[..., None] * t           # (..., S)
            sj = torch.relu(s0[..., None] + accel[..., None] * t)  # (..., S) >=0
            vjx = sj * torch.cos(hj)
            vjy = sj * torch.sin(hj)
            dispx = torch.cumsum(vjx * dt, dim=-1)
            dispy = torch.cumsum(vjy * dt, dim=-1)
            off = torch.stack([dispx, dispy], dim=-1)              # (..., S, 2)
            return off.reshape(*off.shape[:-2], steps * 2)
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
        batch_indices=None,
    ):
        feat = instance_feature + anchor_embed
        output = self.layers(feat)
        # v10: optionally decouple the dynamic head from the encoder (read a
        # DETACHED feature) so DynamicLoss trains only dynamic_layers and never
        # perturbs the shared encoder -> current-frame occ stays == base.
        dyn_in = feat.detach() if self.decouple_dynamic else feat
        dynamic_logits = self.dynamic_layers(dyn_in) if self.use_dynamic else None
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
                offset = self._motion_offset(
                    feat, mask, gaussian.means, gt_boxes,
                    anchor=anchor, batch_indices=batch_indices)
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
        GT box, computed ON CPU. Doing per-gaussian box tests on the GPU INSIDE
        the DDP-wrapped forward corrupts the CUDA stream and crashes the next
        spconv (empty tensor) on 4-GPU (confirmed by bisection); CPU keeps it off
        the CUDA stream. Detached + no_grad -> pure data. Background -> zeros.
        means: (B, G, 3); gt_boxes: (B, T, >=9) [x,y,z,dx,dy,dz,heading,vx,vy,...].
        """
        if gt_boxes is None:
            return None
        m = torch.nan_to_num(means.detach().float(), nan=0.0,
                             posinf=0.0, neginf=0.0).cpu()
        gt = gt_boxes
        if not torch.is_tensor(gt):
            gt = torch.as_tensor(gt)
        # The dataset normally sanitizes unavailable nuScenes velocities, but
        # keep this model boundary defensive for old PKLs and external callers.
        # A single NaN on one DDP rank otherwise poisons the all-reduced gradient
        # and makes all anchors non-finite on the next iteration.
        gt = torch.nan_to_num(gt.float().cpu(), nan=0.0,
                      posinf=0.0, neginf=0.0)
        if gt.dim() == 2:
            gt = gt[None]
        B, G = m.shape[0], m.shape[1]
        T = gt.shape[1]
        ms = torch.zeros((B, G, 3), dtype=torch.float32)
        if T == 0:
            return ms.to(means.device)
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
        return ms.to(means.device)

    def _motion_offset(self, feat, mask, means, gt_boxes, anchor=None,
                       batch_indices=None):
        """v10/v11c: motion-conditioned bounded-CTRA offset for current-frame
        gaussians.

        v10: the offset head reads feat.detach() concatenated with the observed
        motion state [vx, vy, heading] from the containing GT box.
        v11c (use_motion_attn): the concat is REPLACED by a MotionCrossAttention
        residual -- current-frame gaussians attend to their historical-frame
        gaussians (geometric, position-keyed) to gather real object-motion
        context (curvature -> turn rate). v0 + the static gate are still taken
        from the observed motion state. Both Q and the historical K/V feature
        paths carry the same offset_grad_scale blend; positions are detached.
        """
        feat_c = feat[mask] if mask is not None else feat
        motion_state = None
        if gt_boxes is not None and mask is not None:
            ms = self._motion_state_from_boxes(means, gt_boxes)   # (B, Gc, 3)
            if ms is not None:
                motion_state = ms.reshape(-1, 3)
        if motion_state is None:
            motion_state = feat_c.new_zeros((feat_c.shape[0], 3))
        motion_state = torch.nan_to_num(
            motion_state.to(device=feat_c.device, dtype=feat_c.dtype),
            nan=0.0, posinf=0.0, neginf=0.0)

        s = self.offset_grad_scale

        def _blend(x):
            # straight-through: forward == x, gradient to encoder scaled by s.
            if s and s > 0:
                return s * x + (1.0 - s) * x.detach()
            return x.detach()

        feat_off = _blend(feat_c)

        if self.use_motion_attn:
            # current-frame gaussians attend to their historical-frame
            # counterparts to gather object-motion context for omega/accel.
            attn_out = torch.zeros_like(feat_off)
            if (anchor is not None and batch_indices is not None
                    and mask is not None):
                hist = ~mask
                if hist.any():
                    feat_h = _blend(feat[hist])
                    lo = feat_off.new_tensor(self.pc_range[:3])
                    span = feat_off.new_tensor(
                        [self.pc_range[i + 3] - self.pc_range[i]
                         for i in range(3)])
                    cur_pos = (means.reshape(-1, 3).detach() - lo) / span
                    hist_pos = (cartesian(anchor[hist], self.pc_range).detach()
                                - lo) / span
                    fmax = batch_indices[:, 1].max()
                    dt = (fmax - batch_indices[hist, 1]).to(feat_off.dtype)
                    attn_out = self.motion_attn(
                        feat_off, cur_pos, feat_h, hist_pos, dt,
                        q_bidx=batch_indices[mask, 0],
                        kv_bidx=batch_indices[hist, 0])
            head_in = feat_off + attn_out
        else:
            head_in = torch.cat([feat_off, motion_state], dim=-1)

        raw = self.offset_layers(head_in)
        # observed GT-box velocity is the CTRA initial velocity (constant, no
        # grad); the head's raw output is only [omega, accel]. This fixes the
        # zero-gradient dead point of the previous predicted-v0 head.
        v0 = motion_state[..., :2]
        off = self._kinematic_rollout(raw, v0=v0)
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
