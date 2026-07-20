from mmengine.registry import MODELS
from mmengine.model import BaseModule
from mmcv.cnn import Scale
import torch.nn as nn, torch
import torch.nn.functional as F
from .utils import linear_relu_ln, GaussianPrediction
from model.utils.safe_ops import safe_sigmoid


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
        # number of future steps = offset_dim // 2; kinematic head param count = 3
        self.kin_steps = offset_dim // 2
        self.kin_param_dim = 3

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
            self.offset_layers = nn.Sequential(
                *linear_relu_ln(embed_dims, 1, 1),
                nn.Linear(self.embed_dims, head_out))

    def _kinematic_rollout(self, kin):
        """CTRV (constant-turn-rate, constant-speed) rollout.

        Args:
            kin: (..., 3) raw params per gaussian = [vx, vy, omega], where
                 (vx, vy) is the initial velocity vector (m/s, world/lidar xy)
                 and omega is the yaw rate (rad/s).
        Returns:
            (..., kin_steps*2) cumulative xy displacement for steps 1..kin_steps,
            matching the free-mode offset layout (reshape(...,6,2) -> [step, xy]).

        Displacement of step j is v0 rotated by (omega*dt*j) times dt; the
        cumulative sum over j gives an arc. omega=0 degenerates to a straight
        line, so the head can express both, but turning is now a single scalar
        (omega) instead of 12 free numbers -> far easier under sparse GT.
        """
        vx = kin[..., 0]
        vy = kin[..., 1]
        omega = kin[..., 2]
        dt = self.kin_dt
        steps = self.kin_steps
        j = torch.arange(1, steps + 1, device=kin.device, dtype=kin.dtype)  # (S,)
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
        mask=None
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
        if self.decouple_offset:
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
            offset = offset_decoupled
            if mask is not None:
                offset = offset[mask].unsqueeze(0)
        else:
            offset = output[..., -self.offset_dim:]
        return output, gaussian, offset

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
