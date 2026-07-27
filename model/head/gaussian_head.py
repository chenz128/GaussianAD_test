import numpy as np
import torch, torch.nn as nn

from mmengine.registry import MODELS
from .base_head import BaseTaskHead
from .localagg.local_aggregate import LocalAggregator
from ..utils.utils import list_2_tensor, get_rotation_matrix


@MODELS.register_module()
class GaussianHead(BaseTaskHead):
    def __init__(
        self,
        init_cfg=None,
        apply_loss_type=None,
        num_classes=18,
        empty_args=None,
        with_empty=False,
        cuda_kwargs=None,
        dataset_type='nusc',
        empty_label=17,
        render_config=None,
        flow_grad_scale=1.0,
        **kwargs,
    ):
        super().__init__(init_cfg)

        self.num_classes = num_classes
        self.aggregator = LocalAggregator(**cuda_kwargs)
        if with_empty:
            self.empty_scalar = nn.Parameter(torch.ones(1, dtype=torch.float)*10)
            self.register_buffer('empty_mean', torch.tensor(empty_args['mean'])[None, None, :])
            self.register_buffer('empty_scale', torch.tensor(empty_args['scale'])[None, None, :])
            self.register_buffer('empty_rot', torch.tensor([1., 0., 0., 0.])[None, None, :])
            self.register_buffer('empty_sem', torch.zeros(self.num_classes)[None, None, :])
            self.register_buffer('empty_opa', torch.ones(1)[None, None, :])
        self.with_emtpy = with_empty
        self.empty_args = empty_args
        self.dataset_type = dataset_type
        self.empty_label = empty_label
        self.pc_range = [-30.0, -30.0, -2.0, 30.0, 30.0, 2.0]
        pc_min = [-30.0, -30.0, -2.0]
        self.register_buffer('pc_min', torch.tensor(pc_min, dtype=torch.float).unsqueeze(0))
        grid_size = [0.5, 0.5, 0.5]
        self.register_buffer('grid_size', torch.tensor(grid_size, dtype=torch.float).unsqueeze(0))

        self.register_buffer('zero_tensor', torch.zeros(1, dtype=torch.float))

        # v12 (P1): fraction of the OccFlowLoss gradient allowed to reach the
        # CURRENT-frame gaussians (means / opacity / semantics / scales) through
        # the future-occ branch. The future occ is built as
        # ``current gaussians (+offset) - ego``, so without this gate the future
        # loss also reshapes the current-frame representation -- a coupling that
        # ``decouple_offset`` does NOT cover (it only gates offset -> encoder).
        #   1.0 -> legacy behaviour (fully coupled)
        #   0.0 -> the future branch trains ONLY the offset; the current frame
        #          is decided solely by OccupancyLoss + DetectionLoss.
        # Straight-through: the forward value is unchanged, only the backward
        # pass is scaled, so the future prediction still uses the real gaussians.
        self.flow_grad_scale = flow_grad_scale

        # 2D Gaussian splatting renderer (pseudo-label supervision)
        self.rasterizer_2d = None
        if render_config is not None:
            from .gaussian_rasterizer import GaussianRasterizer2D
            self.rasterizer_2d = GaussianRasterizer2D(**render_config)

    def init_weights(self):
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    def _flow_blend(self, x):
        """Straight-through gradient scaling for the future-occ branch (v12/P1).

        ``s*x + (1-s)*x.detach()`` has the SAME forward value as ``x`` but its
        gradient is scaled by ``s``, so the future-occ loss can be prevented
        from reshaping the current-frame gaussians without changing what the
        future branch actually sees.
        """
        s = self.flow_grad_scale
        if s >= 1.0:
            return x
        if s <= 0.0:
            return x.detach()
        return s * x + (1.0 - s) * x.detach()

    def _sampling(self, gt_xyz, gt_label, gt_mask=None):
        if gt_mask is None:
            gt_label = gt_label.flatten(1)
            gt_xyz = gt_xyz.flatten(1, 3)
        else:
            assert gt_label.shape[0] == 1, "OccLoss does not support bs > 1"
            gt_label = gt_label[gt_mask].reshape(1, -1)
            gt_xyz = gt_xyz[gt_mask].reshape(1, -1, 3)
        return gt_xyz, gt_label

    def prepare_gaussian_args(self, gaussians):
        means = gaussians.means # b, g, 3
        scales = gaussians.scales # b, g, 3
        rotations = gaussians.rotations # b, g, 4
        opacities = gaussians.semantics # b, g, c
        origi_opa = gaussians.opacities # b, g, 1
        if origi_opa.numel() == 0:
            origi_opa = torch.ones_like(opacities[..., :1], requires_grad=False)
        if self.with_emtpy:
            assert opacities.shape[-1] == self.num_classes - 1
            if 'kitti' in self.dataset_type:
                opacities = torch.cat([torch.zeros_like(opacities[..., :1]), opacities], dim=-1)
            else:
                opacities = torch.cat([opacities, torch.zeros_like(opacities[..., :1])], dim=-1)
            means = torch.cat([means, self.empty_mean], dim=1)
            scales = torch.cat([scales, self.empty_scale], dim=1)
            rotations = torch.cat([rotations, self.empty_rot], dim=1)
            empty_sem = self.empty_sem.clone()
            empty_sem[..., self.empty_label] += self.empty_scalar
            opacities = torch.cat([opacities, empty_sem], dim=1)
            origi_opa = torch.cat([origi_opa, self.empty_opa], dim=1)

        bs, g, _ = means.shape
        S = torch.zeros(bs, g, 3, 3, dtype=means.dtype, device=means.device)
        S[..., 0, 0] = scales[..., 0]
        S[..., 1, 1] = scales[..., 1]
        S[..., 2, 2] = scales[..., 2]
        R = get_rotation_matrix(rotations) # b, g, 3, 3
        M = torch.matmul(S, R)
        Cov = torch.matmul(M.transpose(-1, -2), M)
        CovInv = Cov.cpu().inverse().cuda() # b, g, 3, 3
        return means, origi_opa, opacities, scales, CovInv

    def get_filtered_lidar(self,lidar):
        means3D_int = ((lidar - self.pc_min) / self.grid_size).to(torch.int)
        mask_x = (means3D_int[:, 0] >= 0) & (means3D_int[:, 0] < 120)
        mask_y = (means3D_int[:, 1] >= 0) & (means3D_int[:, 1] < 120)
        mask_z = (means3D_int[:, 2] >= 0) & (means3D_int[:, 2] < 8)
        mask = mask_x & mask_y & mask_z
        # squeeze(-1) not squeeze(): when exactly ONE gaussian is in the grid,
        # nonzero returns (1,1) and squeeze() collapses it to a 0-d scalar ->
        # len(mask) raises "len() of a 0-d tensor". squeeze(-1) keeps it 1-D
        # for any count (0, 1, or many). Edge case exposed by GT-ego motion
        # pushing almost all gaussians out of the grid.
        mask = torch.nonzero(mask).squeeze(-1)
        valid = True
        if len(mask) == 0:
            valid = False
            return lidar, mask, valid
        return lidar[mask].unsqueeze(0).contiguous(), mask, valid

    def forward_flow(self,
                    sampled_xyz,
                    representation_temp,
                    metas=None,
                    gs=None,
                    **kwargs):
        offset = kwargs['offset'].reshape(1,-1,6,2)
        zeros = torch.zeros((*offset.shape[:-1],1)).to(offset.device)
        offset = torch.cat((offset, zeros), dim=-1)

        gaussian = representation_temp['gaussian']
        # v12 (P1): gate the gradient the future branch sends back into the
        # current-frame gaussians (see self.flow_grad_scale). Forward values are
        # untouched, so the future occ is still built from the real gaussians.
        means = self._flow_blend(gaussian.means)
        gs = tuple(self._flow_blend(t) for t in gs)
        means_fut = means[...,None,:] + offset
        pred_flow = []
        # Ego motion for occ_flow: use the GT ego trajectory, NOT the planner
        # prediction. planner_head is frozen (random init) in this oracle
        # experiment, so ego_fut_preds is garbage. GT ego keeps offset's world
        # frame consistent with PhysicsLoss.loss_vel (target = v_box*t, no ego).
        ego_gt = metas['ego_fut_trajs']
        if not torch.is_tensor(ego_gt):
            ego_gt = torch.as_tensor(ego_gt)
        ego_gt = ego_gt.to(offset.device).float()        # (B, 6, 2) per-step
        if ego_gt.dim() == 2:
            ego_gt = ego_gt[None]                         # (1, 6, 2)
        ego_gt = torch.nan_to_num(ego_gt, nan=0.0, posinf=0.0, neginf=0.0)
        planner_res = ego_gt.cumsum(dim=1)                # (1, 6, 2) cumulative
        planner_res = torch.cat((planner_res, torch.zeros(*planner_res.shape[:-1],1).to(planner_res.device)), dim=-1)
        for i in range(6):
            origi_opa, opacities, scales, CovInv = gs
            mean_single = means_fut[...,i,:]
            mean_single = mean_single - planner_res[:,i:i+1,:]
            mean_single, mask, valid = self.get_filtered_lidar(mean_single[0])

            if not valid:#valid来自于get_filtered_lidar函数，表示当前帧的预测点云是否在范围内，如果不在范围内，则将mean_single设置为原始的means，并将flow_valid_flag设置为0，表示该帧的预测流无效。
                mean_single = means
                # gs tensors carry the extra empty gaussian (G+1); means is the
                # raw G gaussians. The valid branch drops the empty via mask
                # (indices 0..G-1 never select it), so match that here by
                # slicing gs to G — otherwise reshape(bs, G) on a G+1 tensor
                # crashes. This fallback is only hit when GT-ego motion pushes
                # all gaussians out of the grid (frozen-planner ego~0 never did).
                g0 = means.shape[1]
                origi_opa = origi_opa[:, :g0]
                opacities = opacities[:, :g0]
                scales = scales[:, :g0]
                CovInv = CovInv[:, :g0]
                metas['flow_info'][0][i]['flow_valid_flag'] = 0
            else:
                origi_opa = origi_opa.squeeze(0)[mask].unsqueeze(0)
                opacities = opacities.squeeze(0)[mask].unsqueeze(0)
                scales = scales.squeeze(0)[mask].unsqueeze(0)
                CovInv = CovInv.squeeze(0)[mask].unsqueeze(0)

            bs, g = mean_single.shape[:2]
            semantics = self.aggregator(
                sampled_xyz.clone().float(),
                mean_single,
                origi_opa.reshape(bs, g),
                opacities,
                scales,
                CovInv)[None].transpose(1, 2)
            pred_flow.append([{'pred_flow': semantics,
                               'sampled_label': metas['flow_info'][0][i]['occ_label'],
                               'flow_valid_flag':metas['flow_info'][0][i]['flow_valid_flag']}])
        return pred_flow

    def forward(
        self,
        representation_temp,
        metas=None,
        **kwargs
    ):
        prediction = []
        occ_xyz = metas['occ_xyz'].to(self.zero_tensor.device)
        occ_label = metas['occ_label'].to(self.zero_tensor.device)
        occ_cam_mask = metas['occ_cam_mask'].to(self.zero_tensor.device)
        sampled_xyz, sampled_label = self._sampling(occ_xyz, occ_label, None)
        gaussians = representation_temp['gaussian']

        means, origi_opa, opacities, scales, CovInv = self.prepare_gaussian_args(gaussians)
        bs, g = means.shape[:2]

        semantics = self.aggregator(
            sampled_xyz.clone().float(),
            means,
            origi_opa.reshape(bs, g),
            opacities,
            scales,
            CovInv)[None].transpose(1, 2) # 1, c, n

        prediction.append(semantics)

        occ_flow = self.forward_flow(
            sampled_xyz,
            representation_temp,
            metas=metas,
            gs=(origi_opa, opacities, scales, CovInv),
            **kwargs)

        output = {
            'pred_occ': prediction,
            'sampled_label': sampled_label,
            'sampled_xyz': sampled_xyz,
            'occ_mask': occ_cam_mask,
            'gaussian': representation_temp['gaussian'],
            'occ_flow': occ_flow,
            'offset': kwargs.get('offset'),
            'dynamic_logits': getattr(representation_temp['gaussian'], 'dynamic_logits', None),
        }

        # 2D splatting for pseudo-label supervision (training only)
        if self.rasterizer_2d is not None:
            if self.training:
                gs_extrins = metas['gs_extrins'].to(self.zero_tensor.device)  # (B, nC, 4, 4)
                gs_intrins = metas['gs_intrins'].to(self.zero_tensor.device)  # (B, nC, 3, 3)
                rendered_sem, rendered_depth, rendered_acc, rendered_var = self.rasterizer_2d(gaussians, gs_extrins, gs_intrins)
                rendered_extra_depth = self._render_extra_frames(gaussians, metas, kwargs)
            else:
                # eval: skip expensive rendering, output None so loss returns 0
                rendered_sem, rendered_depth, rendered_acc, rendered_var, rendered_extra_depth = None, None, None, None, None
            output['rendered_sem'] = rendered_sem
            output['rendered_depth'] = rendered_depth
            output['rendered_acc'] = rendered_acc
            output['rendered_var'] = rendered_var
            output['rendered_extra_depth'] = rendered_extra_depth

        # Dynamic rendering (for DynamicLoss) — gs_extrins/gs_intrins already set above
        if self.rasterizer_2d is not None and self.training and getattr(gaussians, 'dynamic_logits', None) is not None:
            rendered_dynamic = self.rasterizer_2d.render_dynamic(gaussians, gs_extrins, gs_intrins)
        else:
            rendered_dynamic = None
        output['rendered_dynamic'] = rendered_dynamic

        return output

    def _render_extra_frames(self, gaussians, metas, kwargs):
        """Render multi-frame (history + future) depth for pseudo supervision.

        History frames render gaussians at original means (dynamic regions are
        masked in the pseudo depth). Future frames move gaussians by the
        predicted offset[idx] before rendering. Returns (B, K, nC, H, W) or None.
        """
        if metas.get('extra_gs_extrins', None) is None:
            return None
        extra_extrins = metas['extra_gs_extrins'].to(self.zero_tensor.device)  # (B, K, nC, 4, 4)
        extra_intrins = metas['extra_gs_intrins'].to(self.zero_tensor.device)  # (B, K, nC, 3, 3)
        offset_idx = metas['extra_offset_idx']   # (B, K)
        valid = metas['extra_valid']             # (B, K)
        B, K = offset_idx.shape[:2]

        offset = kwargs.get('offset', None)
        if offset is not None:
            offset = offset.reshape(1, -1, 6, 2)  # (1, G, 6, 2)

        means = gaussians.means        # (B, G, 3)
        quats = gaussians.rotations    # (B, G, 4)
        scales = gaussians.scales      # (B, G, 3)
        opa = gaussians.opacities      # (B, G, 1)
        H, W = self.rasterizer_2d.height, self.rasterizer_2d.width
        out = means.new_zeros((B, K, extra_extrins.shape[2], H, W))
        for b in range(B):
            for k in range(K):
                if not bool(valid[b, k]):
                    continue
                idx = int(offset_idx[b, k])
                if idx < 0:
                    means_k = means[b]
                else:
                    if offset is None:
                        continue  # no motion prediction available → skip future frame
                    off = offset[b, :, idx]                       # (G, 2)
                    off = torch.cat([off, off.new_zeros((off.shape[0], 1))], dim=-1)  # (G, 3)
                    means_k = means[b] + off
                depth_k = self.rasterizer_2d.render_depth_only(
                    means_k, quats[b], scales[b], opa[b, :, 0],
                    extra_extrins[b, k], extra_intrins[b, k])
                out[b, k] = depth_k
        return out  # (B, K, nC, H, W)

