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

    def init_weights(self):
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

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
        CovInv = Cov.float().cpu().inverse().cuda() # b, g, 3, 3; keep float32 (local_aggregate needs fp32)
        return means, origi_opa, opacities, scales, CovInv

    def get_filtered_lidar(self,lidar):
        means3D_int = ((lidar - self.pc_min) / self.grid_size).to(torch.int)
        mask_x = (means3D_int[:, 0] >= 0) & (means3D_int[:, 0] < 120)
        mask_y = (means3D_int[:, 1] >= 0) & (means3D_int[:, 1] < 120)
        mask_z = (means3D_int[:, 2] >= 0) & (means3D_int[:, 2] < 8)
        mask = mask_x & mask_y & mask_z
        mask = torch.nonzero(mask).squeeze()
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
        means = gaussian.means
        means_fut = means[...,None,:] + offset
        pred_flow = []
        cmd = metas['ego_fut_cmd'].argmax(dim=-1)
        planner_res = kwargs['ego_fut_preds'].cumsum(dim=1)[0,cmd,...]
        planner_res = torch.cat((planner_res, torch.zeros(*planner_res.shape[:-1],1).to(planner_res.device)), dim=-1)
        for i in range(6):
            origi_opa, opacities, scales, CovInv = gs
            mean_single = means_fut[...,i,:]
            mean_single = mean_single - planner_res[:,i:i+1,:]
            mean_single, mask, valid = self.get_filtered_lidar(mean_single[0])

            if not valid:
                mean_single = means
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

        return {
            'pred_occ': prediction,
            'sampled_label': sampled_label,
            'sampled_xyz': sampled_xyz,
            'occ_mask': occ_cam_mask,
            'gaussian': representation_temp['gaussian'],
            'occ_flow': occ_flow
        }

