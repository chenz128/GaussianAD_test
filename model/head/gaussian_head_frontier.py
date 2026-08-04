import torch
import torch.nn.functional as F

from mmengine.registry import MODELS

from .frontier_generator import FrontierGenerator
from .gaussian_head import GaussianHead


@MODELS.register_module()
class GaussianHeadFrontier(GaussianHead):
    """GaussianHead whose future branch keeps a constant gaussian count.

    ``GaussianHead.forward_flow`` drops every gaussian that leaves the occ
    window, so by t+3s roughly 40% of the slots are gone and the band that just
    entered in front of the ego has no contributor at all. Here those slots are
    recycled instead: they are re-placed inside the newly-visible band and their
    attributes are produced by ``FrontierGenerator``, supervised end-to-end by
    the existing future-occ loss.
    """

    def __init__(self, frontier_generator=None, **kwargs):
        super().__init__(**kwargs)
        cfg = dict(frontier_generator or {})
        cfg.setdefault('pc_range', tuple(self.pc_range))
        cfg.setdefault(
            'num_classes',
            self.num_classes - 1 if self.with_emtpy else self.num_classes)
        self.frontier_generator = FrontierGenerator(**cfg)

    def get_in_range_mask(self, lidar):
        """Same grid test as ``get_filtered_lidar`` but returns a bool mask."""
        grid = ((lidar - self.pc_min) / self.grid_size).to(torch.int)
        return (
            (grid[..., 0] >= 0) & (grid[..., 0] < 120)
            & (grid[..., 1] >= 0) & (grid[..., 1] < 120)
            & (grid[..., 2] >= 0) & (grid[..., 2] < 8)
        )

    def forward_flow(self, sampled_xyz, representation_temp, metas=None,
                     gs=None, **kwargs):
        offset = kwargs['offset'].reshape(1, -1, 6, 2)
        offset = torch.cat(
            [offset, offset.new_zeros(*offset.shape[:-1], 1)], dim=-1)

        gaussian = representation_temp['gaussian']
        means = self._flow_blend(gaussian.means)
        gs = tuple(self._flow_blend(t) for t in gs)
        means_fut = means[..., None, :] + offset

        ego_gt = metas['ego_fut_trajs']
        if not torch.is_tensor(ego_gt):
            ego_gt = torch.as_tensor(ego_gt)
        ego_gt = ego_gt.to(offset.device).float()
        if ego_gt.dim() == 2:
            ego_gt = ego_gt[None]
        ego_gt = torch.nan_to_num(ego_gt, nan=0.0, posinf=0.0, neginf=0.0)
        ego_cum = ego_gt.cumsum(dim=1)
        ego_cum = torch.cat(
            [ego_cum, ego_cum.new_zeros(*ego_cum.shape[:-1], 1)], dim=-1)

        num_real = means.shape[1]
        pred_flow = []
        for i in range(6):
            origi_opa, sem_all, scales, cov_inv = gs
            mean_i = means_fut[..., i, :] - ego_cum[:, i:i + 1, :]
            inside = self.get_in_range_mask(mean_i)

            gen = self.frontier_generator(
                ego_disp=ego_cum[:, i], num_gaussians=num_real, time_index=i)

            keep = inside[..., None]
            keep_cov = inside[..., None, None]
            gen_sem = gen['semantics']
            if gen_sem.shape[-1] < sem_all.shape[-1]:
                gen_sem = F.pad(
                    gen_sem, (0, sem_all.shape[-1] - gen_sem.shape[-1]))

            mean_i = torch.where(keep, mean_i, gen['means'])
            opa_i = torch.where(keep, origi_opa[:, :num_real], gen['opacities'])
            sem_i = torch.where(keep, sem_all[:, :num_real], gen_sem)
            sca_i = torch.where(keep, scales[:, :num_real], gen['scales'])
            cov_i = torch.where(keep_cov, cov_inv[:, :num_real], gen['cov_inv'])

            if self.with_emtpy and self.flow_include_empty:
                mean_i = torch.cat(
                    [mean_i, self.empty_mean.to(mean_i.dtype)], dim=1)
                opa_i = torch.cat([opa_i, origi_opa[:, -1:]], dim=1)
                sem_i = torch.cat([sem_i, sem_all[:, -1:]], dim=1)
                sca_i = torch.cat([sca_i, scales[:, -1:]], dim=1)
                cov_i = torch.cat([cov_i, cov_inv[:, -1:]], dim=1)

            bs, g = mean_i.shape[:2]
            semantics = self.aggregator(
                sampled_xyz.clone().float(),
                mean_i,
                opa_i.reshape(bs, g),
                sem_i,
                sca_i,
                cov_i)[None].transpose(1, 2)

            pred_flow.append([{
                'pred_flow': semantics,
                'sampled_label': metas['flow_info'][0][i]['occ_label'],
                'flow_valid_flag': metas['flow_info'][0][i]['flow_valid_flag'],
                'frontier_ratio': (~inside).float().mean().detach(),
            }])
        return pred_flow
