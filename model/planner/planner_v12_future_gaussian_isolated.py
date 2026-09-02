"""FutAttn-Global-Residual consuming explicit predicted Future Gaussians."""

import torch

from mmengine.registry import MODELS

from .planner_v12 import VADHeadFutAttnGlobalResidual


@MODELS.register_module()
class VADHeadFutAttnGlobalResidualFutureGaussianIsolated(
        VADHeadFutAttnGlobalResidual):
    """Best V12 Planner with a mandatory GT-free Future-Gaussian interface."""

    _FORBIDDEN_INPUTS = {
        'metas',
        'gt_boxes',
        'ego_fut_trajs',
        'future_lidar2global',
        'flow_info',
        'occ_label',
    }

    def _build_future_gaussians(self, results):
        leaked = self._FORBIDDEN_INPUTS.intersection(results)
        if leaked:
            raise AssertionError(
                'GT/label fields reached Planner inputs: '
                f'{sorted(leaked)}')
        if 'planner_future_gaussians' not in results:
            raise KeyError(
                'planner_future_gaussians is mandatory; fallback to '
                'current Gaussian + oracle offset is forbidden')
        if 'planner_future_gaussian_mask' not in results:
            raise KeyError('planner_future_gaussian_mask is mandatory')

        future_features = results['planner_future_gaussians']
        padding_mask = results['planner_future_gaussian_mask'].bool()
        if future_features.dim() != 4:
            raise ValueError(
                'planner_future_gaussians must be (B,T,K,28), got '
                f'{tuple(future_features.shape)}')
        batch, timesteps, count, width = future_features.shape
        if timesteps != self.fut_ts or width != 28:
            raise ValueError(
                f'expected (B,{self.fut_ts},K,28), got '
                f'{tuple(future_features.shape)}')
        if padding_mask.shape != (batch, timesteps, count):
            raise ValueError(
                'future Gaussian mask shape mismatch: '
                f'{tuple(padding_mask.shape)}')
        if padding_mask.all(dim=-1).any():
            raise ValueError('a future timestep contains no valid Gaussian')

        future_features = torch.nan_to_num(
            future_features, nan=0.0, posinf=0.0, neginf=0.0)
        future_content = self.fut_gaussian_fus_mlp(future_features)
        time_encoding = self._build_time_encoding(future_content)
        future_key = future_content + time_encoding[None, :, None, :]
        return future_content, future_key, time_encoding, padding_mask

    def forward(self, results):
        agent_query, agent_mask = self.prepare_agent_query(results)
        map_query = self.prepare_map_query(results)
        gaussian_query = self.gaussian_fus_mlp(results['gaussian_output'])

        batch = agent_query.shape[0]
        ego_query = self.ego_query.weight.unsqueeze(0).expand(batch, -1, -1)
        ego_agent_query = self.ego_agent_decoder(
            query=ego_query.permute(1, 0, 2),
            key=agent_query.permute(1, 0, 2),
            value=agent_query.permute(1, 0, 2),
            key_padding_mask=agent_mask)
        ego_map_query = self.ego_map_decoder(
            query=ego_agent_query,
            key=map_query.permute(1, 0, 2),
            value=map_query.permute(1, 0, 2))
        ego_gaussian_query = self.ego_gaussian_decoder(
            query=ego_map_query,
            key=gaussian_query.permute(1, 0, 2),
            value=gaussian_query.permute(1, 0, 2))

        current_features = torch.cat([
            ego_agent_query.permute(1, 0, 2),
            ego_map_query.permute(1, 0, 2),
            ego_gaussian_query.permute(1, 0, 2),
        ], dim=-1)

        future_content, future_key, _, future_mask = (
            self._build_future_gaussians(results))
        num_gaussians = future_content.shape[2]

        # Per-frame collision-grounded branch.
        fut_ego = self.ego_to_fut(current_features)
        fut_ego = fut_ego.expand(-1, self.fut_ts, -1)
        fut_ego = fut_ego + self.fut_pos.weight[None]
        fut_query = fut_ego.permute(1, 0, 2)
        if self.fut_self_decoder is not None:
            fut_query = self.fut_self_decoder(
                query=fut_query, key=fut_query, value=fut_query)
        folded_query = fut_query.permute(1, 0, 2).reshape(
            batch * self.fut_ts, self.embed_dims).unsqueeze(0)
        folded_kv = future_content.reshape(
            batch * self.fut_ts, num_gaussians,
            self.embed_dims).permute(1, 0, 2)
        folded_mask = future_mask.reshape(
            batch * self.fut_ts, num_gaussians)
        if self.ego_fut_gaussian_decoder is not None:
            folded_query = self.ego_fut_gaussian_decoder(
                query=folded_query,
                key=folded_kv,
                value=folded_kv,
                key_padding_mask=folded_mask)
        per_frame_features = folded_query.squeeze(0).reshape(
            batch, self.fut_ts, self.embed_dims)
        per_frame_prediction = self.fut_out_mlp(per_frame_features).reshape(
            batch, self.fut_ts, self.ego_fut_mode, 2).permute(0, 2, 1, 3)

        # Global low-L2 branch.
        global_key = future_key.reshape(
            batch, self.fut_ts * num_gaussians,
            self.embed_dims).permute(1, 0, 2)
        global_value = future_content.reshape(
            batch, self.fut_ts * num_gaussians,
            self.embed_dims).permute(1, 0, 2)
        global_mask = future_mask.reshape(
            batch, self.fut_ts * num_gaussians)
        if self.global_fut_gaussian_decoder is not None:
            global_future_query = self.global_fut_gaussian_decoder(
                query=ego_gaussian_query,
                key=global_key,
                value=global_value,
                key_padding_mask=global_mask)
        else:
            global_future_query = ego_gaussian_query
        global_features = torch.cat([
            current_features,
            global_future_query.permute(1, 0, 2),
        ], dim=-1)
        global_prediction = self.global_shape_mlp(global_features).reshape(
            batch, self.ego_fut_mode, self.fut_ts, 2)

        gate_in = torch.cat([
            global_features.expand(-1, self.fut_ts, -1),
            per_frame_features,
        ], dim=-1)
        gate = self.refine_gate_mlp(gate_in).tanh().reshape(
            batch, 1, self.fut_ts, 1)
        time_scale = (0.2 + 0.8 * torch.arange(
            1, self.fut_ts + 1,
            device=gate.device,
            dtype=gate.dtype) / self.fut_ts).reshape(1, 1, self.fut_ts, 1)
        gate = gate * time_scale
        main_prediction = per_frame_prediction + gate * (
            global_prediction - per_frame_prediction)

        return {
            'ego_fut_preds': main_prediction,
            'ego_fut_aux_preds': global_prediction,
            'ego_fut_per_frame_preds': per_frame_prediction,
        }
