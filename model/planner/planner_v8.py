import math

import torch
import torch.nn as nn
from mmengine.registry import MODELS

from .planner_v7 import VADHeadDualTimeResidual


class GaussianResidualDiTBlock(nn.Module):
    """DiT block with temporal self-attention and local Gaussian attention."""

    def __init__(self, embed_dims, num_heads, feedforward_channels, dropout):
        super().__init__()
        self.self_norm = nn.LayerNorm(embed_dims, elementwise_affine=False)
        self.gaussian_norm = nn.LayerNorm(embed_dims, elementwise_affine=False)
        self.ffn_norm = nn.LayerNorm(embed_dims, elementwise_affine=False)
        self.condition_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(embed_dims, embed_dims * 6))
        self.self_attention = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=dropout, batch_first=True)
        self.gaussian_attention = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, feedforward_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_channels, embed_dims),
        )

    @staticmethod
    def _modulate(value, shift, scale):
        return value * (1.0 + scale) + shift

    def forward(self, tokens, condition, local_gaussians):
        batch, modes, timesteps, channels = tokens.shape
        modulation = self.condition_modulation(condition)
        self_shift, self_scale, gaussian_shift, gaussian_scale, \
            ffn_shift, ffn_scale = modulation.chunk(6, dim=-1)

        self_tokens = self._modulate(
            self.self_norm(tokens), self_shift, self_scale)
        self_tokens = self_tokens.reshape(batch * modes, timesteps, channels)
        self_tokens = self.self_attention(
            self_tokens, self_tokens, self_tokens, need_weights=False)[0]
        tokens = tokens + self_tokens.reshape(
            batch, modes, timesteps, channels)

        gaussian_tokens = self._modulate(
            self.gaussian_norm(tokens), gaussian_shift, gaussian_scale)
        gaussian_tokens = gaussian_tokens.permute(0, 2, 1, 3).reshape(
            batch * timesteps * modes, 1, channels)
        local_gaussians = local_gaussians.reshape(
            batch * timesteps * modes, local_gaussians.shape[-2], channels)
        gaussian_tokens = self.gaussian_attention(
            gaussian_tokens, local_gaussians, local_gaussians,
            need_weights=False)[0]
        gaussian_tokens = gaussian_tokens.reshape(
            batch, timesteps, modes, channels).permute(0, 2, 1, 3)
        tokens = tokens + gaussian_tokens

        ffn_tokens = self._modulate(
            self.ffn_norm(tokens), ffn_shift, ffn_scale)
        return tokens + self.ffn(ffn_tokens)


@MODELS.register_module()
class VADHeadGaussianResidualDiT(VADHeadDualTimeResidual):
    """Refine a Gaussian-aware deterministic proposal in residual space."""

    def __init__(self,
                 *args,
                 dit_num_layers=2,
                 dit_num_heads=8,
                 dit_feedforward_channels=512,
                 dit_dropout=0.1,
                 local_gaussian_topk=128,
                 num_diffusion_timesteps=100,
                 diffusion_truncation_step=20,
                 num_inference_steps=2,
                 beta_start=1e-4,
                 beta_end=2e-2,
                 residual_scale=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
                 **kwargs):
        self.dit_num_layers = dit_num_layers
        self.dit_num_heads = dit_num_heads
        self.dit_feedforward_channels = dit_feedforward_channels
        self.dit_dropout = dit_dropout
        self.local_gaussian_topk = local_gaussian_topk
        self.num_diffusion_timesteps = num_diffusion_timesteps
        self.diffusion_truncation_step = diffusion_truncation_step
        self.num_inference_steps = num_inference_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.residual_scale_values = residual_scale
        super().__init__(*args, **kwargs)

    def _init_layers(self):
        super()._init_layers()
        if len(self.residual_scale_values) != self.fut_ts:
            raise ValueError(
                'residual_scale must contain one value per future timestep')
        if not 1 <= self.diffusion_truncation_step \
                <= self.num_diffusion_timesteps:
            raise ValueError('invalid diffusion_truncation_step')

        self.residual_input = nn.Linear(2, self.embed_dims)
        self.proposal_position = nn.Sequential(
            nn.Linear(2, self.embed_dims),
            nn.SiLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )
        self.diffusion_time_mlp = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims * 4),
            nn.SiLU(),
            nn.Linear(self.embed_dims * 4, self.embed_dims),
        )
        self.chain_condition = nn.Sequential(
            nn.Linear(self.embed_dims * 3, self.embed_dims),
            nn.SiLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )
        self.command_embedding = nn.Embedding(
            self.ego_fut_mode, self.embed_dims)
        self.residual_time_embedding = nn.Embedding(
            self.fut_ts, self.embed_dims)
        self.dit_blocks = nn.ModuleList([
            GaussianResidualDiTBlock(
                self.embed_dims,
                self.dit_num_heads,
                self.dit_feedforward_channels,
                self.dit_dropout)
            for _ in range(self.dit_num_layers)
        ])
        self.output_norm = nn.LayerNorm(
            self.embed_dims, elementwise_affine=False)
        self.output_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(self.embed_dims, self.embed_dims * 2))
        self.noise_output = nn.Linear(self.embed_dims, 2)

        residual_scale = torch.as_tensor(
            self.residual_scale_values, dtype=torch.float32)
        self.register_buffer(
            'residual_scale', residual_scale.view(1, 1, self.fut_ts, 1))
        betas = torch.linspace(
            self.beta_start, self.beta_end,
            self.num_diffusion_timesteps, dtype=torch.float32)
        alpha_bars = torch.cumprod(1.0 - betas, dim=0)
        self.register_buffer(
            'diffusion_alpha_bars',
            torch.cat([torch.ones(1), alpha_bars], dim=0))

    def init_weights(self):
        super().init_weights()
        nn.init.normal_(self.command_embedding.weight, std=0.02)
        nn.init.normal_(self.residual_time_embedding.weight, std=0.02)
        nn.init.zeros_(self.noise_output.weight)
        nn.init.zeros_(self.noise_output.bias)

    def _future_gaussian_xy(self, results):
        gaussian_output = results['gaussian_output']
        batch, num_gaussians = gaussian_output.shape[:2]
        offset = results.get('offset')
        if offset is None:
            future_offset = gaussian_output.new_zeros(
                (batch, self.fut_ts, num_gaussians, 2))
        else:
            future_offset = offset.reshape(
                batch, num_gaussians, self.fut_ts, 2).permute(0, 2, 1, 3)
            future_offset = torch.nan_to_num(
                future_offset, nan=0.0, posinf=0.0, neginf=0.0)
        return gaussian_output[:, None, :, :2] + future_offset

    def _forward_chain(self, results):
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
        future_content, future_global_kv = \
            self._build_future_gaussian_features(results)
        num_gaussians = future_content.shape[2]
        global_future_query = self.ego_fut_gaussian_decoder(
            query=ego_gaussian_query,
            key=future_global_kv.reshape(
                batch, self.fut_ts * num_gaussians,
                self.embed_dims).permute(1, 0, 2),
            value=future_global_kv.reshape(
                batch, self.fut_ts * num_gaussians,
                self.embed_dims).permute(1, 0, 2))
        global_features = torch.cat([
            current_features,
            global_future_query.permute(1, 0, 2),
        ], dim=-1)
        global_prediction = self.ego_fut_decoder(global_features).reshape(
            batch, self.ego_fut_mode, self.fut_ts, 2)

        time_query = self.ego_to_fut(current_features).expand(
            -1, self.fut_ts, -1)
        time_query = time_query + self.fut_pos.weight[None, :, :]
        time_query = time_query.permute(1, 0, 2)
        if self.fut_self_decoder is not None:
            time_query = self.fut_self_decoder(
                query=time_query,
                key=time_query,
                value=time_query,
                attn_masks=[self._relative_time_attn_bias(time_query)])

        folded_query = time_query.permute(1, 0, 2).reshape(
            batch * self.fut_ts, self.embed_dims).unsqueeze(0)
        folded_kv = future_content.reshape(
            batch * self.fut_ts, num_gaussians,
            self.embed_dims).permute(1, 0, 2)
        if self.time_gaussian_decoder is not None:
            folded_query = self.time_gaussian_decoder(
                query=folded_query, key=folded_kv, value=folded_kv)

        time_features = folded_query.squeeze(0).reshape(
            batch, self.fut_ts, self.embed_dims)
        future_summary = future_content.mean(dim=2)
        global_context = self.global_context_proj(global_features).expand(
            -1, self.fut_ts, -1)
        fusion_features = torch.cat(
            [global_context, time_features, future_summary], dim=-1)
        residual = self.residual_mlp(fusion_features).reshape(
            batch, self.fut_ts, self.ego_fut_mode, 2).permute(0, 2, 1, 3)
        gate = self.gate_mlp(fusion_features).sigmoid()
        chain_prediction = global_prediction + gate[:, None, :, :] * residual
        return (
            chain_prediction, fusion_features, future_content,
            self._future_gaussian_xy(results))

    def _diffusion_timestep_embedding(self, timesteps):
        half = self.embed_dims // 2
        frequency = torch.exp(
            -math.log(10000.0) * torch.arange(
                half, device=timesteps.device, dtype=torch.float32)
            / max(half - 1, 1))
        phase = timesteps.float()[:, None] * frequency[None, :]
        embedding = torch.cat([phase.sin(), phase.cos()], dim=-1)
        if embedding.shape[-1] < self.embed_dims:
            embedding = torch.cat(
                [embedding, embedding.new_zeros(embedding.shape[0], 1)],
                dim=-1)
        return self.diffusion_time_mlp(embedding)

    def _select_local_gaussians(
            self, candidate_positions, future_xy, future_content):
        modes = candidate_positions.shape[1]
        num_gaussians = future_content.shape[2]
        topk = min(self.local_gaussian_topk, num_gaussians)
        candidate_positions = candidate_positions.permute(0, 2, 1, 3)
        distance = (
            candidate_positions[:, :, :, None, :]
            - future_xy[:, :, None, :, :]).square().sum(dim=-1)
        indices = distance.topk(topk, dim=-1, largest=False).indices
        expanded_content = future_content[:, :, None, :, :].expand(
            -1, -1, modes, -1, -1)
        indices = indices[..., None].expand(
            -1, -1, -1, -1, self.embed_dims)
        return torch.gather(expanded_content, 3, indices)

    def _predict_noise(self, noisy_residual, timesteps, chain_positions,
                       chain_features, future_xy, future_content):
        modes = noisy_residual.shape[1]
        candidate_positions = (
            chain_positions + noisy_residual * self.residual_scale)
        local_gaussians = self._select_local_gaussians(
            candidate_positions, future_xy, future_content)

        condition = self.chain_condition(chain_features)[:, None, :, :]
        condition = condition.expand(-1, modes, -1, -1)
        condition = condition + self.proposal_position(
            chain_positions / self.residual_scale)
        condition = condition + self.command_embedding.weight[
            None, :, None, :]
        condition = condition + self.residual_time_embedding.weight[
            None, None, :, :]
        condition = condition + self._diffusion_timestep_embedding(
            timesteps)[:, None, None, :]

        tokens = self.residual_input(noisy_residual) + condition
        for block in self.dit_blocks:
            tokens = block(tokens, condition, local_gaussians)
        shift, scale = self.output_modulation(condition).chunk(2, dim=-1)
        tokens = self.output_norm(tokens) * (1.0 + scale) + shift
        return self.noise_output(tokens)

    def _predict_clean_residual(self, noisy_residual, noise_prediction,
                                timesteps):
        alpha_bar = self.diffusion_alpha_bars[timesteps].to(
            dtype=noisy_residual.dtype).view(-1, 1, 1, 1)
        clean = (
            noisy_residual - (1.0 - alpha_bar).sqrt() * noise_prediction
        ) / alpha_bar.sqrt().clamp_min(1e-6)
        return clean.clamp(min=-4.0, max=4.0)

    @staticmethod
    def _positions_to_displacements(positions):
        origin = positions.new_zeros((*positions.shape[:2], 1, 2))
        return torch.diff(torch.cat([origin, positions], dim=2), dim=2)

    def _apply_refinement(self, chain_positions, clean_residual):
        return chain_positions + clean_residual * self.residual_scale

    def _training_forward(self, results, chain_prediction, chain_features,
                          future_content, future_xy):
        chain_positions = chain_prediction.cumsum(dim=2)
        ground_truth = results['metas']['ego_fut_trajs']
        while ground_truth.dim() > 3 and ground_truth.shape[1] == 1:
            ground_truth = ground_truth.squeeze(1)
        if ground_truth.dim() == 2:
            ground_truth = ground_truth.unsqueeze(0)
        ground_truth = ground_truth[:, :self.fut_ts, :2]
        ground_truth_positions = ground_truth.cumsum(dim=1)
        clean_residual = (
            ground_truth_positions[:, None, :, :]
            - chain_positions.detach()) / self.residual_scale

        batch = chain_prediction.shape[0]
        timesteps = torch.randint(
            1, self.diffusion_truncation_step + 1,
            (batch,), device=chain_prediction.device)
        alpha_bar = self.diffusion_alpha_bars[timesteps].to(
            dtype=chain_prediction.dtype).view(-1, 1, 1, 1)
        noise_target = torch.randn_like(clean_residual)
        noisy_residual = (
            alpha_bar.sqrt() * clean_residual
            + (1.0 - alpha_bar).sqrt() * noise_target)
        noise_prediction = self._predict_noise(
            noisy_residual, timesteps, chain_positions,
            chain_features, future_xy, future_content)
        deployment_residual = torch.zeros_like(clean_residual)
        deployment_timesteps = torch.full(
            (batch,), self.diffusion_truncation_step,
            device=chain_prediction.device, dtype=torch.long)
        deployment_noise = self._predict_noise(
            deployment_residual, deployment_timesteps, chain_positions,
            chain_features, future_xy, future_content)
        clean_prediction = self._predict_clean_residual(
            deployment_residual, deployment_noise, deployment_timesteps)
        final_positions = self._apply_refinement(
            chain_positions, clean_prediction)
        return {
            'ego_fut_preds': self._positions_to_displacements(final_positions),
            'ego_chain_preds': chain_prediction,
            'residual_diffusion_noise_pred': noise_prediction,
            'residual_diffusion_noise_target': noise_target,
            'residual_diffusion_timesteps': timesteps,
        }

    def _inference_forward(self, chain_prediction, chain_features,
                           future_content, future_xy):
        chain_positions = chain_prediction.cumsum(dim=2)
        residual = chain_prediction.new_zeros(chain_prediction.shape)
        schedule = torch.linspace(
            self.diffusion_truncation_step, 1,
            self.num_inference_steps, device=chain_prediction.device)
        schedule = schedule.round().long()

        for index, timestep in enumerate(schedule):
            timesteps = timestep.expand(chain_prediction.shape[0])
            noise_prediction = self._predict_noise(
                residual, timesteps, chain_positions,
                chain_features, future_xy, future_content)
            clean_prediction = self._predict_clean_residual(
                residual, noise_prediction, timesteps)
            previous_timestep = (
                schedule[index + 1] if index + 1 < len(schedule)
                else timestep.new_zeros(()))
            previous_alpha = self.diffusion_alpha_bars[
                previous_timestep].to(dtype=chain_prediction.dtype)
            residual = (
                previous_alpha.sqrt() * clean_prediction
                + (1.0 - previous_alpha).sqrt() * noise_prediction)

        final_positions = self._apply_refinement(
            chain_positions, residual)
        return {
            'ego_fut_preds': self._positions_to_displacements(final_positions),
            'ego_chain_preds': chain_prediction,
        }

    def forward(self, results):
        chain_prediction, chain_features, future_content, future_xy = \
            self._forward_chain(results)
        if self.training:
            return self._training_forward(
                results, chain_prediction, chain_features,
                future_content, future_xy)
        return self._inference_forward(
            chain_prediction, chain_features, future_content, future_xy)
