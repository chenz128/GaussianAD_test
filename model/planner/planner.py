import copy

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from mmengine.registry import MODELS
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence


@MODELS.register_module()
class VADHead(nn.Module):
    def __init__(self,
                 embed_dims=128,
                 fut_ts=6,
                 fut_mode=6,
                 ego_fut_mode=3,
                 ego_agent_decoder=None,
                 ego_map_decoder=None,
                 ego_gaussian_decoder=None,
                 **kwargs):
        super(VADHead, self).__init__()

        self.embed_dims = embed_dims
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.ego_fut_mode = ego_fut_mode
        self.ego_agent_decoder = ego_agent_decoder
        self.ego_map_decoder = ego_map_decoder
        self.ego_gaussian_decoder = ego_gaussian_decoder

        self._init_layers()

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""

        self.ego_query = nn.Embedding(1, self.embed_dims)

        if self.ego_agent_decoder is not None:
            self.ego_agent_decoder = build_transformer_layer_sequence(self.ego_agent_decoder)

        if self.ego_map_decoder is not None:
            self.ego_map_decoder = build_transformer_layer_sequence(self.ego_map_decoder)

        if self.ego_gaussian_decoder is not None:
            self.ego_gaussian_decoder = build_transformer_layer_sequence(self.ego_gaussian_decoder)

        ego_fut_decoder = []
        ego_fut_dec_in_dim = self.embed_dims * 3
        for _ in range(2):
            ego_fut_decoder.append(nn.Linear(ego_fut_dec_in_dim, ego_fut_dec_in_dim))
            ego_fut_decoder.append(nn.ReLU())
        ego_fut_decoder.append(nn.Linear(ego_fut_dec_in_dim, self.ego_fut_mode * self.fut_ts * 2))
        self.ego_fut_decoder = nn.Sequential(*ego_fut_decoder)

        self.agent_fus_mlp = nn.Sequential(
            nn.Linear(10, self.embed_dims, bias=True),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims, bias=True))

        self.map_fus_mlp = nn.Sequential(
            nn.Linear(43, self.embed_dims, bias=True),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims, bias=True))

        self.gaussian_fus_mlp = nn.Sequential(
            nn.Linear(28, self.embed_dims, bias=True),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims, bias=True))

    def init_weights(self):
        """Initialize weights"""
        if self.ego_agent_decoder is not None:
            for p in self.ego_agent_decoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        if self.ego_map_decoder is not None:
            for p in self.ego_map_decoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        if self.ego_gaussian_decoder is not None:
            for p in self.ego_gaussian_decoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def prepare_agent_query(self, results, query_num=500):
        box_dict = results["final_box_dicts"]

        pred_boxe = box_dict[0]['pred_boxes']
        pred_score = box_dict[0]['pred_scores']
        pred = torch.hstack([pred_boxe, pred_score[:, None]]).unsqueeze(0)
        pred = self.agent_fus_mlp(pred)[0]
        pad_mask = torch.tensor([False], device=pred.device).repeat(query_num)
        if pred.shape[0] < query_num:
            pad_qnum = query_num - pred.shape[0]
            pad_mask[pred.shape[0]:] = True
            pad_query = torch.zeros((pad_qnum, pred.shape[-1]), device=pred.device)
            pred = torch.cat([pred, pad_query], dim=0)
        return pred.unsqueeze(0), pad_mask.unsqueeze(0)

    def prepare_map_query(self, results):
        cls_scores = results["all_cls_scores"][0][0]
        pts_preds = results["all_pts_preds"][0][0].view(cls_scores.shape[0], 40)
        pred = torch.hstack([cls_scores, pts_preds])
        pred = self.map_fus_mlp(pred)
        return pred.unsqueeze(0)

    def forward(self,
                results
            ):
        agent_query, agent_mask = self.prepare_agent_query(results)
        map_query = self.prepare_map_query(results)
        gaussian_query = self.gaussian_fus_mlp(results['gaussian_output'])

        # planning
        batch = agent_query.shape[0]
        ego_his_feats = self.ego_query.weight.unsqueeze(0).repeat(batch, 1, 1)
        ego_query = ego_his_feats

        # ego <-> agent interaction
        ego_agent_query = self.ego_agent_decoder(
            query=ego_query.permute(1, 0, 2),
            key=agent_query.permute(1, 0, 2),
            value=agent_query.permute(1, 0, 2),
            key_padding_mask=agent_mask)

        # ego <-> map interaction
        ego_map_query = self.ego_map_decoder(
            query=ego_agent_query,
            key=map_query.permute(1, 0, 2),
            value=map_query.permute(1, 0, 2))

        # ego <-> gaussian interaction
        ego_gs_query = self.ego_gaussian_decoder(
            query=ego_map_query,
            key=gaussian_query.permute(1, 0, 2),
            value=gaussian_query.permute(1, 0, 2))

        ego_feats = torch.cat(
            [ego_agent_query.permute(1, 0, 2),
                ego_map_query.permute(1, 0, 2),
                ego_gs_query.permute(1, 0, 2)],
            dim=-1
        )  # [B, 1, 2D]

        # Ego prediction
        outputs_ego_trajs = self.ego_fut_decoder(ego_feats)
        outputs_ego_trajs = outputs_ego_trajs.reshape(outputs_ego_trajs.shape[0],
                                                      self.ego_fut_mode, self.fut_ts, 2)

        outs = {'ego_fut_preds': outputs_ego_trajs}

        return outs
