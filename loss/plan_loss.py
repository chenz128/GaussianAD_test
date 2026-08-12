import math
import mmcv
import torch
from torch import nn as nn
from torch.nn import functional as F
from mmdet.models import weighted_loss, L1Loss

from . import OPENOCC_LOSS
from .base_loss import BaseLoss


@OPENOCC_LOSS.register_module()
class PlanLoss(BaseLoss):
    def __init__(
        self,
        weight=1,
        loss_weights=None,
        col_loss_weight=0.0,
        col_safe_margin=0.5,
        col_sat=False,
        input_dict=None):
        super().__init__()

        self.weight = weight
        self.loss_weights = loss_weights
        self.input_dict = input_dict
        self.ego_fut_mode = 3

        self.plan_reg_loss = L1Loss()
        self.plan_bound_loss = PlanMapBoundLoss(loss_weight=1.0, dis_thresh=1.0)
        self.plan_dir_loss = PlanMapDirectionLoss(loss_weight=0.5)
        # Collision-avoidance constraint (opt-in). Default weight 0.0 keeps every
        # existing config (base_plan etc.) numerically unchanged; a config that
        # wants it (e.g. futgau_detach_false_col) passes col_loss_weight>0.
        self.col_loss_weight = col_loss_weight
        self.col_sat = col_sat
        if col_loss_weight and col_loss_weight > 0:
            if col_sat:
                # SAT (Separating Axis Theorem) collision: ego + agent as
                # oriented boxes, overlap depth via axis projections.
                self.plan_col_loss = PlanAgentSATCollisionLoss(
                    loss_weight=col_loss_weight, safe_margin=col_safe_margin)
            else:
                self.plan_col_loss = PlanAgentCollisionLoss(
                    loss_weight=col_loss_weight, safe_margin=col_safe_margin)
        else:
            self.plan_col_loss = None

    def get_loss(self, inputs):
        """"Loss function.
        Args:

            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            input:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        map_all_cls_scores = inputs['all_cls_scores'].squeeze(0)
        map_all_pts_preds = inputs['all_pts_preds'].squeeze(0)
        ego_fut_preds = inputs['ego_fut_preds']
        ego_fut_gt = inputs['ego_fut_gt']
        ego_fut_masks = inputs['ego_fut_masks']
        ego_fut_cmd = inputs['ego_fut_cmd']

        # Planning Loss
        ego_fut_gt = ego_fut_gt.squeeze(1)
        ego_fut_masks = ego_fut_masks.squeeze(1).squeeze(1)
        ego_fut_cmd = ego_fut_cmd.squeeze(1).squeeze(1)

        ego_fut_gt = ego_fut_gt.unsqueeze(1).repeat(1, self.ego_fut_mode, 1, 1)
        loss_plan_l1_weight = ego_fut_cmd[..., None, None] * ego_fut_masks[:, None, :, None]
        loss_plan_l1_weight = loss_plan_l1_weight.repeat(1, 1, 1, 2)

        loss_plan_l1 = self.plan_reg_loss(
            ego_fut_preds,
            ego_fut_gt,
            loss_plan_l1_weight,
        )

        loss_plan_bound = self.plan_bound_loss(
            ego_fut_preds[ego_fut_cmd==1],
            map_all_pts_preds,
            map_all_cls_scores.sigmoid(),
            weight=ego_fut_masks
        )

        loss_plan_dir = self.plan_dir_loss(
            ego_fut_preds[ego_fut_cmd==1],
            map_all_pts_preds,
            map_all_cls_scores.sigmoid(),
            weight=ego_fut_masks
        )

        loss_plan_l1 = torch.nan_to_num(loss_plan_l1)
        loss_plan_bound = torch.nan_to_num(loss_plan_bound)
        loss_plan_dir = torch.nan_to_num(loss_plan_dir)

        total = loss_plan_l1 + loss_plan_bound + loss_plan_dir

        # Collision-avoidance constraint (opt-in): push the command-selected ego
        # trajectory outside every valid GT agent's future footprint. It is
        # differentiable w.r.t. ego_fut_preds; the agent GT is a fixed target.
        if self.plan_col_loss is not None:
            metas = inputs.get('metas', None)
            if metas is not None:
                ego_cmd_pred = ego_fut_preds[ego_fut_cmd == 1]  # (B, fut_ts, 2)
                loss_plan_col = self.plan_col_loss(
                    ego_cmd_pred,
                    metas.get('attr_labels_planner', None),
                    metas.get('fut_valid_flag', None),
                    ego_fut_masks,
                    metas.get('gt_boxes', None))
                total = total + torch.nan_to_num(loss_plan_col)

        return total

    def forward(self, inputs):
        loss = self.weight * self.get_loss(inputs)
        return loss


@OPENOCC_LOSS.register_module()
class PlanAgentCollisionLoss(nn.Module):
    """Differentiable collision-avoidance loss (ego <-> GT agents).

    For every future timestep the predicted ego position (cumulative sum of the
    per-step displacement, same convention as ``plan_map_bound_loss``) is pushed
    outside a safety circle around each valid agent's future centre. A circle
    (bounding radius) approximation of both footprints keeps the loss cheap and
    yaw-free while directly targeting the ``plan_obj_box_col`` metric.

    The agent ground truth (positions / sizes / masks) is a fixed target, so the
    gradient flows only into ``ego_fut_preds``.

    Args:
        loss_weight (float): weight of the collision loss.
        safe_margin (float): extra clearance (metres) added on top of the sum of
            the ego and agent bounding-circle radii.
        fut_ts (int): number of future timesteps (default 6).
        ego_width / ego_length (float): ego footprint used for the ego radius.
    """

    def __init__(self, loss_weight=1.0, safe_margin=0.5, fut_ts=6,
                 ego_width=1.85, ego_length=4.084):
        super().__init__()
        self.loss_weight = loss_weight
        self.safe_margin = safe_margin
        self.fut_ts = fut_ts
        # Mean half-extent (isotropic) radius. This is deliberately smaller
        # than the circumscribed half-diagonal so the constraint only fires for
        # genuine near-collisions instead of every normally-spaced neighbour.
        self.ego_radius = 0.25 * (ego_width + ego_length)

    @staticmethod
    def _sample_valid(fut_valid_flag, b):
        if fut_valid_flag is None:
            return True
        v = fut_valid_flag
        if isinstance(v, (list, tuple)):
            v = v[b] if b < len(v) else v[0]
        elif torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] > b:
            v = v[b]
        if torch.is_tensor(v):
            v = v.reshape(-1)
            return bool(v[0].item()) if v.numel() > 0 else True
        return bool(v)

    def forward(self, ego_fut_preds, attr_labels, fut_valid_flag, ego_fut_masks,
                agent_boxes=None):
        # agent_boxes is accepted for a common call signature with the SAT loss.
        del agent_boxes
        # ego_fut_preds: (B, fut_ts, 2) per-step displacement (command mode)
        # attr_labels:   (B, A, 34) padded agent GT (layout: dataset.py)
        # ego_fut_masks: (B, fut_ts) valid-timestep mask
        if (ego_fut_preds is None or not torch.is_tensor(ego_fut_preds)
                or ego_fut_preds.numel() == 0):
            return torch.zeros((), device=getattr(ego_fut_preds, 'device', None))
        if attr_labels is None or not torch.is_tensor(attr_labels):
            return ego_fut_preds.new_zeros(())

        device = ego_fut_preds.device
        attr = attr_labels.to(device).float()
        if attr.dim() == 2:
            attr = attr[None]
        B = ego_fut_preds.shape[0]
        T = min(self.fut_ts, ego_fut_preds.shape[1])
        t2, t3 = self.fut_ts * 2, self.fut_ts * 3   # 12, 18

        total = ego_fut_preds.new_zeros(())
        count = 0
        for b in range(min(B, attr.shape[0])):
            if not self._sample_valid(fut_valid_flag, b):
                continue
            attr_b = attr[b]                                    # (A, 34)
            if attr_b.shape[-1] < t3 + 10:
                continue
            fut_trajs = attr_b[:, :t2].reshape(-1, self.fut_ts, 2)[:, :T]  # (A,T,2)
            fut_mask = attr_b[:, t2:t3][:, :T]                  # (A, T)
            lcf = attr_b[:, t3 + 1:t3 + 10]                     # (A, 9)
            agent_xy = lcf[:, 0:2]                              # (A, 2)
            agent_w = lcf[:, 5].clamp(min=0.0)                  # (A,)
            agent_l = lcf[:, 6].clamp(min=0.0)                  # (A,)

            if fut_mask.sum() == 0:
                continue

            # absolute future centres (same lidar frame as the ego trajectory)
            agent_fut = agent_xy[:, None, :] + fut_trajs.cumsum(dim=1)   # (A,T,2)
            ego_fut = ego_fut_preds[b, :T].cumsum(dim=0)                 # (T, 2)

            dist = torch.linalg.norm(
                ego_fut[None, :, :] - agent_fut, dim=-1)                 # (A, T)
            agent_radius = 0.25 * (agent_l + agent_w)                     # (A,)
            thresh = agent_radius[:, None] + self.ego_radius + self.safe_margin

            penalty = torch.relu(thresh - dist)                          # (A, T)
            mask = fut_mask
            if ego_fut_masks is not None and torch.is_tensor(ego_fut_masks):
                em = ego_fut_masks.to(device).float()
                em = em[b] if em.dim() >= 2 else em
                mask = mask * em[:T][None, :]
            penalty = penalty * mask
            denom = mask.sum().clamp(min=1.0)
            total = total + penalty.sum() / denom
            count += 1

        if count > 0:
            total = total / count
        return self.loss_weight * total


@OPENOCC_LOSS.register_module()
class PlanAgentSATCollisionLoss(nn.Module):
    """Differentiable collision-avoidance loss using Separating Axis Theorem (SAT).

    Unlike the circle-based :class:`PlanAgentCollisionLoss`, this treats ego and
    each agent as **oriented boxes** (cx, cy, w, l, yaw) and computes the overlap
    depth via SAT axis projections. This matches the ``plan_obj_box_col`` metric
    semantics (oriented/axis-aligned rectangles rasterised on the occupancy grid)
    far more closely than the isotropic-circle approximation, and produces a
    gradient that pushes the ego box out of the agent box along the actual
    penetration direction.

    Gradient policy:
        - ego box centre is differentiable (flows back to ``ego_fut_preds``). Its
          yaw is taken from the GT ego yaw (or trajectory tangent) and **fixed**
          (detached) so the loss only pushes the ego *position* out, which keeps
          training stable and matches the metric where ego is an axis-aligned box.
        - agent boxes are GT targets (fixed), including their future yaw.

    Args:
        loss_weight (float): weight of the collision loss.
        safe_margin (float): extra clearance (m) added on top of the SAT overlap.
        fut_ts (int): number of future timesteps (default 6).
        ego_width / ego_length (float): ego footprint dims.
    """

    def __init__(self, loss_weight=1.0, safe_margin=0.5, fut_ts=6,
                 ego_width=1.85, ego_length=4.084):
        super().__init__()
        self.loss_weight = loss_weight
        self.safe_margin = safe_margin
        self.fut_ts = fut_ts
        self.ego_hw = 0.5 * ego_width
        self.ego_hl = 0.5 * ego_length

    @staticmethod
    def _sample_valid(fut_valid_flag, b):
        if fut_valid_flag is None:
            return True
        v = fut_valid_flag
        if isinstance(v, (list, tuple)):
            v = v[b] if b < len(v) else v[0]
        elif torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] > b:
            v = v[b]
        if torch.is_tensor(v):
            v = v.reshape(-1)
            return bool(v[0].item()) if v.numel() > 0 else True
        return bool(v)

    def forward(self, ego_fut_preds, attr_labels, fut_valid_flag, ego_fut_masks,
                agent_boxes=None):
        # ego_fut_preds: (B, fut_ts, 2) per-step displacement (command mode)
        # attr_labels:   (B, A, 34) padded agent GT (layout below)
        # agent_boxes:   (B, A, >=7) current GT boxes in dataset box convention
        #                 (x, y, z, width, length, height, yaw, ...)
        # ego_fut_masks: (B, fut_ts) valid-timestep mask
        if (ego_fut_preds is None or not torch.is_tensor(ego_fut_preds)
                or ego_fut_preds.numel() == 0):
            return torch.zeros((), device=getattr(ego_fut_preds, 'device', None))
        if attr_labels is None or not torch.is_tensor(attr_labels):
            return ego_fut_preds.new_zeros(())

        device = ego_fut_preds.device
        attr = attr_labels.to(device).float()
        if attr.dim() == 2:
            attr = attr[None]
        B = ego_fut_preds.shape[0]
        T = min(self.fut_ts, ego_fut_preds.shape[1])
        t2, t3 = self.fut_ts * 2, self.fut_ts * 3   # 12, 18

        # Ego is treated as an axis-aligned box (yaw fixed to 0), which matches
        # the ``plan_obj_box_col`` metric and keeps the gradient pushing only the
        # ego *position* out (stable, differentiable w.r.t. ego_fut_preds).
        total = ego_fut_preds.new_zeros(())
        count = 0
        for b in range(min(B, attr.shape[0])):
            if not self._sample_valid(fut_valid_flag, b):
                continue
            attr_b = attr[b]                                    # (A, 34)
            if attr_b.shape[-1] < t3 + 10:
                continue
            fut_trajs = attr_b[:, :t2].reshape(-1, self.fut_ts, 2)[:, :T]  # (A,T,2)
            fut_mask = attr_b[:, t2:t3][:, :T]                  # (A, T)
            lcf = attr_b[:, t3 + 1:t3 + 10]                     # (A, 9)
            fut_yaw_delta = attr_b[:, t3 + 10:t3 + 10 + T]     # (A, T)

            # Match PlanningMetric.get_birds_eye_view_label exactly.  ``gt_boxes``
            # uses (x, y, z, width, length, height, yaw, ...); its yaw is converted
            # to the LiDAR convention used by the rasterised metric.  The future
            # yaw label stores per-step deltas, so it must be accumulated.
            boxes_b = None
            if torch.is_tensor(agent_boxes):
                boxes_b = agent_boxes.to(device).float()
                if boxes_b.dim() >= 3:
                    boxes_b = boxes_b[b]
                if boxes_b.dim() != 2 or boxes_b.shape[-1] < 7:
                    boxes_b = None
            if boxes_b is not None:
                agent_count = min(attr_b.shape[0], boxes_b.shape[0])
                attr_b = attr_b[:agent_count]
                fut_trajs = fut_trajs[:agent_count]
                fut_mask = fut_mask[:agent_count]
                lcf = lcf[:agent_count]
                fut_yaw_delta = fut_yaw_delta[:agent_count]
                boxes_b = boxes_b[:agent_count]
                agent_xy = boxes_b[:, 0:2]
                agent_w = boxes_b[:, 3].clamp(min=0.0)
                agent_l = boxes_b[:, 4].clamp(min=0.0)
                agent_yaw = -(boxes_b[:, 6] + math.pi / 2)
            else:
                # Fallback for callers without gt_boxes. lcf uses the same
                # (width, length) layout, but cannot guarantee metric-perfect yaw.
                agent_xy = lcf[:, 0:2]
                agent_w = lcf[:, 5].clamp(min=0.0)
                agent_l = lcf[:, 6].clamp(min=0.0)
                agent_yaw = -(lcf[:, 2] + math.pi / 2)

            if fut_mask.sum() == 0:
                continue

            agent_fut = agent_xy[:, None, :] + fut_trajs.cumsum(dim=1)   # (A,T,2)
            # PlanningMetric evaluates a fixed, axis-aligned ego rectangle with
            # centre (traj_x + 0.5, traj_y), length on x and width on y.
            ego_fut = ego_fut_preds[b, :T].cumsum(dim=0)
            ego_fut = ego_fut + ego_fut.new_tensor([0.5, 0.0])

            # ---- SAT overlap depth between ego box and each agent box ----
            # Both conventions below match PlanningMetric: x is vehicle length,
            # y is vehicle width; agent yaw is an absolute LiDAR-frame yaw.
            ego_hx = self.ego_hl
            ego_hy = self.ego_hw
            ahx = 0.5 * agent_l[:, None, None]                # (A,1,1)
            ahy = 0.5 * agent_w[:, None, None]                # (A,1,1)
            ay = agent_yaw[:, None] + fut_yaw_delta.cumsum(dim=1)  # (A,T)

            # rotation matrices for agent boxes
            cos_a = torch.cos(ay)                              # (A,T)
            sin_a = torch.sin(ay)

            # delta between centres: ego_fut (T,2) - agent_fut (A,T,2)
            # d = (dx, dy) in lidar frame
            dx = ego_fut[:, 0][None, :, None] - agent_fut[..., 0][..., None]  # (A,T,1)
            dy = ego_fut[:, 1][None, :, None] - agent_fut[..., 1][..., None]

            # Rotate the delta into each agent's local frame:
            #   d_local = R^T(-yaw) * d  (world->agent)
            dlx = cos_a[..., None] * dx + sin_a[..., None] * dy  # (A,T,1)
            dly = -sin_a[..., None] * dx + cos_a[..., None] * dy

            # SAT separation check on the two axes of the agent box (in agent frame,
            # ego is a box with half-extents that must be rotated into agent frame).
            # Ego half-extents in agent frame: rotate (ego_hx, ego_hy) by (yaw_agent - yaw_ego)
            # Since yaw_ego=0, rotation is by -yaw_agent.
            # ego corners half-extent projection onto agent axes:
            ca = cos_a[..., None]   # (A,T,1)
            sa = sin_a[..., None]
            # half-extent of ego box projected on agent x-axis (world->agent rotate by -yaw)
            proj_ego_x = ego_hx * torch.abs(ca) + ego_hy * torch.abs(sa)
            proj_ego_y = ego_hx * torch.abs(sa) + ego_hy * torch.abs(ca)

            # separation on each agent axis
            sep_x = torch.abs(dlx) - (ahx + proj_ego_x)
            sep_y = torch.abs(dly) - (ahy + proj_ego_y)

            # Now check the two axes of the ego box (axis-aligned):
            # project agent's extent onto world x/y axes and compare with ego half-extent.
            # agent half-extent projected onto world x = ahx*|cos| + ahy*|sin|
            proj_agent_x = ahx * torch.abs(ca) + ahy * torch.abs(sa)
            proj_agent_y = ahx * torch.abs(sa) + ahy * torch.abs(ca)
            sep_wx = torch.abs(dx) - (ego_hx + proj_agent_x)
            sep_wy = torch.abs(dy) - (ego_hy + proj_agent_y)

            # Boxes overlap only when every SAT axis overlaps. Therefore the
            # largest separation is the active minimum-translation axis; its
            # negative is the penetration depth (positive iff all axes overlap).
            seps = torch.stack([sep_x.squeeze(-1), sep_y.squeeze(-1),
                                sep_wx.squeeze(-1), sep_wy.squeeze(-1)], dim=-1)  # (A,T,4)
            max_sep = seps.max(dim=-1).values                          # (A,T)
            overlap = -max_sep                                          # >0 when colliding

            penalty = torch.relu(overlap + self.safe_margin)           # (A,T)

            mask = fut_mask
            if ego_fut_masks is not None and torch.is_tensor(ego_fut_masks):
                em = ego_fut_masks.to(device).float()
                em = em[b] if em.dim() >= 2 else em
                mask = mask * em[:T][None, :]
            penalty = penalty * mask
            denom = mask.sum().clamp(min=1.0)
            total = total + penalty.sum() / denom
            count += 1

        if count > 0:
            total = total / count
        return self.loss_weight * total


class PlanMapBoundLoss(nn.Module):
    """Planning constraint to push ego vehicle away from the lane boundary.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
        map_thresh (float, optional): confidence threshold to filter map predictions.
        lane_bound_cls_idx (float, optional): lane_boundary class index.
        dis_thresh (float, optional): distance threshold between ego vehicle and lane bound.
        point_cloud_range (list, optional): point cloud range.
    """

    def __init__(
        self,
        reduction='mean',
        loss_weight=1.0,
        map_thresh=0.5,
        lane_bound_cls_idx=2,
        dis_thresh=1.0,
        point_cloud_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
        perception_detach=False
    ):
        super(PlanMapBoundLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.map_thresh = map_thresh
        self.lane_bound_cls_idx = lane_bound_cls_idx
        self.dis_thresh = dis_thresh
        self.pc_range = point_cloud_range
        self.perception_detach = perception_detach

    def forward(self,
                ego_fut_preds,
                lane_preds,
                lane_score_preds,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            ego_fut_preds (Tensor): [B, fut_ts, 2]
            lane_preds (Tensor): [B, num_vec, num_pts, 2]
            lane_score_preds (Tensor): [B, num_vec, 3]
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        if self.perception_detach:
            lane_preds = lane_preds.detach()
            lane_score_preds = lane_score_preds.detach()

        # filter lane element according to confidence score and class
        not_lane_bound_mask = lane_score_preds[..., self.lane_bound_cls_idx] < self.map_thresh
        # denormalize map pts
        lane_bound_preds = lane_preds.clone()
        lane_bound_preds[...,0:1] = (lane_bound_preds[..., 0:1] * (self.pc_range[3] -
                                self.pc_range[0]) + self.pc_range[0])
        lane_bound_preds[...,1:2] = (lane_bound_preds[..., 1:2] * (self.pc_range[4] -
                                self.pc_range[1]) + self.pc_range[1])
        # pad not-lane-boundary cls and low confidence preds
        lane_bound_preds[not_lane_bound_mask] = 1e6

        loss_bbox = self.loss_weight * plan_map_bound_loss(ego_fut_preds, lane_bound_preds,
                                                           weight=weight, dis_thresh=self.dis_thresh,
                                                           reduction=reduction, avg_factor=avg_factor)
        return loss_bbox


@mmcv.utils.parrots_jit.jit(derivate=True, coderize=True)
@weighted_loss
def plan_map_bound_loss(pred, target, dis_thresh=1.0):
    """Planning map bound constraint (L1 distance).

    Args:
        pred (torch.Tensor): ego_fut_preds, [B, fut_ts, 2].
        target (torch.Tensor): lane_bound_preds, [B, num_vec, num_pts, 2].
        weight (torch.Tensor): [B, fut_ts]

    Returns:
        torch.Tensor: Calculated loss [B, fut_ts]
    """
    pred = pred.cumsum(dim=-2)
    ego_traj_starts = pred[:, :-1, :]
    ego_traj_ends = pred
    B, T, _ = ego_traj_ends.size()
    padding_zeros = torch.zeros((B, 1, 2), dtype=pred.dtype, device=pred.device)  # initial position
    ego_traj_starts = torch.cat((padding_zeros, ego_traj_starts), dim=1)
    _, V, P, _ = target.size()
    ego_traj_expanded = ego_traj_ends.unsqueeze(2).unsqueeze(3)  # [B, T, 1, 1, 2]
    maps_expanded = target.unsqueeze(1)  # [1, 1, M, P, 2]
    dist = torch.linalg.norm(ego_traj_expanded - maps_expanded, dim=-1)  # [B, T, M, P]
    dist = dist.min(dim=-1, keepdim=False)[0]
    min_inst_idxs = torch.argmin(dist, dim=-1).tolist()
    batch_idxs = [[i] for i in range(dist.shape[0])]
    ts_idxs = [[i for i in range(dist.shape[1])] for j in range(dist.shape[0])]
    bd_target = target.unsqueeze(1).repeat(1, pred.shape[1], 1, 1, 1)
    min_bd_insts = bd_target[batch_idxs, ts_idxs, min_inst_idxs]  # [B, T, P, 2]
    bd_inst_starts = min_bd_insts[:, :, :-1, :].flatten(0, 2)
    bd_inst_ends = min_bd_insts[:, :, 1:, :].flatten(0, 2)
    ego_traj_starts = ego_traj_starts.unsqueeze(2).repeat(1, 1, P-1, 1).flatten(0, 2)
    ego_traj_ends = ego_traj_ends.unsqueeze(2).repeat(1, 1, P-1, 1).flatten(0, 2)

    intersect_mask = segments_intersect(ego_traj_starts, ego_traj_ends,
                                        bd_inst_starts, bd_inst_ends)
    intersect_mask = intersect_mask.reshape(B, T, P-1)
    intersect_mask = intersect_mask.any(dim=-1)
    intersect_idx = (intersect_mask == True).nonzero()

    target = target.view(target.shape[0], -1, target.shape[-1])
    # [B, fut_ts, num_vec*num_pts]
    dist = torch.linalg.norm(pred[:, :, None, :] - target[:, None, :, :], dim=-1)
    min_idxs = torch.argmin(dist, dim=-1).tolist()
    batch_idxs = [[i] for i in range(dist.shape[0])]
    ts_idxs = [[i for i in range(dist.shape[1])] for j in range(dist.shape[0])]
    min_dist = dist[batch_idxs, ts_idxs, min_idxs]
    loss = min_dist
    safe_idx = loss > dis_thresh
    unsafe_idx = loss <= dis_thresh
    loss[safe_idx] = 0
    loss[unsafe_idx] = dis_thresh - loss[unsafe_idx]

    for i in range(len(intersect_idx)):
        loss[intersect_idx[i, 0], intersect_idx[i, 1]:] = 0

    return loss


def segments_intersect(line1_start, line1_end, line2_start, line2_end):
    # Calculating the differences
    dx1 = line1_end[:, 0] - line1_start[:, 0]
    dy1 = line1_end[:, 1] - line1_start[:, 1]
    dx2 = line2_end[:, 0] - line2_start[:, 0]
    dy2 = line2_end[:, 1] - line2_start[:, 1]

    # Calculating determinants
    det = dx1 * dy2 - dx2 * dy1
    det_mask = det != 0

    # Checking if lines are parallel or coincident
    parallel_mask = torch.logical_not(det_mask)

    # Calculating intersection parameters
    t1 = ((line2_start[:, 0] - line1_start[:, 0]) * dy2
          - (line2_start[:, 1] - line1_start[:, 1]) * dx2) / det
    t2 = ((line2_start[:, 0] - line1_start[:, 0]) * dy1
          - (line2_start[:, 1] - line1_start[:, 1]) * dx1) / det

    # Checking intersection conditions
    intersect_mask = torch.logical_and(
        torch.logical_and(t1 >= 0, t1 <= 1),
        torch.logical_and(t2 >= 0, t2 <= 1)
    )

    # Handling parallel or coincident lines
    intersect_mask[parallel_mask] = False

    return intersect_mask


class PlanMapDirectionLoss(nn.Module):
    """Planning loss to force the ego heading angle consistent with lane direction.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
        theta_thresh (float, optional): angle diff thresh between ego and lane.
        point_cloud_range (list, optional): point cloud range.
    """

    def __init__(
        self,
        reduction='mean',
        loss_weight=1.0,
        map_thresh=0.5,
        dis_thresh=2.0,
        lane_div_cls_idx=0,
        point_cloud_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
    ):
        super(PlanMapDirectionLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.map_thresh = map_thresh
        self.dis_thresh = dis_thresh
        self.lane_div_cls_idx = lane_div_cls_idx
        self.pc_range = point_cloud_range

    def forward(self,
                ego_fut_preds,
                lane_preds,
                lane_score_preds,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            ego_fut_preds (Tensor): [B, fut_ts, 2]
            lane_preds (Tensor): [B, num_vec, num_pts, 2]
            lane_score_preds (Tensor): [B, num_vec, 3]
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        # filter lane element according to confidence score and class
        not_lane_div_mask = lane_score_preds[..., self.lane_div_cls_idx] < self.map_thresh
        # denormalize map pts
        lane_div_preds = lane_preds.clone()
        lane_div_preds[...,0:1] = (lane_div_preds[..., 0:1] * (self.pc_range[3] -
                                self.pc_range[0]) + self.pc_range[0])
        lane_div_preds[...,1:2] = (lane_div_preds[..., 1:2] * (self.pc_range[4] -
                                self.pc_range[1]) + self.pc_range[1])
        # pad not-lane-divider cls and low confidence preds
        lane_div_preds[not_lane_div_mask] = 1e6

        loss_bbox = self.loss_weight * plan_map_dir_loss(ego_fut_preds, lane_div_preds,
                                                           weight=weight, dis_thresh=self.dis_thresh,
                                                           reduction=reduction, avg_factor=avg_factor)
        return loss_bbox


@mmcv.utils.parrots_jit.jit(derivate=True, coderize=True)
@weighted_loss
def plan_map_dir_loss(pred, target, dis_thresh=2.0):
    """Planning ego-map directional loss.

    Args:
        pred (torch.Tensor): ego_fut_preds, [B, fut_ts, 2].
        target (torch.Tensor): lane_div_preds, [B, num_vec, num_pts, 2].
        weight (torch.Tensor): [B, fut_ts]

    Returns:
        torch.Tensor: Calculated loss [B, fut_ts]
    """
    num_map_pts = target.shape[2]
    pred = pred.cumsum(dim=-2)
    traj_dis = torch.linalg.norm(pred[:, -1, :] - pred[:, 0, :], dim=-1)
    static_mask = traj_dis < 1.0
    target = target.unsqueeze(1).repeat(1, pred.shape[1], 1, 1, 1)

    # find the closest map instance for ego at each timestamp
    dist = torch.linalg.norm(pred[:, :, None, None, :] - target, dim=-1)
    dist = dist.min(dim=-1, keepdim=False)[0]
    min_inst_idxs = torch.argmin(dist, dim=-1).tolist()
    batch_idxs = [[i] for i in range(dist.shape[0])]
    ts_idxs = [[i for i in range(dist.shape[1])] for j in range(dist.shape[0])]
    target_map_inst = target[batch_idxs, ts_idxs, min_inst_idxs]  # [B, fut_ts, num_pts, 2]

    # calculate distance
    dist = torch.linalg.norm(pred[:, :, None, :] - target_map_inst, dim=-1)
    min_pts_idxs = torch.argmin(dist, dim=-1)
    min_pts_next_idxs = min_pts_idxs.clone()
    is_end_point = (min_pts_next_idxs == num_map_pts-1)
    not_end_point = (min_pts_next_idxs != num_map_pts-1)
    min_pts_next_idxs[is_end_point] = num_map_pts - 2
    min_pts_next_idxs[not_end_point] = min_pts_next_idxs[not_end_point] + 1
    min_pts_idxs = min_pts_idxs.tolist()
    min_pts_next_idxs = min_pts_next_idxs.tolist()
    traj_yaw = torch.atan2(torch.diff(pred[..., 1]), torch.diff(pred[..., 0]))  # [B, fut_ts-1]
    # last ts yaw assume same as previous
    traj_yaw = torch.cat([traj_yaw, traj_yaw[:, [-1]]], dim=-1)  # [B, fut_ts]
    min_pts = target_map_inst[batch_idxs, ts_idxs, min_pts_idxs]
    dist = torch.linalg.norm(min_pts - pred, dim=-1)
    dist_mask = dist > dis_thresh
    min_pts = min_pts.unsqueeze(2)
    min_pts_next = target_map_inst[batch_idxs, ts_idxs, min_pts_next_idxs].unsqueeze(2)
    map_pts = torch.cat([min_pts, min_pts_next], dim=2)
    lane_yaw = torch.atan2(torch.diff(map_pts[..., 1]).squeeze(-1), torch.diff(map_pts[..., 0]).squeeze(-1))  # [B, fut_ts]
    yaw_diff = traj_yaw - lane_yaw
    yaw_diff[yaw_diff > math.pi] =  yaw_diff[yaw_diff > math.pi] - math.pi
    yaw_diff[yaw_diff > math.pi/2] = yaw_diff[yaw_diff > math.pi/2] - math.pi
    yaw_diff[yaw_diff < -math.pi] = yaw_diff[yaw_diff < -math.pi] + math.pi
    yaw_diff[yaw_diff < -math.pi/2] = yaw_diff[yaw_diff < -math.pi/2] + math.pi
    yaw_diff[dist_mask] = 0  # loss = 0 if no lane around ego
    yaw_diff[static_mask] = 0  # loss = 0 if ego is static

    loss = torch.abs(yaw_diff)

    return loss  # [B, fut_ts]
