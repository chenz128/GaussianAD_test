"""Official GaussianAD v1 planning metric wrapper without trajectory cumsum."""

import torch

from dataset.metric_stp3 import PlanningMetric


def compute_planner_metric_v1(
    pred_ego_fut_trajs,
    gt_ego_fut_trajs,
    gt_agent_boxes,
    gt_agent_feats,
    fut_valid_flag,
):
    metric = {'fut_valid_flag': bool(fut_valid_flag)}
    for second in range(1, 4):
        for name in (
            'plan_L2', 'plan_L2_stp3', 'plan_obj_col',
            'plan_obj_col_stp3', 'plan_obj_box_col',
            'plan_obj_box_col_stp3',
        ):
            metric[f'{name}_{second}s'] = 0.0

    assert pred_ego_fut_trajs.shape[0] == 1, 'v1 metric requires batch 1'
    planning_metric = PlanningMetric()
    segmentation, pedestrian = planning_metric.get_label(
        gt_agent_boxes, gt_agent_feats)
    occupancy = torch.logical_or(segmentation, pedestrian)

    if not bool(fut_valid_flag):
        return metric
    for index in range(3):
        steps = (index + 1) * 2
        prediction = pred_ego_fut_trajs[0, :steps].detach().to(
            gt_ego_fut_trajs.device)
        target = gt_ego_fut_trajs[0, :steps]
        l2 = planning_metric.compute_L2(prediction, target)
        l2_stp3 = planning_metric.compute_L2_stp3(prediction, target)
        obj_col, box_col = planning_metric.evaluate_coll(
            pred_ego_fut_trajs[:, :steps].detach().to(
                gt_ego_fut_trajs.device),
            gt_ego_fut_trajs[:, :steps],
            occupancy.to(gt_ego_fut_trajs.device),
        )
        second = index + 1
        metric[f'plan_L2_{second}s'] = l2
        metric[f'plan_L2_stp3_{second}s'] = l2_stp3
        metric[f'plan_obj_col_{second}s'] = obj_col.mean().item()
        metric[f'plan_obj_col_stp3_{second}s'] = obj_col[-1].item()
        metric[f'plan_obj_box_col_{second}s'] = box_col.mean().item()
        metric[f'plan_obj_box_col_stp3_{second}s'] = box_col[-1].item()
    return metric
