# NeoAD table vs. GaussianAD: nuScenes planning-metric audit

Date: 2026-09-03

## Executive finding

NeoAD-Tiny/Base numbers in Table 3 are not produced by the standard UniAD,
ST-P3, or VAD planning metric implemented by the other open-source methods.
The public NeoAD evaluator changes both the trajectory error and collision
definitions in ways that systematically reduce the reported values.  The
NeoAD values therefore must not be copied into a direct GaussianAD comparison
table.

The comparison table itself also contains a provenance error: the row labelled
`RAG-Driver [67]` uses numbers from RDA-Driver, while the cited RAG-Driver
repository evaluates BDD-X explanations rather than nuScenes planning.  The
first L2 entry is additionally changed from RDA-Driver's published `0.17` to
`0.34`.

Primary sources:

- [NeoAD paper](https://openaccess.thecvf.com/content/CVPR2026W/AUTOPILOT/papers/Wang_Drive_Like_Humans_Plan_Like_Machines_An_Explicit_Sense_and_CVPRW_2026_paper.pdf)
- [NeoAD official code](https://github.com/Summer72Wang/NeoAD)
- [RDA-Driver ECCV paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05259.pdf)
- [RAG-Driver official code](https://github.com/YuanJianhao508/RAG-Driver)

## Reproducible code evidence

NeoAD was audited at commit
`ea27425fd12fe710a2d9d5b86f061b6391bed63b`.

### NeoAD L2 is not 1/2/3-second trajectory L2

In `tools/train_planner.py`, function `temporal_l2_delta_cum_dim`:

1. absolute trajectories are first converted to increments;
2. the first increment of both prediction and GT is set to zero, removing the
   first waypoint error;
3. the Euclidean error is divided by two;
4. the denominator is again multiplied by two;
5. every horizon uses the denominator of all six valid future steps even when
   only the first two or four errors are in its numerator.

The resulting 1-second number can therefore be reduced by much more than a
constant factor and cannot be interpreted as endpoint L2 or horizon ADE.

### NeoAD collision is a weak point-persistence event

In `tools/train_planner.py`, function
`collision_rate_endpoint_centerbox`:

- only the 1/2/3-second endpoints are checked;
- the ego vehicle is represented by its centre point, not its
  1.85 m x 4.084 m footprint;
- a hit is counted only if the endpoint and the preceding step both collide
  (`require_persistence=True`);
- width and length are empirically swapped in the point-in-OBB test;
- the validation call passes `gt_xy=None`, disabling the documented GT
  collision mask.

A trajectory that clips an agent with a front corner for one frame is counted
by GaussianAD/VAD but is invisible to this NeoAD function.  Near-zero results
are therefore expected even without a genuinely collision-free planner.

### Not a distributed-validation shard bug

NeoAD uses a distributed sampler only for training; validation has no sampler
and each process traverses the full set.  Rank zero performs the reported
aggregation.  This is redundant computation, but it is not the source of the
low values.

## Open-source method protocol matrix

| Method/code audited | Trajectory/time aggregation | Collision geometry and time aggregation | Comparable to GaussianAD main keys? |
|---|---|---|---|
| GaussianAD intended / VAD | displacement -> cumulative position; ADE over first 2/4/6 steps | full axis-aligned ego footprint; vehicles + pedestrians; GT-collision mask; mean over first 2/4/6 steps | Yes, when GaussianAD's two `cumsum` lines are active |
| UniAD | absolute positions; values at steps 2/4/6 | full footprint; GT mask; endpoint values at steps 2/4/6 | No; compare only to explicitly reported endpoint keys and align occupancy source |
| Agent-Driver | selectable `uniad` or `stp3`; paper-style output uses UniAD endpoints | full footprint; GT mask; separate UniAD/ST-P3 cached occupancy | No direct comparison to GaussianAD cumulative keys |
| OpenDriveVLA | prints both UniAD endpoint and ST-P3 cumulative forms | full footprint; GT mask; default cache is vehicle-only; ST-P3 requires a non-default flag | Only after selecting ST-P3 and vehicle+pedestrian data |
| OmniDrive | absolute positions; cumulative ADE | 0.1 m raster; vehicle-only; rotated ego footprint with extra sensor offset; **any hit up to horizon** | No |
| dVLM-AD | offers UniAD endpoint and ST-P3 cumulative scripts | corresponding cached occupancy; a second evaluator uses any-hit collision and 0.1 m-style geometry | Only with the explicit ST-P3 path and matched cache |
| OccWorld | displacement -> cumulative position; cumulative plus endpoint outputs | VAD/ST-P3 footprint; vehicles + pedestrians; GT mask | Main cumulative keys are close to GaussianAD |
| World4Drive | copied VAD planning metric and aggregation | VAD/ST-P3 footprint; vehicles + pedestrians; GT mask | Yes, subject to identical validation tokens |
| Drive-OccWorld | cumulative trajectories; repository contains cumulative and single variants | full raster footprint with GT mask; multiple internally inconsistent evaluation paths | Not without identifying the exact paper-producing path |
| DiffusionDrive `nusc` branch | absolute output vs cumulative GT; per-step values converted to cumulative horizon means for its table | full oriented Shapely box against future boxes; GT mask | No; geometry/source differ even though time aggregation is cumulative |
| RAG-Driver | no nuScenes planning evaluator in the public repository | no auditable nuScenes collision implementation | No |
| DriveDreamer | released code is for world/video generation; no matching nuScenes planning evaluator found | not auditable | No |
| OmniScene | official planner repository currently contains only a short README | not auditable | No |

Audited commits:

- VAD `1688c4b1c3a9e2e7873ca9700ff8058170c0e3c8`
- UniAD `609ee083ea51c3521c323f1279dfc4cee0e60467`
- ST-P3 `69aabefd2610951d9e34238142776ed2228673be`
- Agent-Driver `3ea566e99d0b27cb01b47b5d030ba635b7eee06e`
- OpenDriveVLA `10e8095bc618d508cb70cca37b6956ac4db6e9f3`
- OmniDrive `ced207333cb18b69a232cbb9f82bf52089227f12`
- OccWorld `1ee7f77ecc4c984a4f7f6411d95c2e6e73806b6e`
- World4Drive `cffb51adeb1f7d02b49c4b74d7262ded62a33ac8`
- Drive-OccWorld `3bd90b39fc83c944e047b35e26c46b3807b203af`
- DiffusionDrive `nusc` branch `ae54fd87b32b3762f20e63ffd0af91d343cade85`
- dVLM-AD `d3ca475d765148087b441e35776f7b00cfc488f4`
- DriveDreamer `da1ca92f831bc23d91b59ad418eb47b41cbb1fa9`
- OmniScene `da85b0ae524fb86b38114f5b513a66d09e872e1f`

Official repositories:

- [VAD](https://github.com/hustvl/VAD)
- [UniAD](https://github.com/OpenDriveLab/UniAD)
- [DiffusionDrive](https://github.com/hustvl/DiffusionDrive)
- [Agent-Driver](https://github.com/USC-GVL/Agent-Driver)
- [OpenDriveVLA](https://github.com/DriveVLA/OpenDriveVLA)
- [OmniDrive](https://github.com/NVlabs/OmniDrive)
- [OccWorld](https://github.com/wzzheng/OccWorld)
- [Drive-OccWorld](https://github.com/yuyang-cloud/Drive-OccWorld)
- [World4Drive](https://github.com/YvanYin/World4Drive)
- [dVLM-AD](https://github.com/SaFo-Lab/dVLM-AD)

## Rows without an auditable nuScenes evaluator

At the audit date, no runnable official nuScenes planning-metric implementation
was found for PARA-Drive/+, DriveVLM/Dual, VLP, DiMA, EMMA, Drive-R1,
VLM-E2E, FASIONAD++, FastDriveVLA, or the planner represented by the minimal
OmniScene release.  Some have papers/project pages or partial inference code;
that is not sufficient to verify the table's collision protocol.

The dVLM-AD repository is now public, but the NeoAD table does not report its
collision columns.  Its two evaluation entry points also expose different
temporal aggregations, so future comparisons must record the exact command.

## GaussianAD correction and reporting contract

`tools/data/convert_nuscenes_infos_to_gaussianad.py` stores ego trajectories as
per-step displacements.  Before evaluating L2 or collision,
`dataset/metric_stp3.py` must execute:

```python
pred_ego_fut_trajs = pred_ego_fut_trajs.cumsum(dim=-2)
gt_ego_fut_trajs = gt_ego_fut_trajs.cumsum(dim=-2)
```

The server working tree had these lines commented out again on 2026-09-03.
Any log produced in that state is invalid.  The v16 validator refuses to run
when the conversion is disabled.

For every future result table, record all of the following:

1. exact code commit and checkpoint hash/path;
2. exact validation token list and count;
3. trajectory representation (increment or absolute position);
4. temporal aggregation (endpoint, mean, or any-hit);
5. ego geometry and sensor-centre offset;
6. obstacle classes/occupancy cache;
7. whether GT-collision frames are excluded.

Only numbers matching all seven fields should appear in the same comparison
block.  NeoAD-Tiny/Base should be labelled `reported, non-comparable protocol`
until its authors publish a standard evaluator or trajectories that can be
re-evaluated locally.
