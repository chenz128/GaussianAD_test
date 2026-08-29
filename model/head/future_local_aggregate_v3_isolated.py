"""Local aggregation adapter for V3 future Gaussians.

The shared ``LocalAggregator`` requires every Gaussian centre to lie inside
the OCC grid.  V3-SE3 deliberately keeps a Gaussian when its centre is just
outside the grid but its finite footprint still overlaps the render volume.
The CUDA implementation safely clips that footprint to the grid in
``getRect``; the original V3 implementation therefore did not apply the
centre-only assertion.

This adapter is used only by the isolated V3 future-flow path.  It reuses the
same autograd kernel and the existing aggregator configuration, so current
frame OCC/Gaussian prediction and model state_dict structure remain unchanged.
"""

import torch

from .localagg.local_aggregate import _LocalAggregate


def aggregate_v3_future(aggregator, pts, means3d, opacities, semantics,
                        scales, cov3d):
    """Aggregate footprint-overlapping V3 future Gaussians."""
    assert pts.shape[0] == 1
    pts = pts.squeeze(0)
    assert not pts.requires_grad
    means3d = means3d.squeeze(0)
    opacities = opacities.squeeze(0)
    semantics = semantics.squeeze(0)
    scales = scales.detach().squeeze(0)
    cov3d = cov3d.squeeze(0)

    points_int = ((pts - aggregator.pc_min) /
                  aggregator.grid_size).to(torch.int)
    assert (points_int.min() >= 0
            and points_int[:, 0].max() < aggregator.H
            and points_int[:, 1].max() < aggregator.W
            and points_int[:, 2].max() < aggregator.D)

    # Do not require the Gaussian centre itself to be in-grid.  The V3-SE3
    # range mask has already verified footprint overlap, and the CUDA getRect
    # implementation clamps the affected voxel rectangle to the grid.
    means3d_int = ((means3d.detach() - aggregator.pc_min) /
                   aggregator.grid_size).to(torch.int)
    radii = torch.ceil(
        scales.max(dim=-1)[0] * aggregator.scale_multiplier /
        aggregator.grid_size).to(torch.int)
    assert radii.min() >= 1
    cov3d = cov3d.flatten(1)[:, [0, 4, 8, 1, 5, 2]]

    logits = _LocalAggregate.apply(
        pts,
        points_int,
        means3d,
        means3d_int,
        opacities,
        semantics,
        radii,
        cov3d,
        aggregator.H,
        aggregator.W,
        aggregator.D,
    )
    if aggregator.inv_softmax:
        raise AssertionError('inverse softmax aggregation is unsupported')
    return logits
