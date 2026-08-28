"""V3-only data transform for future LiDAR poses.

This module is imported only by the isolated V3 configs.  It intentionally
avoids changing the shared NuScenesDataset implementation.
"""

import numpy as np

from . import OPENOCC_TRANSFORMS


@OPENOCC_TRANSFORMS.register_module()
class BuildFutureLidarPoseV3Isolated:
    """Collect six future lidar-to-global transforms before flow processing."""

    def __call__(self, results):
        current = np.asarray(results['lidar2global'])
        flow_info = results['flow_info']
        if len(flow_info) != 6:
            raise ValueError(
                f'V3 requires six future flow frames, got {len(flow_info)}')
        results['future_lidar2global'] = np.stack([
            np.asarray(frame['lidar2global'])
            if frame['flow_valid_flag'] else current
            for frame in flow_info
        ]).astype(np.float32)
        return results

