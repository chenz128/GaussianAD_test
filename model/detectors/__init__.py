from .detector3d_template import Detector3DTemplate
from .voxelnext import VoxelNeXt
from .maptrv2 import MapTRv2

__all__ = [
    'Detector3DTemplate',
    'VoxelNeXt',
    'MapTRv2'
]


def build_detector(model_cfg, num_class, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )

    return model
