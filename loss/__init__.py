from mmengine.registry import Registry
OPENOCC_LOSS = Registry('openocc_loss')
MAPTR_LOSS = Registry('maptr_loss')
MAPTR_COST = Registry('maptr_cost')

from .multi_loss import MultiLoss
from .occupancy_loss import OccupancyLoss
from .detection_loss import DetectionLoss
from .map_loss import PtsL1Loss, OrderedPtsL1Cost
from .plan_loss import PlanLoss
from .occupancy_loss_flow import OccupancyFlowLoss
from .render_loss import RenderLoss
from .dynamic_loss import DynamicLoss
from .physics_loss import PhysicsLoss

from .time_query_plan_loss import TimeQueryPlanLoss
from .residual_diffusion_plan_loss import ResidualDiffusionPlanLoss
