import sys, os
sys.path.insert(0,'.')
from mmengine import Config
cfg = Config.fromfile('config/nuscenes_gs25600_base_plan/nuscenes_gs25600_base_plan.py')
from dataset import get_dataloader
_, vl = get_dataloader(cfg.train_dataset_config, cfg.val_dataset_config, cfg.train_loader, cfg.val_loader, dist=False, val_only=True)
import model
from mmseg.models import build_segmentor
