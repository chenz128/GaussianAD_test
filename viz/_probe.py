import sys, torch
sys.path.insert(0,'.')
from mmengine import Config
cfg = Config.fromfile('config/nuscenes_gs25600_base_plan/nuscenes_gs25600_base_plan.py')
from dataset import get_dataloader
_, vl = get_dataloader(cfg.train_dataset_config, cfg.val_dataset_config, cfg.train_loader, cfg.val_loader, dist=False, val_only=True)
import model
from mmseg.models import build_segmentor
m = build_segmentor(cfg.model)
ck=torch.load('exp/nuscenes_gs25600_base_plan/checkpoints/epoch_15.pth',map_location='cpu')
sd=ck['state_dict'] if 'state_dict' in ck else ck
m.load_state_dict(sd, strict=False); m.cuda().eval()
data=next(iter(vl))
for k in list(data.keys()):
    if isinstance(data[k],torch.Tensor): data[k]=data[k].cuda()
with torch.no_grad():
    res=m(imgs=data['img'], metas=data)
sz=res['sampled_xyz'][0]
print('RES', flush=True)
print('sampled_xyz', tuple(sz.shape), flush=True)
print('x', float(sz[:,0].min()), float(sz[:,0].max()), flush=True)
print('y', float(sz[:,1].min()), float(sz[:,1].max()), flush=True)
print('z unique', torch.unique(sz[:,2]).cpu().numpy(), flush=True)
print('pred_occ[-1][0]', tuple(res['pred_occ'][-1][0].shape), flush=True)
print('ego_fut_preds', tuple(res['ego_fut_preds'].shape), flush=True)
