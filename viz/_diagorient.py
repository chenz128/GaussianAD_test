import os, sys, numpy as np, torch
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,ROOT); os.chdir(ROOT)
from mmengine import Config
import model
from dataset import get_dataloader
cfg=Config.fromfile('exp/nuscenes_gs25600_v12_fixempty/nuscenes_gs25600_gtbox_oracle_v12.py')
_,vl=get_dataloader(cfg.train_dataset_config,cfg.val_dataset_config,cfg.train_loader,cfg.val_loader,dist=False,val_only=True)
np.set_printoptions(precision=2,suppress=True,linewidth=200)
it=iter(vl)
for _ in range(6): d=next(it)   # idx5
ol=np.asarray(d['occ_label'][0].cpu() if torch.is_tensor(d['occ_label']) else d['occ_label'][0])
if ol.ndim==4: ol=ol[0]
gb=np.asarray(d['gt_boxes'][0].cpu() if torch.is_tensor(d['gt_boxes']) else d['gt_boxes'][0])
ft=np.asarray(d['ego_fut_trajs'][0].cpu() if torch.is_tensor(d['ego_fut_trajs']) else d['ego_fut_trajs'][0])
print('ego_fut cum end (col0,col1):', np.cumsum(ft,0)[-1], flush=True)
print('occ shape', ol.shape, flush=True)
# class ids in occ
ids,cnts=np.unique(ol,return_counts=True)
print('occ classes:', dict(zip(ids.tolist(),cnts.tolist())), flush=True)
# truck=10, car=4, driveable=11. locate truck voxels centroid in (dim0,dim1)
for cls,name in [(10,'truck'),(4,'car'),(11,'driveable')]:
    m=(ol==cls).any(2)
    if m.sum():
        i,j=np.where(m)
        print(f'{name}({cls}): n={m.sum()} dim0(idx)~{i.mean():.0f}->x={i.mean()*0.5-30:+.1f}  dim1(idx)~{j.mean():.0f}->y={j.mean()*0.5-30:+.1f}', flush=True)
# gt boxes: show class col9, x,y, and which are behind
print('--- gt boxes (col9=cls, x=col0, y=col1) ---', flush=True)
for b in gb:
    c9=int(b[9]) if gb.shape[1]>9 else -1
    print(f'  cls={c9} x={b[0]:+.1f} y={b[1]:+.1f} L={b[4]:.1f} W={b[3]:.1f} yaw={b[6]:+.2f}', flush=True)
