"""定向搜索：近距(<8m) + 大面积(box投影>25%画面) + 真动(spd>1m/s)的可动 box，
用于验证 LiDAR 仲裁能否救回"路过的动态大车"。打印 token/cam/spd/area_frac。
"""
import os
import numpy as np
import mmengine
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points, BoxVisibility

NUSC_ROOT = 'data/nuscenes'
PKL = 'data/nuscenes_cam/nuscenes_infos_train_gaussian_ad_v4.pkl'
SENSOR_TYPES = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
SCALE, CROP_TOP, RH, RW = 0.44, 140, 256, 704
MOVABLE_PREFIX = ('vehicle', 'human')


def main():
    nusc = NuScenes(version='v1.0-trainval', dataroot=NUSC_ROOT, verbose=False)
    data = mmengine.load(PKL)
    si, kf = data['infos'], data['metadata']
    img_area = float(RH * RW)
    s0 = int(os.environ.get('S0', '0'))
    s1 = int(os.environ.get('S1', str(len(kf))))
    ss = int(os.environ.get('SS', '5'))
    found = 0
    for gi in range(s0, min(s1, len(kf)), ss):
        st, fr = kf[gi]
        scene = si[st]
        fr = int(np.clip(fr, 0, len(scene) - 1))
        token = scene[fr]['token']
        sample = nusc.get('sample', token)
        for cam in SENSOR_TYPES:
            sd_token = sample['data'][cam]
            _, boxes_cam, camK = nusc.get_sample_data(sd_token, box_vis_level=BoxVisibility.ANY)
            for box in boxes_cam:
                if not box.name.startswith(MOVABLE_PREFIX):
                    continue
                vel = nusc.box_velocity(box.token)
                if np.isnan(vel).any():
                    continue
                spd = float(np.linalg.norm(vel))
                if spd <= 1.0:
                    continue
                # 深度 = box 中心相机系 z
                ctr = box.center
                if ctr[2] <= 0 or ctr[2] > 8.0:
                    continue
                cor = view_points(box.corners(), camK, normalize=True)[:2]
                xs = cor[0] * SCALE
                ys = cor[1] * SCALE - CROP_TOP
                x0, x1 = max(0, xs.min()), min(RW, xs.max())
                y0, y1 = max(0, ys.min()), min(RH, ys.max())
                if x1 <= x0 or y1 <= y0:
                    continue
                area_frac = (x1 - x0) * (y1 - y0) / img_area
                if area_frac < 0.25:
                    continue
                print(f'gi={gi} token={token[:10]} cam={cam} spd={spd:.1f} '
                      f'depth={ctr[2]:.1f} area_frac={area_frac:.2f} name={box.name}',
                      flush=True)
                found += 1
        if found >= 15:
            break
    print(f'TOTAL {found}', flush=True)


if __name__ == '__main__':
    main()
