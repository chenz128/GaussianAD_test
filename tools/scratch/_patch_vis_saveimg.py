"""Patch remote visualize.py: save a stitched surround-camera image when
--vis-gaussian is on, so we can recognize the scene. Idempotent."""
import re

P = 'visualize.py'
src = open(P).read()

HELPER = '''
def _save_surround(imgs, save_dir, name):
    import os as _os, numpy as _np, cv2 as _cv2
    t = imgs
    while t.dim() > 4:
        t = t[0] if t.shape[0] == 1 else t.reshape(-1, *t.shape[-3:])
    t = t[-6:]
    mean = _np.array([123.675, 116.28, 103.53])
    std = _np.array([58.395, 57.12, 57.375])
    a = t.detach().cpu().numpy().transpose(0, 2, 3, 1) * std + mean
    a = _np.clip(a, 0, 255).astype(_np.uint8)  # RGB, load order:
    # [FRONT, FRONT_RIGHT, FRONT_LEFT, BACK, BACK_LEFT, BACK_RIGHT]
    order = [2, 0, 1, 4, 3, 5]
    labels = ['FRONT_LEFT', 'FRONT', 'FRONT_RIGHT',
              'BACK_LEFT', 'BACK', 'BACK_RIGHT']
    rows = []
    for r in range(2):
        cols = []
        for c in range(3):
            k = order[r * 3 + c]
            im = a[k][:, :, ::-1].copy()  # RGB->BGR for cv2
            _cv2.putText(im, labels[r * 3 + c], (12, 34),
                         _cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)
            cols.append(im)
        rows.append(_np.concatenate(cols, axis=1))
    grid = _np.concatenate(rows, axis=0)
    _cv2.imwrite(_os.path.join(save_dir, name + '.jpg'), grid)

'''

if '_save_surround' not in src:
    # insert helper right after the first import block (after 'import os')
    idx = src.index('\n', src.index('def main')) if 'def main' in src else 0
    # simpler: insert before the first 'def ' definition
    m = re.search(r'^def ', src, re.M)
    pos = m.start()
    src = src[:pos] + HELPER + '\n' + src[pos:]

CALL = ("            if args.vis_gaussian:\n"
        "                _save_surround(input_imgs, os.path.join(args.work_dir, 'vis'), f'val_{i_iter_val}_cam')\n")
anchor = "            result_dict = my_model(imgs=input_imgs, metas=data)\n"
if "_save_surround(input_imgs" not in src:
    src = src.replace(anchor, anchor + CALL, 1)

open(P, 'w').write(src)
print('patched OK, has helper:', '_save_surround' in src,
      'has call:', "_save_surround(input_imgs" in src)
