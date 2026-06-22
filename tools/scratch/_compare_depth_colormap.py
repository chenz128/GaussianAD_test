"""One-off: compare OLD vs NEW depth colormap on a real metric3d depth map.

Layout (2 rows x 2 cols), per camera block:
  row 1: real GT depth      [OLD cmap vmax=40 | NEW jet vmax=30]
  row 2: simulated PRED      [OLD cmap vmax=40 | NEW jet vmax=30]
         (sim PRED = GT clipped to 30m to mimic the ±30m Gaussian range,
          far/sky >50m set to 0 to mimic "no Gaussian coverage")

Usage:
  python tools/_compare_depth_colormap.py <depth_npy> [out_jpg]
"""
import sys
import numpy as np
import cv2

CAMS = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']


def old_cmap(d, vmax=40.0):
    norm = np.clip(d / vmax, 0.0, 1.0)
    r = np.clip(norm * 4 - 2, 0, 1)
    g = np.clip(np.minimum(norm * 4, 4 - norm * 4), 0, 1)
    b = np.clip(1 - norm * 4, 0, 1)
    rgb = np.stack([r, g, b], -1)
    rgb[d <= 0] = 0.5
    return (rgb * 255).astype(np.uint8)


def new_cmap(d, vmax=30.0):
    norm = np.clip(d / vmax, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4 * norm - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * norm - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * norm - 1), 0, 1)
    rgb = np.stack([r, g, b], -1)
    rgb[d <= 0] = 0.5
    return (rgb * 255).astype(np.uint8)


def label(img, txt):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(img, txt, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    path = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else '/tmp/depth_cmap_compare.jpg'
    arr = np.load(path).astype(np.float32)        # (6, 900, 1600)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    blocks = []
    for cam in (1, 4):                             # CAM_FRONT, CAM_BACK
        d = cv2.resize(arr[cam], (704, 396), interpolation=cv2.INTER_NEAREST)

        # simulated PRED: clip to 30m, far/sky (>50m) -> no coverage (0)
        d_pred = d.copy()
        sky = d_pred > 50.0
        d_pred = np.clip(d_pred, 0.0, 30.0)
        d_pred[sky] = 0.0

        row1 = np.concatenate([
            label(old_cmap(d, 40), f'{CAMS[cam]}  GT depth | OLD cmap (vmax=40)'),
            label(new_cmap(d, 30), f'{CAMS[cam]}  GT depth | NEW jet  (vmax=30)'),
        ], axis=1)
        row2 = np.concatenate([
            label(old_cmap(d_pred, 40), 'sim PRED (<=30m) | OLD cmap  -> all green'),
            label(new_cmap(d_pred, 30), 'sim PRED (<=30m) | NEW jet   -> graded'),
        ], axis=1)
        sep = np.full((3, row1.shape[1], 3), 60, np.uint8)
        blocks.extend([row1, sep, row2])
        blocks.append(np.full((8, row1.shape[1], 3), 30, np.uint8))

    canvas = np.concatenate(blocks[:-1], axis=0)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)   # our arrays are RGB
    cv2.imwrite(out, canvas)
    print('saved', out, canvas.shape)


if __name__ == '__main__':
    main()
