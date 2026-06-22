"""
分析 nuscenes_gs25600_4gpu_v4 模型生成的 Gaussian 透明度（opacity）分布。
用法（远端 H20）：
  cd /data/chenz/GaussianAD
  CUDA_VISIBLE_DEVICES=0 /data/chenz/conda_env/GaussianAD/bin/python \
      tools/viz/analyze_opacity.py \
      --py-config config/nuscenes_gs25600.py \
      --work-dir out/nuscenes_gs25600_4gpu_v4 \
      --num-batches 100
"""
import argparse, os, os.path as osp, sys
import torch, numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--py-config', default='config/nuscenes_gs25600.py')
    parser.add_argument('--work-dir', default='out/nuscenes_gs25600_4gpu_v4')
    parser.add_argument('--num-batches', type=int, default=100,
                        help='采样多少个 val batch（每 batch 1 帧，3600 个高斯）')
    args = parser.parse_args()

    import warnings; warnings.filterwarnings("ignore")

    from mmengine import Config
    from mmengine.runner import set_random_seed
    from mmengine.logging import MMLogger
    from mmseg.models import build_segmentor

    set_random_seed(42)
    torch.backends.cudnn.benchmark = True

    cfg = Config.fromfile(args.py_config)
    cfg.work_dir = args.work_dir

    log_file = osp.join(args.work_dir, 'analyze_opacity.log')
    os.makedirs(args.work_dir, exist_ok=True)
    logger = MMLogger('analyze_opacity', log_file=log_file)
    MMLogger._instance_dict['analyze_opacity'] = logger

    import model as _model_module
    from dataset import get_dataloader

    my_model = build_segmentor(cfg.model)
    my_model.init_weights()
    my_model = my_model.cuda()

    # 加载最新 checkpoint
    ckpt_path = osp.join(args.work_dir, 'latest.pth')
    assert osp.exists(ckpt_path), f"找不到 {ckpt_path}"
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    missing, unexpected = my_model.load_state_dict(state, strict=False)
    print(f"checkpoint loaded from {ckpt_path}")
    if missing:
        print(f"  missing keys ({len(missing)}): {missing[:5]}")
    if unexpected:
        print(f"  unexpected keys ({len(unexpected)}): {unexpected[:5]}")

    # 强制 num_workers=0 防止多进程 dataloader 在某些环境下死锁
    val_loader_cfg = dict(cfg.val_loader)
    val_loader_cfg['num_workers'] = 0
    _, val_loader = get_dataloader(
        cfg.train_dataset_config,
        cfg.val_dataset_config,
        cfg.train_loader,
        val_loader_cfg,
        dist=False,
        val_only=True,
    )

    my_model.eval()
    os.environ['eval'] = 'true'

    all_opas = []  # 每次 append (G,) numpy array

    print(f"\n开始推理，最多采样 {args.num_batches} 个 batch ...")
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            if i >= args.num_batches:
                break

            for k in list(data.keys()):
                if isinstance(data[k], torch.Tensor):
                    data[k] = data[k].cuda()
            input_imgs = data.pop('img')
            result = my_model(imgs=input_imgs, metas=data)

            # representation = {'gaussian': GaussianPrediction}
            rep = result.get('representation', None)
            if rep is None:
                print(f"[WARN] batch {i}: result 中没有 'representation'，跳过")
                continue
            gaussian = rep.get('gaussian', None) if isinstance(rep, dict) else None
            if gaussian is None:
                print(f"[WARN] batch {i}: representation 中没有 'gaussian'，跳过")
                continue

            opa = gaussian.opacities  # (B, G, 1)
            all_opas.append(opa.squeeze(-1).flatten().cpu().float().numpy())

            if (i + 1) % 10 == 0:
                print(f"  已处理 {i+1} 个 batch ...")

    if len(all_opas) == 0:
        print("[ERROR] 未采集到任何 opacity 数据，请检查模型输出")
        sys.exit(1)

    opas = np.concatenate(all_opas)  # (N_total,)
    n_total = len(opas)
    print(f"\n共采集 {n_total} 个 Gaussian opacity 值（{len(all_opas)} 帧 × ~{n_total//len(all_opas)} 个/帧）")

    # ── 基础统计 ──────────────────────────────────────────
    print("\n===== 基础统计 =====")
    print(f"  min:    {opas.min():.6f}")
    print(f"  max:    {opas.max():.6f}")
    print(f"  mean:   {opas.mean():.6f}")
    print(f"  median: {np.median(opas):.6f}")
    print(f"  std:    {opas.std():.6f}")

    quantiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    print("\n  分位数:")
    for q in quantiles:
        print(f"    p{int(q*100):>3d}%: {np.quantile(opas, q):.6f}")

    # ── 区间分布 ──────────────────────────────────────────
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    bin_labels = ['[0.0, 0.1)', '[0.1, 0.2)', '[0.2, 0.3)', '[0.3, 0.4)',
                  '[0.4, 0.5)', '[0.5, 0.6)', '[0.6, 0.7)', '[0.7, 0.8)',
                  '[0.8, 0.9)', '[0.9, 1.0]']
    counts, _ = np.histogram(opas, bins=bins)
    print("\n===== 区间分布 =====")
    print(f"  {'区间':<16} {'数量':>10}  {'占比':>8}  {'累计占比':>10}")
    print("  " + "-" * 50)
    cumsum = 0
    for label, cnt in zip(bin_labels, counts):
        pct = cnt / n_total * 100
        cumsum += pct
        bar = '█' * int(pct / 2)
        print(f"  {label:<16} {cnt:>10,}  {pct:>7.2f}%  {cumsum:>9.2f}%  {bar}")

    # ── 极端区间细化 ──────────────────────────────────────
    print("\n===== 高透明度区间细化（>0.9）=====")
    fine_bins = [0.90, 0.92, 0.94, 0.96, 0.98, 1.01]
    fine_labels = ['[0.90, 0.92)', '[0.92, 0.94)', '[0.94, 0.96)', '[0.96, 0.98)', '[0.98, 1.00]']
    counts_fine, _ = np.histogram(opas, bins=fine_bins)
    for label, cnt in zip(fine_labels, counts_fine):
        pct = cnt / n_total * 100
        print(f"  {label:<16} {cnt:>10,}  {pct:>7.3f}%")

    print("\n===== 低透明度区间细化（<0.1）=====")
    fine_bins2 = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1]
    fine_labels2 = ['[0.00, 0.02)', '[0.02, 0.04)', '[0.04, 0.06)', '[0.06, 0.08)', '[0.08, 0.10)']
    counts_fine2, _ = np.histogram(opas, bins=fine_bins2)
    for label, cnt in zip(fine_labels2, counts_fine2):
        pct = cnt / n_total * 100
        print(f"  {label:<16} {cnt:>10,}  {pct:>7.3f}%")

    # ── 双峰诊断 ──────────────────────────────────────────
    near_zero = (opas < 0.1).mean() * 100
    near_one  = (opas > 0.9).mean() * 100
    middle    = ((opas >= 0.1) & (opas <= 0.9)).mean() * 100
    print(f"\n===== 双峰诊断 =====")
    print(f"  近零（< 0.1）:  {near_zero:.2f}%")
    print(f"  中间（0.1~0.9）: {middle:.2f}%")
    print(f"  近一（> 0.9）:  {near_one:.2f}%")
    if near_zero + near_one > 70:
        print("  ➜ 分布呈双峰（稀疏化明显）")
    else:
        print("  ➜ 分布较均匀（未出现明显稀疏化）")

    print("\n分析完成。")


if __name__ == '__main__':
    main()
