#!/usr/bin/env python3

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


for style_name in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot"):
    try:
        plt.style.use(style_name)
        break
    except OSError:
        continue


TRAIN_RE = re.compile(
    r"\[TRAIN\] Epoch\s+(?P<epoch>\d+) Iter\s+(?P<iter>\d+)\/(?P<iters_per_epoch>\d+): "
    r"Loss: (?P<loss>[0-9.]+) \((?P<loss_avg>[0-9.]+)\), grad_norm: (?P<grad_norm>[0-9.]+), "
    r"lr: (?P<lr>[0-9.eE+-]+), time: (?P<iter_time>[0-9.]+) \((?P<data_time>[0-9.]+)\)"
)

LOSS_RE = re.compile(
    r"OccupancyLoss: (?P<occupancy_loss>[0-9.]+), "
    r"OccupancyFlowLoss: (?P<occupancy_flow_loss>[0-9.]+), "
    r"DetectionLoss: (?P<detection_loss>[0-9.]+), "
    r"MapLoss: (?P<map_loss>[0-9.]+), "
    r"PlanLoss: (?P<plan_loss>[0-9.]+)"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Parse GaussianAD training log and generate line charts.")
    parser.add_argument("--log", required=True, help="Path to training log file.")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory to save plots. Default: sibling folder named <log_stem>_plots.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="DPI for output images.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=11,
        help="Moving-average window for smoothing. Set to 1 to disable smoothing.",
    )
    return parser.parse_args()


def parse_log(log_path):
    records = []
    pending = None

    with open(log_path, "r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()

            train_match = TRAIN_RE.search(line)
            if train_match:
                data = {key: float(value) for key, value in train_match.groupdict().items()}
                data["epoch"] = int(data["epoch"])
                data["iter"] = int(data["iter"])
                data["iters_per_epoch"] = int(data["iters_per_epoch"])
                data["global_step"] = data["epoch"] * data["iters_per_epoch"] + data["iter"]
                pending = data
                records.append(pending)
                continue

            loss_match = LOSS_RE.search(line)
            if loss_match and pending is not None:
                pending.update({key: float(value) for key, value in loss_match.groupdict().items()})
                pending = None

    return records


def ensure_out_dir(log_path, out_dir):
    if out_dir:
        path = Path(out_dir)
    else:
        path = Path(log_path).with_suffix("")
        path = path.parent / f"{path.name}_plots"
    path.mkdir(parents=True, exist_ok=True)
    return path


def smooth_values(values, window):
    if window <= 1 or len(values) < 3:
        return np.asarray(values, dtype=np.float64)

    window = min(window, len(values))
    if window % 2 == 0:
        window -= 1
    if window <= 1:
        return np.asarray(values, dtype=np.float64)

    arr = np.asarray(values, dtype=np.float64)
    pad = window // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(padded, kernel, mode="valid")


def plot_smoothed_series(ax, x, y, label, smooth_window, color=None):
    y = np.asarray(y, dtype=np.float64)
    smoothed = smooth_values(y, smooth_window)

    if smooth_window > 1:
        ax.plot(x, y, color=color, linewidth=0.9, alpha=0.18)
    ax.plot(x, smoothed, label=label, color=color, linewidth=2.2)


def save_overview(records, out_dir, dpi, smooth_window):
    x = [item["global_step"] for item in records]

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    plot_smoothed_series(axes[0], x, [item["loss"] for item in records], "loss", smooth_window, color="tab:blue")
    plot_smoothed_series(axes[0], x, [item["loss_avg"] for item in records], "loss_avg", smooth_window, color="tab:orange")
    axes[0].set_title("Training Loss")
    axes[0].set_ylabel("loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    plot_smoothed_series(axes[1], x, [item["grad_norm"] for item in records], "grad_norm", smooth_window, color="tab:orange")
    axes[1].set_title("Gradient Norm")
    axes[1].set_ylabel("grad_norm")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plot_smoothed_series(axes[2], x, [item["lr"] for item in records], "lr", max(1, smooth_window // 2), color="tab:green")
    axes[2].set_title("Learning Rate")
    axes[2].set_ylabel("lr")
    axes[2].set_xlabel("global_step")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(out_dir / "train_overview.png", dpi=dpi)
    plt.close(fig)


def save_component_losses(records, out_dir, dpi, smooth_window):
    component_keys = [
        "occupancy_loss",
        "occupancy_flow_loss",
        "detection_loss",
        "map_loss",
        "plan_loss",
    ]
    available = [key for key in component_keys if any(key in item for item in records)]
    if not available:
        return

    x = [item["global_step"] for item in records]
    fig, ax = plt.subplots(figsize=(14, 7))
    for key in available:
        y = [item.get(key) for item in records]
        plot_smoothed_series(ax, x, y, key, smooth_window, color=None)

    ax.set_title("Training Component Losses")
    ax.set_xlabel("global_step")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "train_component_losses.png", dpi=dpi)
    plt.close(fig)


def save_timing(records, out_dir, dpi, smooth_window):
    x = [item["global_step"] for item in records]
    fig, ax = plt.subplots(figsize=(14, 7))
    plot_smoothed_series(ax, x, [item["iter_time"] for item in records], "iter_time", smooth_window, color="tab:blue")
    plot_smoothed_series(ax, x, [item["data_time"] for item in records], "data_time", smooth_window, color="tab:orange")
    ax.set_title("Training Timing")
    ax.set_xlabel("global_step")
    ax.set_ylabel("seconds")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "train_timing.png", dpi=dpi)
    plt.close(fig)


def main():
    args = parse_args()
    records = parse_log(args.log)
    if not records:
        raise SystemExit(f"No training records found in log: {args.log}")

    out_dir = ensure_out_dir(args.log, args.out_dir)
    save_overview(records, out_dir, args.dpi, args.smooth_window)
    save_component_losses(records, out_dir, args.dpi, args.smooth_window)
    save_timing(records, out_dir, args.dpi, args.smooth_window)

    print(f"Parsed {len(records)} training points from {args.log}")
    print(f"Saved plots to {out_dir}")


if __name__ == "__main__":
    main()