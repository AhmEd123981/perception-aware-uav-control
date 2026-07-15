"""Plot thesis training curves from deep_learning/weights/training_log.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Class names are chosen from the log's per-class width so this works for both the
# real Agriculture-Vision 4-class model and the older synthetic 5-class benchmark.
CLASS_SCHEMES = {
    4: ["background", "drydown", "weed_cluster", "double_plant"],
    5: ["soil", "healthy_crop", "diseased_crop", "weed_patch", "water_stress"],
}


def class_names(log: list[dict]) -> list[str]:
    n = 0
    for row in log:
        pci = row.get("val", {}).get("per_class_iou")
        if isinstance(pci, list) and pci:
            n = len(pci)
            break
    return CLASS_SCHEMES.get(n, [f"class_{i}" for i in range(n)])


def load_log(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError(f"Training log is empty or invalid: {path}")
    return data


def extract_curve(log: list[dict], key: str) -> list[float]:
    values: list[float] = []
    for row in log:
        if key == "train_loss":
            values.append(float(row.get("train_loss", np.nan)))
        else:
            values.append(float(row.get("val", {}).get(key, np.nan)))
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, default=Path("deep_learning/weights/training_log_2020.json"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/perception_logs/day7_thesis_package/training_curves.png"),
    )
    args = parser.parse_args()

    log = load_log(args.log)
    classes = class_names(log)
    epochs = [int(row["epoch"]) for row in log]
    train_loss = extract_curve(log, "train_loss")
    val_miou = extract_curve(log, "mean_iou")
    per_class_iou = np.array([
        row.get("val", {}).get("per_class_iou", [np.nan] * len(classes)) for row in log
    ], dtype=float)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    axes[0].plot(epochs, train_loss, marker="o", linewidth=1.8)
    axes[0].set_ylabel("Train loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, val_miou, marker="o", linewidth=1.8, color="#1f77b4")
    axes[1].set_ylabel("Val mIoU")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)

    for idx, name in enumerate(classes):
        axes[2].plot(epochs, per_class_iou[:, idx], marker=".", linewidth=1.4, label=name)
    axes[2].set_ylabel("Per-class IoU")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylim(0, 1)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(ncol=3, fontsize=8)

    fig.suptitle("Crop Segmentation Training Curves")
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
