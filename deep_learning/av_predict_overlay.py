"""Visualize Agriculture-Vision model predictions on real held-out test patches.

Loads the trained checkpoint, runs inference on a sample of test images, and
writes side-by-side panels: original aerial RGB | predicted problem map |
ground-truth problem map. This is the qualitative "it works on real imagery"
evidence for the thesis/paper.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import sys
import os

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_av as T

# class -> RGB color (background transparent)
COLORS = {
    0: (0, 0, 0),        # background (not drawn)
    1: (230, 159, 0),    # drydown - orange
    2: (213, 30, 30),    # weed_cluster - red
    3: (0, 158, 115),    # double_plant - green
}
LEGEND = {1: "drydown", 2: "weed_cluster", 3: "double_plant"}


def colorize(label: np.ndarray) -> np.ndarray:
    out = np.zeros((*label.shape, 3), dtype=np.uint8)
    for cid, col in COLORS.items():
        out[label == cid] = col
    return out


def overlay(rgb: np.ndarray, label: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    col = colorize(label)
    mask = (label > 0)[..., None]
    blended = np.where(mask, (alpha * col + (1 - alpha) * rgb).astype(np.uint8), rgb)
    return blended


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(r"C:\Users\Ahmed\av_work\av_subset"))
    ap.add_argument("--ckpt", type=Path, default=Path("deep_learning/weights/av_segmentation_2020_best.pt"))
    ap.add_argument("--out", type=Path, default=Path("results/analysis/av_predictions"))
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--image-size", type=int, default=256)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    model = T.build_model(False).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    stems = [l.strip() for l in (args.root / "splits" / "test.txt").read_text().splitlines() if l.strip()]
    # prefer patches that actually contain an anomaly so the panels are informative
    scored = []
    for s in stems:
        m = T._LUT[cv2.imread(str(args.root / "masks" / f"{s}.png"), cv2.IMREAD_GRAYSCALE)]
        scored.append((int(((m > 0) & (m != 255)).sum()), s))
    scored.sort(reverse=True)
    pick = [s for _, s in scored[:args.n]]

    for s in pick:
        img = cv2.cvtColor(cv2.imread(str(args.root / "images" / f"{s}.jpg")), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (args.image_size, args.image_size))
        gt = T._LUT[cv2.imread(str(args.root / "masks" / f"{s}.png"), cv2.IMREAD_GRAYSCALE)]
        gt = cv2.resize(gt, (args.image_size, args.image_size), interpolation=cv2.INTER_NEAREST)
        x = torch.from_numpy(np.ascontiguousarray(
            (img.astype(np.float32) / 255.0).transpose(2, 0, 1))).unsqueeze(0)
        with torch.no_grad():
            pred = torch.argmax(model(x), 1)[0].cpu().numpy().astype(np.uint8)
        panel = np.concatenate([img, overlay(img, pred), overlay(img, gt)], axis=1)
        cv2.imwrite(str(args.out / f"pred_{s}.png"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))

    # legend image
    legend = np.full((90, 360, 3), 255, np.uint8)
    y = 20
    for cid, name in LEGEND.items():
        cv2.rectangle(legend, (10, y - 12), (34, y + 6), COLORS[cid][::-1], -1)
        cv2.putText(legend, name, (44, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        y += 28
    cv2.imwrite(str(args.out / "legend.png"), legend)
    print(f"wrote {len(pick)} prediction panels (left=RGB, middle=prediction, right=ground truth) -> {args.out}")


if __name__ == "__main__":
    main()
