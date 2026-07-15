"""Prepare a CPU-sized Agriculture-Vision (2017) training subset.

The raw 2017 "miniscale" release stores, per 512x512 patch:
  field_images/rgb/<stem>.jpg      RGB patch
  field_images/nir/<stem>.jpg      NIR patch (unused here)
  field_labels/<class>/<stem>.png  one binary mask per anomaly class
  field_masks/<stem>.png           valid-region mask (255 = valid)

This script composes the 8 binary class masks into a single label map
(0 = background/healthy, 1..8 = anomaly class, 255 = invalid/ignore) and
copies a balanced subset into an images/ + masks/ + splits/ layout that the
training script consumes. The split (train/val/test) follows the official
field-level split in data2017_splits.json so no field leaks across splits.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

# Canonical class order -> integer id. id 0 is background.
CLASSES = ["background", "drydown", "endrow", "nutrient_deficiency",
           "planter_skip", "storm_damage", "water", "waterway", "weed_cluster"]
CLASS_ID = {c: i for i, c in enumerate(CLASSES)}
ANOMALY = CLASSES[1:]
# Paint order: low priority first so rarer classes win in overlaps.
PAINT_ORDER = ["drydown", "weed_cluster", "nutrient_deficiency", "endrow",
               "water", "waterway", "planter_skip", "storm_damage"]
IGNORE = 255


def compose_label(raw: Path, stem: str) -> np.ndarray:
    valid = np.array(Image.open(raw / "field_masks" / f"{stem}.png"))
    label = np.zeros(valid.shape[:2], dtype=np.uint8)
    for cname in PAINT_ORDER:
        p = raw / "field_labels" / cname / f"{stem}.png"
        if p.exists():
            m = np.array(Image.open(p))
            label[m > 0] = CLASS_ID[cname]
    label[valid == 0] = IGNORE
    return label


def has_anomaly(raw: Path, stem: str) -> bool:
    for cname in ANOMALY:
        p = raw / "field_labels" / cname / f"{stem}.png"
        if p.exists() and np.asarray(Image.open(p)).max() > 0:
            return True
    return False


def stems_by_field(rgb_dir: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for f in os.listdir(rgb_dir):
        if f.endswith(".jpg"):
            stem = f[:-4]
            field = stem.split("_")[0]
            out[field].append(stem)
    return out


def select(stems: list[str], raw: Path, n: int, anomaly_frac: float, rng: random.Random) -> list[str]:
    rng.shuffle(stems)
    anom, bg = [], []
    for s in stems:
        (anom if has_anomaly(raw, s) else bg).append(s)
        if len(anom) + len(bg) >= n * 4:  # enough to choose from
            break
    n_anom = min(len(anom), int(n * anomaly_frac))
    n_bg = min(len(bg), n - n_anom)
    chosen = anom[:n_anom] + bg[:n_bg]
    rng.shuffle(chosen)
    return chosen[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path,
                    default=Path(r"C:\Users\Ahmed\av_work\data2017\data2017_miniscale"))
    ap.add_argument("--splits-json", type=Path,
                    default=Path(r"C:\Users\Ahmed\av_work\data2017_splits.json"))
    ap.add_argument("--out", type=Path, default=Path(r"C:\Users\Ahmed\av_work\av_subset"))
    ap.add_argument("--n-train", type=int, default=1200)
    ap.add_argument("--n-val", type=int, default=300)
    ap.add_argument("--n-test", type=int, default=300)
    ap.add_argument("--anomaly-frac", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    raw = args.raw
    rgb_dir = raw / "field_images" / "rgb"
    field_split = json.loads(args.splits_json.read_text())
    by_field = stems_by_field(rgb_dir)

    (args.out / "images").mkdir(parents=True, exist_ok=True)
    (args.out / "masks").mkdir(parents=True, exist_ok=True)
    (args.out / "splits").mkdir(parents=True, exist_ok=True)

    targets = {"train": args.n_train, "val": args.n_val, "test": args.n_test}
    pixel_counts = np.zeros(len(CLASSES), dtype=np.int64)
    summary = {}

    for split, n in targets.items():
        fields = field_split.get(split, [])
        pool: list[str] = []
        for fid in fields:
            pool.extend(by_field.get(fid, []))
        chosen = select(pool, raw, n, args.anomaly_frac, rng)
        lines = []
        for stem in chosen:
            shutil.copy2(rgb_dir / f"{stem}.jpg", args.out / "images" / f"{stem}.jpg")
            label = compose_label(raw, stem)
            Image.fromarray(label).save(args.out / "masks" / f"{stem}.png")
            if split == "train":
                vals, cnts = np.unique(label[label != IGNORE], return_counts=True)
                for v, c in zip(vals, cnts):
                    pixel_counts[v] += c
            lines.append(stem)
        (args.out / "splits" / f"{split}.txt").write_text("\n".join(lines))
        summary[split] = len(lines)
        print(f"{split}: {len(lines)} patches from {len(fields)} fields")

    # class pixel frequency + inverse-frequency weights (train split)
    freq = pixel_counts / max(1, pixel_counts.sum())
    weights = 1.0 / np.sqrt(freq + 1e-6)
    weights = weights / weights.mean()
    meta = {
        "classes": CLASSES,
        "counts": summary,
        "train_pixel_fraction": {CLASSES[i]: float(freq[i]) for i in range(len(CLASSES))},
        "class_weights": {CLASSES[i]: float(weights[i]) for i in range(len(CLASSES))},
        "ignore_index": IGNORE,
        "source": "Agriculture-Vision 2017 miniscale (CVPR 2020 release)",
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))
    print("\nTrain pixel fraction per class:")
    for i, c in enumerate(CLASSES):
        print(f"  {c:20s} {freq[i]*100:6.2f}%   weight={weights[i]:.2f}")
    print("\nwrote subset ->", args.out)


if __name__ == "__main__":
    main()
