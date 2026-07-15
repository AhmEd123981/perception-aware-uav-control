"""Generate a lightweight synthetic aerial crop segmentation dataset.

This script mirrors the MATLAB procedural renderer so training can start before
real orthomosaic imagery is available. Masks are saved as zero-based class IDs:
0 soil, 1 healthy crop, 2 diseased crop, 3 weed patch, 4 water stress.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np


CLASSES = ["soil", "healthy_crop", "diseased_crop", "weed_patch", "water_stress"]


def draw_field(rng: np.random.Generator, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    mask = np.zeros((height, width), dtype=np.uint8)
    base = np.array([110, 82, 48], dtype=np.int16)
    image = base + rng.integers(-12, 13, (height, width, 3), dtype=np.int16)
    image = np.clip(image, 0, 255).astype(np.uint8)

    row_period = rng.integers(42, 70)
    row_width = rng.integers(18, 34)
    for y in range(0, height, row_period):
        y2 = min(height, y + row_width)
        crop = np.array([38, 126, 48], dtype=np.int16)
        crop_texture = crop + rng.integers(-20, 21, (y2 - y, width, 3), dtype=np.int16)
        image[y:y2, :, :] = np.clip(crop_texture, 0, 255).astype(np.uint8)
        mask[y:y2, :] = 1

    for class_id, color in [(2, (150, 130, 40)), (3, (45, 145, 65)), (4, (190, 180, 65))]:
        for _ in range(rng.integers(1, 4)):
            margin_x = max(10, width // 10)
            margin_y = max(10, height // 10)
            cx = int(rng.integers(margin_x, max(margin_x + 1, width - margin_x)))
            cy = int(rng.integers(margin_y, max(margin_y + 1, height - margin_y)))
            ax = int(rng.integers(max(4, width // 35), max(5, width // 8)))
            ay = int(rng.integers(max(4, height // 35), max(5, height // 8)))
            angle = float(rng.uniform(0, 180))
            cv2.ellipse(mask, (cx, cy), (ax, ay), angle, 0, 360, int(class_id), -1)
            cv2.ellipse(image, (cx, cy), (ax, ay), angle, 0, 360, color, -1)

    if random.random() < 0.25:
        image = cv2.GaussianBlur(image, (5, 5), 0)
    return image, mask


def main(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    for sub in ["images", "masks", "metadata"]:
        (args.output / sub).mkdir(parents=True, exist_ok=True)

    stems = []
    for i in range(args.count):
        stem = f"synthetic_{i:06d}"
        image, mask = draw_field(rng, args.width, args.height)
        cv2.imwrite(str(args.output / "images" / f"{stem}.png"), image)
        cv2.imwrite(str(args.output / "masks" / f"{stem}.png"), mask)
        seed_bucket = i // args.bucket_size
        (args.output / "metadata" / f"{stem}.json").write_text(
            json.dumps({"classes": CLASSES, "sample_index": i, "seed_bucket": seed_bucket}, indent=2)
        )
        stems.append(stem)

    stems = stratified_bucket_split(stems, args.bucket_size)
    split_dir = args.output.parent / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    n_train = int(0.70 * len(stems))
    n_val = int(0.15 * len(stems))
    (split_dir / "train.txt").write_text("\n".join(stems[:n_train]))
    (split_dir / "val.txt").write_text("\n".join(stems[n_train:n_train + n_val]))
    (split_dir / "test.txt").write_text("\n".join(stems[n_train + n_val:]))


def stratified_bucket_split(stems: list[str], bucket_size: int) -> list[str]:
    """Sort samples by RNG seed bucket so neighboring buckets do not leak splits."""
    buckets: dict[int, list[str]] = {}
    for stem in stems:
        idx = int(stem.split("_")[1])
        bucket = idx // bucket_size
        buckets.setdefault(bucket, []).append(stem)
    ordered_buckets = sorted(buckets)
    return [stem for bucket in ordered_buckets for stem in sorted(buckets[bucket])]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("datasets/synthetic"))
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--bucket-size", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
