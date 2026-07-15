"""Build a larger, better-balanced Agriculture-Vision 2017 subset.

Improvements over av_prepare_subset.py:
- Uses the three classes with real 2017 support: drydown (7550 patches),
  weed_cluster (5296) and double_plant (3774). water/others are too rare and
  fold into background.
- Targeted selection so weed_cluster and double_plant (the less dominant
  classes) are well represented instead of being swamped by drydown.
- Writes masks already in final target ids {0,1,2,3, 255=ignore}.

Output layout: <out>/images/<stem>.jpg, <out>/masks/<stem>.png,
<out>/splits/{train,val,test}.txt, <out>/meta.json
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

CLASSES = ["background", "drydown", "weed_cluster", "double_plant"]
# raw label folder -> target id; paint low->high priority (rarer wins overlaps)
PAINT = [("drydown", 1), ("weed_cluster", 2), ("double_plant", 3)]
IGNORE = 255


def compose(raw: Path, stem: str) -> np.ndarray:
    valid = np.array(Image.open(raw / "field_masks" / f"{stem}.png"))
    lab = np.zeros(valid.shape[:2], dtype=np.uint8)
    for folder, tid in PAINT:
        p = raw / "field_labels" / folder / f"{stem}.png"
        if p.exists():
            m = np.array(Image.open(p))
            lab[m > 0] = tid
    lab[valid == 0] = IGNORE
    return lab


def label_present(raw: Path, stem: str, folder: str) -> bool:
    p = raw / "field_labels" / folder / f"{stem}.png"
    return p.exists() and np.asarray(Image.open(p)).max() > 0


def stems_by_field(rgb_dir: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for f in os.listdir(rgb_dir):
        if f.endswith(".jpg"):
            out[f[:-4].split("_")[0]].append(f[:-4])
    return out


def select(pool: list[str], raw: Path, n: int, rng: random.Random) -> list[str]:
    """Pick n patches: ~40% with double_plant, ~40% with weed, ~20% other."""
    rng.shuffle(pool)
    dp, weed, other = [], [], []
    cap = n * 5
    for s in pool:
        if len(dp) + len(weed) + len(other) >= cap:
            break
        if label_present(raw, s, "double_plant"):
            dp.append(s)
        elif label_present(raw, s, "weed_cluster"):
            weed.append(s)
        else:
            other.append(s)
    n_dp = min(len(dp), int(n * 0.40))
    n_weed = min(len(weed), int(n * 0.40))
    n_other = min(len(other), n - n_dp - n_weed)
    chosen = dp[:n_dp] + weed[:n_weed] + other[:n_other]
    # top up if short
    leftovers = dp[n_dp:] + weed[n_weed:] + other[n_other:]
    rng.shuffle(leftovers)
    chosen += leftovers[: max(0, n - len(chosen))]
    rng.shuffle(chosen)
    return chosen[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path,
                    default=Path(r"C:\Users\Ahmed\av_work\data2017\data2017_miniscale"))
    ap.add_argument("--splits-json", type=Path,
                    default=Path(r"C:\Users\Ahmed\av_work\data2017_splits.json"))
    ap.add_argument("--out", type=Path, default=Path(r"C:\Users\Ahmed\av_work\av_subset_v2"))
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-val", type=int, default=600)
    ap.add_argument("--n-test", type=int, default=600)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    raw = args.raw
    rgb_dir = raw / "field_images" / "rgb"
    field_split = json.loads(args.splits_json.read_text())
    by_field = stems_by_field(rgb_dir)
    for sub in ("images", "masks", "splits"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)

    targets = {"train": args.n_train, "val": args.n_val, "test": args.n_test}
    pix = np.zeros(len(CLASSES), dtype=np.int64)
    patch_has = {c: 0 for c in CLASSES[1:]}
    counts = {}

    for split, n in targets.items():
        pool: list[str] = []
        for fid in field_split.get(split, []):
            pool.extend(by_field.get(fid, []))
        chosen = select(pool, raw, n, rng)
        for stem in chosen:
            shutil.copy2(rgb_dir / f"{stem}.jpg", args.out / "images" / f"{stem}.jpg")
            lab = compose(raw, stem)
            Image.fromarray(lab).save(args.out / "masks" / f"{stem}.png")
            if split == "train":
                v, c = np.unique(lab[lab != IGNORE], return_counts=True)
                for vi, ci in zip(v, c):
                    pix[vi] += ci
                for ci, name in enumerate(CLASSES):
                    if ci > 0 and (lab == ci).any():
                        patch_has[name] += 1
        (args.out / "splits" / f"{split}.txt").write_text("\n".join(chosen))
        counts[split] = len(chosen)
        print(f"{split}: {len(chosen)} patches")

    frac = pix / max(1, pix.sum())
    print("\nTrain pixel fraction / patches-containing:")
    for i, c in enumerate(CLASSES):
        ph = patch_has.get(c, counts["train"])
        print(f"  {c:<14}{frac[i]*100:6.2f}%   patches_with={ph if i>0 else counts['train']}")
    (args.out / "meta.json").write_text(json.dumps({
        "classes": CLASSES, "counts": counts,
        "train_pixel_fraction": {CLASSES[i]: float(frac[i]) for i in range(len(CLASSES))},
        "train_patches_with_class": patch_has,
        "source": "Agriculture-Vision 2017 miniscale (CVPR 2020)",
    }, indent=2))
    print("\nwrote ->", args.out)


if __name__ == "__main__":
    main()
