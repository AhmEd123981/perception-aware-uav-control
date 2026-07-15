"""Download the full Agriculture-Vision CVPR-2020 dataset (2017+2018+2019) and
build one combined, balanced training subset.

We already have 2017 extracted. This script downloads the 2018 and 2019
"miniscale" packages from the public AWS bucket (anonymous), extracts them,
deletes the tarballs, then draws a balanced subset from all three years into a
single combined dataset that the trainer consumes.

Classes (written as final ids): background(0), drydown(1), weed_cluster(2),
double_plant(3), 255=ignore.  Split follows each year's official field split.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import tarfile
import time
from pathlib import Path

import boto3
import numpy as np
from botocore import UNSIGNED
from botocore.config import Config
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from av_prepare_subset_v2 import CLASSES, compose, select  # reuse logic

BUCKET = "intelinair-data-releases"
BASE = "agriculture-vision/cvpr_paper_2020/Dataset"
WORK = Path(r"C:\Users\Ahmed\av_work")
YEARS = [2017, 2018, 2019]

s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))


def download(key: str, dest: Path) -> None:
    size = s3.head_object(Bucket=BUCKET, Key=key)["ContentLength"]
    print(f"  downloading {dest.name} ({size/1e9:.2f} GB)", flush=True)
    st = {"n": 0, "t": time.time()}

    def cb(b):
        st["n"] += b
        if time.time() - st["t"] > 10:
            print(f"    {st['n']/1e9:.2f}/{size/1e9:.2f} GB", flush=True)
            st["t"] = time.time()

    s3.download_file(BUCKET, key, str(dest), Callback=cb)


def ensure_year(year: int) -> Path:
    raw = WORK / f"data{year}" / f"data{year}_miniscale"
    splits = WORK / f"data{year}_splits.json"
    if not splits.exists():
        s3.download_file(BUCKET, f"{BASE}/data{year}_splits.json", str(splits))
    if raw.exists() and (raw / "field_images" / "rgb").exists():
        print(f"year {year}: already extracted", flush=True)
        return raw
    tar = WORK / f"data{year}_miniscale.tar.gz"
    if not tar.exists():
        download(f"{BASE}/data{year}_miniscale.tar.gz", tar)
    print(f"year {year}: extracting...", flush=True)
    (WORK / f"data{year}").mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar, "r:gz") as tf:
        tf.extractall(WORK / f"data{year}")
    tar.unlink()  # reclaim space
    print(f"year {year}: extracted, tarball removed", flush=True)
    return raw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=WORK / "av_subset_2020")
    ap.add_argument("--per-year-train", type=int, default=1500)
    ap.add_argument("--per-year-val", type=int, default=300)
    ap.add_argument("--per-year-test", type=int, default=300)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    for sub in ("images", "masks", "splits"):
        d = args.out / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    per_split = {"train": args.per_year_train, "val": args.per_year_val, "test": args.per_year_test}
    combined = {"train": [], "val": [], "test": []}
    pix = np.zeros(len(CLASSES), dtype=np.int64)

    for year in YEARS:
        raw = ensure_year(year)
        field_split = json.loads((WORK / f"data{year}_splits.json").read_text())
        by_field = {}
        for f in os.listdir(raw / "field_images" / "rgb"):
            if f.endswith(".jpg"):
                by_field.setdefault(f[:-4].split("_")[0], []).append(f[:-4])
        for split, n in per_split.items():
            pool = []
            for fid in field_split.get(split, []):
                pool.extend(by_field.get(fid, []))
            chosen = select(pool, raw, n, rng)
            for stem in chosen:
                new = f"{year}_{stem}"
                shutil.copy2(raw / "field_images" / "rgb" / f"{stem}.jpg",
                             args.out / "images" / f"{new}.jpg")
                lab = compose(raw, stem)
                Image.fromarray(lab).save(args.out / "masks" / f"{new}.png")
                combined[split].append(new)
                if split == "train":
                    v, c = np.unique(lab[lab != 255], return_counts=True)
                    for vi, ci in zip(v, c):
                        pix[vi] += ci
            print(f"  year {year} {split}: {len(chosen)} patches", flush=True)

    for split, stems in combined.items():
        rng.shuffle(stems)
        (args.out / "splits" / f"{split}.txt").write_text("\n".join(stems))

    frac = pix / max(1, pix.sum())
    (args.out / "meta.json").write_text(json.dumps({
        "classes": CLASSES, "years": YEARS,
        "counts": {k: len(v) for k, v in combined.items()},
        "train_pixel_fraction": {CLASSES[i]: float(frac[i]) for i in range(len(CLASSES))},
        "source": "Agriculture-Vision CVPR-2020 (2017+2018+2019 miniscale)",
    }, indent=2))
    print("\ncombined counts:", {k: len(v) for k, v in combined.items()}, flush=True)
    print("train pixel fraction:", {CLASSES[i]: round(float(frac[i]) * 100, 2) for i in range(len(CLASSES))}, flush=True)
    print("wrote ->", args.out, flush=True)
    print("DONE_BUILD_2020", flush=True)


if __name__ == "__main__":
    main()
