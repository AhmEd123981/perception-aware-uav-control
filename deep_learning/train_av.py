"""Train a U-Net on the real Agriculture-Vision (2017) aerial crop data.

This is the real-data counterpart to train.py (which used synthetic imagery).
The 2017 split has strong support for drydown, weed_cluster and water; the other
anomaly classes are nearly absent, so we segment a compact, learnable set:

    0 background / healthy field
    1 drydown          (dry / senescing crop)
    2 weed_cluster     (weed infestation)
    3 water            (standing water / flooding)

The remaining raw anomaly classes are folded into background at load time.
Invalid (out-of-field) pixels carry label 255 and are ignored in the loss and
metrics. Class-balanced cross-entropy compensates for the heavy imbalance.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

CLASSES = ["background", "drydown", "weed_cluster", "double_plant"]
IGNORE = 255

# av_prepare_subset_v2.py writes masks already in final target ids {0..3, 255},
# so the lookup is the identity (kept for a uniform code path / older subsets).
_LUT = np.arange(256, dtype=np.uint8)


def build_model(pretrained: bool = False) -> torch.nn.Module:
    return smp.Unet(encoder_name="resnet18",
                    encoder_weights="imagenet" if pretrained else None,
                    in_channels=3, classes=len(CLASSES))


class AVDataset(Dataset):
    def __init__(self, root: Path, split_file: Path, image_size: int = 256, train: bool = True):
        self.root = root
        self.items = [l.strip() for l in split_file.read_text().splitlines() if l.strip()]
        self.image_size = image_size
        self.transform = self._tf(train)

    def __len__(self):
        return len(self.items)

    def _tf(self, train):
        t = [A.Resize(self.image_size, self.image_size, interpolation=cv2.INTER_LINEAR)]
        if train:
            t += [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.3), A.RandomRotate90(p=0.5),
                  A.RandomBrightnessContrast(0.25, 0.25, p=0.5)]
        return A.Compose(t)

    def __getitem__(self, idx):
        stem = self.items[idx]
        img = cv2.imread(str(self.root / "images" / f"{stem}.jpg"), cv2.IMREAD_COLOR)
        msk = cv2.imread(str(self.root / "masks" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        if img is None or msk is None:
            raise FileNotFoundError(stem)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        msk = _LUT[msk]  # remap to target classes, keep 255 ignore
        tr = self.transform(image=img, mask=msk)
        image = tr["image"].astype(np.float32) / 255.0
        mask = tr["mask"].astype(np.int64)
        return (torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))),
                torch.from_numpy(np.ascontiguousarray(mask)))


def confusion_update(conf, t, p, n):
    t = t.cpu().numpy().ravel(); p = p.cpu().numpy().ravel()
    v = (t >= 0) & (t < n)
    return conf + np.bincount(n * t[v] + p[v], minlength=n * n).reshape(n, n)


def metrics_from_confusion(conf):
    tp = np.diag(conf).astype(float)
    tc = conf.sum(1).astype(float); pc = conf.sum(0).astype(float)
    union = tc + pc - tp
    iou = np.divide(tp, union, out=np.full_like(tp, np.nan), where=union > 0)
    prec = np.divide(tp, pc, out=np.full_like(tp, np.nan), where=pc > 0)
    rec = np.divide(tp, tc, out=np.full_like(tp, np.nan), where=tc > 0)
    return {"per_class_iou": iou.tolist(), "per_class_precision": prec.tolist(),
            "per_class_recall": rec.tolist(),
            "mean_iou": float(np.nanmean(iou)) if np.any(np.isfinite(iou)) else 0.0,
            "confusion_matrix": conf.astype(int).tolist()}


def print_table(epoch, loss, m):
    print(f"\nEpoch {epoch:03d} | train_loss={loss:.4f} | val_mIoU={m['mean_iou']:.4f}")
    print(f"{'class':<14}{'IoU':>9}{'prec':>9}{'recall':>9}")
    for i, c in enumerate(CLASSES):
        f = lambda v: "n/a" if not np.isfinite(v) else f"{v:.3f}"
        print(f"{c:<14}{f(m['per_class_iou'][i]):>9}{f(m['per_class_precision'][i]):>9}{f(m['per_class_recall'][i]):>9}")


def evaluate(model, loader, device, n):
    model.eval(); conf = np.zeros((n, n), np.int64)
    with torch.no_grad():
        for x, y in tqdm(loader, desc="val", leave=False):
            pred = torch.argmax(model(x.to(device)), 1)
            conf = confusion_update(conf, y.to(device), pred, n)
    return metrics_from_confusion(conf)


def compute_class_weights(ds, n):
    counts = np.zeros(n, np.float64)
    for stem in ds.items:
        m = _LUT[cv2.imread(str(ds.root / "masks" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)]
        m = m[m != IGNORE]
        counts += np.bincount(m, minlength=n)
    freq = counts / max(1.0, counts.sum())
    w = np.clip(np.sqrt(np.median(freq[freq > 0]) / (freq + 1e-9)), 0.5, 8.0)
    return torch.tensor(w, dtype=torch.float32), freq


def jr(v):
    if isinstance(v, dict): return {k: jr(x) for k, x in v.items()}
    if isinstance(v, list): return [jr(x) for x in v]
    if isinstance(v, float) and not np.isfinite(v): return None
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(r"C:\Users\Ahmed\av_work\av_subset"))
    ap.add_argument("--out", type=Path, default=Path("deep_learning/weights/av_segmentation.pt"))
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--early-stop", type=int, default=6)
    args = ap.parse_args()

    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cpu")
    n = len(CLASSES)
    sp = args.root / "splits"
    tr_ds = AVDataset(args.root, sp / "train.txt", args.image_size, True)
    va_ds = AVDataset(args.root, sp / "val.txt", args.image_size, False)
    te_ds = AVDataset(args.root, sp / "test.txt", args.image_size, False)
    tr = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    va = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    te = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    weights, freq = compute_class_weights(tr_ds, n)
    print("class pixel fraction / weight:")
    for i, c in enumerate(CLASSES):
        print(f"  {c:<14}{freq[i]*100:6.2f}%   w={weights[i]:.2f}")

    model = build_model(args.pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(weight=weights.to(device), ignore_index=IGNORE)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    best = -1.0; best_ep = 0; bad = 0; log = []
    best_path = args.out.parent / "av_segmentation_best.pt"
    for ep in range(1, args.epochs + 1):
        model.train(); tot = 0.0
        for x, y in tqdm(tr, desc=f"epoch {ep:03d}"):
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x.to(device)), y.to(device))
            loss.backward(); opt.step(); tot += float(loss.detach())
        trl = tot / max(1, len(tr))
        m = evaluate(model, va, device, n); print_table(ep, trl, m)
        ckpt = {"model": model.state_dict(), "classes": CLASSES, "encoder_name": "resnet18",
                "image_size": args.image_size, "epoch": ep, "mean_iou": m["mean_iou"]}
        torch.save(ckpt, args.out)
        if m["mean_iou"] > best:
            best = m["mean_iou"]; best_ep = ep; bad = 0; torch.save(ckpt, best_path)
        else:
            bad += 1
        log.append(jr({"epoch": ep, "train_loss": trl, "val": m, "best_epoch": best_ep,
                       "best_mean_iou": best}))
        (args.out.parent / "training_log_av.json").write_text(json.dumps(log, indent=2))
        if args.early_stop and bad >= args.early_stop:
            print(f"early stop at {ep}, best ep {best_ep} mIoU {best:.4f}"); break

    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device)["model"])
    tm = evaluate(model, te, device, n); print_table(best_ep, float("nan"), tm)
    (args.out.parent / "test_metrics_av.json").write_text(
        json.dumps(jr({"best_epoch": best_ep, "classes": CLASSES, "test": tm}), indent=2))
    print(f"held_out_test_mIoU={tm['mean_iou']:.4f}")


if __name__ == "__main__":
    main()
