"""Resumable CPU trainer for the Agriculture-Vision model.

Long CPU runs get killed when the session restarts, so this trainer saves a
FULL checkpoint (model + optimizer + epoch + best state + log) every epoch and
resumes from it on the next launch. Reuses the dataset/model/eval code from
train_av.py; only adds resilience. Safe to relaunch any number of times.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_av as T


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(r"C:\Users\Ahmed\av_work\av_subset_2020"))
    ap.add_argument("--tag", type=str, default="2020")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--early-stop", type=int, default=8)
    ap.add_argument("--warm-start", type=Path, default=None,
                    help="model-only checkpoint to initialise weights on a fresh run")
    args = ap.parse_args()

    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cpu")
    n = len(T.CLASSES)
    wdir = Path("deep_learning/weights")
    ckpt_path = wdir / f"av_{args.tag}_resume.pt"      # full resumable state
    best_path = wdir / f"av_segmentation_{args.tag}_best.pt"
    log_path = wdir / f"training_log_{args.tag}.json"
    wdir.mkdir(parents=True, exist_ok=True)

    sp = args.root / "splits"
    tr_ds = T.AVDataset(args.root, sp / "train.txt", args.image_size, True)
    va_ds = T.AVDataset(args.root, sp / "val.txt", args.image_size, False)
    te_ds = T.AVDataset(args.root, sp / "test.txt", args.image_size, False)
    tr = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    va = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    te = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    weights, freq = T.compute_class_weights(tr_ds, n)
    print("class pixel fraction / weight:", flush=True)
    for i, c in enumerate(T.CLASSES):
        print(f"  {c:<14}{freq[i]*100:6.2f}%   w={weights[i]:.2f}", flush=True)

    model = T.build_model(args.pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(weight=weights.to(device), ignore_index=T.IGNORE)

    start_ep, best, best_ep, bad, log = 1, -1.0, 0, 0, []
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start_ep = ck["epoch"] + 1; best = ck["best"]; best_ep = ck["best_ep"]
        bad = ck["bad"]; log = ck["log"]
        print(f"RESUMED from epoch {ck['epoch']} (best {best:.4f} @ ep{best_ep})", flush=True)
    elif args.warm_start and args.warm_start.exists():
        w = torch.load(args.warm_start, map_location=device)
        model.load_state_dict(w["model"])
        print(f"warm-started weights from {args.warm_start.name}", flush=True)

    for ep in range(start_ep, args.epochs + 1):
        model.train(); tot = 0.0
        for x, y in tqdm(tr, desc=f"epoch {ep:03d}"):
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x.to(device)), y.to(device))
            loss.backward(); opt.step(); tot += float(loss.detach())
        trl = tot / max(1, len(tr))
        m = T.evaluate(model, va, device, n); T.print_table(ep, trl, m)

        if m["mean_iou"] > best:
            best, best_ep, bad = m["mean_iou"], ep, 0
            torch.save({"model": model.state_dict(), "classes": T.CLASSES,
                        "encoder_name": "resnet18", "image_size": args.image_size,
                        "epoch": ep, "mean_iou": best}, best_path)
        else:
            bad += 1
        log.append(T.jr({"epoch": ep, "train_loss": trl, "val": m,
                         "best_epoch": best_ep, "best_mean_iou": best}))
        log_path.write_text(json.dumps(log, indent=2))
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": ep,
                    "best": best, "best_ep": best_ep, "bad": bad, "log": log}, ckpt_path)
        if args.early_stop and bad >= args.early_stop:
            print(f"early stop at {ep}; best {best:.4f} @ ep{best_ep}", flush=True)
            break

    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device)["model"])
    tm = T.evaluate(model, te, device, n); T.print_table(best_ep, float("nan"), tm)
    (wdir / f"test_metrics_{args.tag}.json").write_text(
        json.dumps(T.jr({"best_epoch": best_ep, "classes": T.CLASSES, "test": tm}), indent=2))
    print(f"held_out_test_mIoU={tm['mean_iou']:.4f}", flush=True)
    print("DONE_TRAIN", flush=True)


if __name__ == "__main__":
    main()
