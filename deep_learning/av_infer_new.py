"""Deploy step: give the trained model a NEW aerial photo -> it tells the problem.

This is what runs when the drone captures a fresh frame in flight. It loads the
ONNX model (the real deployment runtime, same as the drone perception loop),
runs it on ANY image, and reports which problems are present and how much of the
field each covers -- plus a colored overlay showing WHERE.

Usage:
    python deep_learning/av_infer_new.py --image path/to/new_photo.jpg
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

CLASSES = ["healthy", "drydown", "weed_cluster", "double_plant"]
DESC = {"healthy": "healthy field / soil", "drydown": "dry or stressed crop",
        "weed_cluster": "weed infestation", "double_plant": "double-seeded rows"}
# BGR colors for the overlay
COLORS = {1: (0, 159, 230), 2: (30, 30, 213), 3: (200, 40, 170)}


def infer(session: ort.InferenceSession, image_bgr: np.ndarray, size: int = 256) -> np.ndarray:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    x = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]  # 1x3xHxW
    logits = session.run(None, {session.get_inputs()[0].name: x})[0]
    return logits[0].argmax(0).astype(np.uint8)  # HxW class ids


def report(pred: np.ndarray) -> dict:
    total = pred.size
    pct = {CLASSES[c]: round(100.0 * int((pred == c).sum()) / total, 2) for c in range(len(CLASSES))}
    problems = {k: v for k, v in pct.items() if k != "healthy" and v >= 1.0}
    main = max(problems, key=problems.get) if problems else None
    return {"coverage_percent": pct, "problems": problems, "main_problem": main}


def overlay(image_bgr: np.ndarray, pred: np.ndarray, size: int = 256, alpha: float = 0.5) -> np.ndarray:
    base = cv2.resize(image_bgr, (size, size))
    out = base.copy()
    for cid, col in COLORS.items():
        m = pred == cid
        out[m] = (alpha * np.array(col) + (1 - alpha) * base[m]).astype(np.uint8)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, required=True, help="a new aerial photo")
    ap.add_argument("--model", type=Path,
                    default=Path("deep_learning/weights/av_segmentation_2020.onnx"))
    ap.add_argument("--out", type=Path, default=Path("results/analysis/new_frame_report"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(args.image))
    if img is None:
        raise FileNotFoundError(args.image)
    sess = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    pred = infer(sess, img)
    rep = report(pred)

    stem = args.image.stem
    cv2.imwrite(str(args.out / f"{stem}_overlay.png"), overlay(img, pred))
    (args.out / f"{stem}_report.json").write_text(json.dumps(rep, indent=2))

    print(f"\n=== FIELD PROBLEM REPORT: {args.image.name} ===")
    for name in CLASSES:
        print(f"  {name:<14} {rep['coverage_percent'][name]:6.2f}%   ({DESC[name]})")
    if rep["main_problem"]:
        mp = rep["main_problem"]
        print(f"\n  --> PROBLEM DETECTED: mainly '{mp}' ({DESC[mp]}), "
              f"{rep['coverage_percent'][mp]:.1f}% of the field.")
        others = [k for k in rep["problems"] if k != mp]
        if others:
            print(f"      also present: {', '.join(others)}")
        print(f"  --> Recommend treatment / revisit of the affected area.")
    else:
        print("\n  --> Field looks healthy (no significant problem area).")
    print(f"\n  overlay + json saved to {args.out}")


if __name__ == "__main__":
    main()
