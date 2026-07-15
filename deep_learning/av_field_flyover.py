"""END-TO-END TEST of step 4: a drone flies over a REAL field and reports problems.

Builds a large field by stitching real (held-out) Agriculture-Vision patches,
simulates a lawnmower survey flight (a moving camera window with overlap), runs
every captured frame through the ONNX model, fuses the results into a whole-field
problem map, then extracts affected zones and a nearest-neighbour revisit route.

This is the full vision: fly -> capture -> detect -> map -> locate problems.

    python deep_learning/av_field_flyover.py
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

# Class 0 is Agriculture-Vision "background" (matches configs/perception_config.m
# and the ONNX worker), not "healthy" — kept consistent across the whole package.
CLASSES = ["background", "drydown", "weed_cluster", "double_plant"]
COLORS = {1: (0, 159, 230), 2: (30, 30, 213), 3: (200, 40, 170)}  # BGR


def build_field(subset: Path, grid: int, rng: random.Random,
                problem_tiles: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Tile grid*grid real test patches into one big field (RGB + GT class map).

    A realistic field is mostly healthy crop with a few localised problem zones
    (Agriculture-Vision's natural balance is ~82% background / ~18% affected).
    We therefore place ``problem_tiles`` of the most problem-rich patches among
    otherwise clean patches, instead of tiling the 16 worst patches (which gave
    an unrealistic ~89%-affected field that saturates affected-zone recall).
    """
    stems = [l.strip() for l in (subset / "splits" / "test.txt").read_text().splitlines() if l.strip()]
    score = {}
    for s in stems:
        m = cv2.imread(str(subset / "masks" / f"{s}.png"), cv2.IMREAD_GRAYSCALE)
        score[s] = int(((m > 0) & (m != 255)).sum())
    ranked = sorted(stems, key=lambda s: score[s], reverse=True)

    n = grid * grid
    if problem_tiles is None:
        problem_tiles = max(1, n // 4)          # ~25% of tiles carry the problems
    problem_tiles = max(0, min(problem_tiles, n))
    top = ranked[:problem_tiles]                 # richest in problems
    clean = [s for s in ranked if score[s] == 0] or ranked[::-1]
    rng.shuffle(clean)
    pick = top + clean[: n - problem_tiles]
    rng.shuffle(pick)

    P = 512
    field = np.zeros((grid * P, grid * P, 3), np.uint8)
    gt = np.zeros((grid * P, grid * P), np.uint8)
    for i, s in enumerate(pick):
        r, c = divmod(i, grid)
        img = cv2.imread(str(subset / "images" / f"{s}.jpg"))
        m = cv2.imread(str(subset / "masks" / f"{s}.png"), cv2.IMREAD_GRAYSCALE)
        m[m == 255] = 0
        field[r*P:(r+1)*P, c*P:(c+1)*P] = cv2.resize(img, (P, P))
        gt[r*P:(r+1)*P, c*P:(c+1)*P] = cv2.resize(m, (P, P), interpolation=cv2.INTER_NEAREST)
    return field, gt


def flyover(field: np.ndarray, sess: ort.InferenceSession, win: int, step: int) -> tuple[np.ndarray, list]:
    """Lawnmower survey: move a camera window, infer each frame, fuse votes."""
    H, W = field.shape[:2]
    votes = np.zeros((H, W, len(CLASSES)), np.float32)
    path = []
    ys = list(range(0, H - win + 1, step)) or [0]
    xs = list(range(0, W - win + 1, step)) or [0]
    for r, y in enumerate(ys):
        row = xs if r % 2 == 0 else xs[::-1]          # boustrophedon (lawnmower)
        for x in row:
            path.append((x + win // 2, y + win // 2))  # camera centre = flight waypoint
            frame = field[y:y+win, x:x+win]
            rgb = cv2.cvtColor(cv2.resize(frame, (256, 256)), cv2.COLOR_BGR2RGB)
            inp = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
            logits = sess.run(None, {sess.get_inputs()[0].name: inp})[0][0]  # 4x256x256
            pred = logits.argmax(0).astype(np.uint8)
            pred = cv2.resize(pred, (win, win), interpolation=cv2.INTER_NEAREST)
            for c in range(len(CLASSES)):
                votes[y:y+win, x:x+win, c] += (pred == c)
    return votes.argmax(2).astype(np.uint8), path


def extract_zones(pred: np.ndarray, min_area: int) -> list[dict]:
    zones = []
    for cid in (1, 2, 3):
        num, lab, stats, cent = cv2.connectedComponentsWithStats((pred == cid).astype(np.uint8), 8)
        for k in range(1, num):
            if stats[k, cv2.CC_STAT_AREA] >= min_area:
                zones.append({"class": CLASSES[cid], "centroid": [int(cent[k][0]), int(cent[k][1])],
                              "area_px": int(stats[k, cv2.CC_STAT_AREA])})
    return zones


def revisit_route(zones: list[dict]) -> list[int]:
    """Nearest-neighbour ordering of zone centroids from the field origin."""
    remaining = list(range(len(zones)))
    order, cur = [], np.array([0, 0])
    while remaining:
        j = min(remaining, key=lambda i: np.hypot(*(np.array(zones[i]["centroid"]) - cur)))
        order.append(j); cur = np.array(zones[j]["centroid"]); remaining.remove(j)
    return order


def overlay(field, pred, alpha=0.45):
    out = field.copy()
    for cid, col in COLORS.items():
        m = pred == cid
        out[m] = (alpha * np.array(col) + (1 - alpha) * field[m]).astype(np.uint8)
    return out


def seg_metrics(pred: np.ndarray, gt: np.ndarray, affected=(1, 2, 3)) -> dict:
    """Quantitative whole-field metrics of the fused prediction against the GT.

    Both ``pred`` and ``gt`` are HxW class-id rasters (av ids 0..3). Reports
    per-class IoU/precision/recall, mean IoU over classes present in the GT,
    overall pixel accuracy, and a binary "any-problem" precision/recall/IoU
    that mirrors the affected-zone recall the MATLAB pipeline scores.
    """
    n = len(CLASSES)
    cm = np.zeros((n, n), np.int64)  # rows = gt, cols = pred
    for t in range(n):
        tmask = gt == t
        if not tmask.any():
            continue
        for p in range(n):
            cm[t, p] = int((tmask & (pred == p)).sum())

    per_iou, per_prec, per_rec = {}, {}, {}
    for c in range(n):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum() - tp)
        fn = int(cm[c, :].sum() - tp)
        per_iou[CLASSES[c]] = round(tp / (tp + fp + fn), 4) if (tp + fp + fn) else None
        per_prec[CLASSES[c]] = round(tp / (tp + fp), 4) if (tp + fp) else None
        per_rec[CLASSES[c]] = round(tp / (tp + fn), 4) if (tp + fn) else None

    present = [c for c in range(n) if (gt == c).any()]
    valid_iou = [per_iou[CLASSES[c]] for c in present if per_iou[CLASSES[c]] is not None]
    miou = round(float(np.mean(valid_iou)), 4) if valid_iou else None

    gt_aff, pred_aff = np.isin(gt, affected), np.isin(pred, affected)
    tp = int((gt_aff & pred_aff).sum())
    fp = int((~gt_aff & pred_aff).sum())
    fn = int((gt_aff & ~pred_aff).sum())
    affected_binary = {
        "precision": round(tp / (tp + fp), 4) if (tp + fp) else None,
        "recall": round(tp / (tp + fn), 4) if (tp + fn) else None,
        "iou": round(tp / (tp + fp + fn), 4) if (tp + fp + fn) else None,
    }
    return {
        "mIoU_classes_present": miou,
        "pixel_accuracy": round(float(np.trace(cm) / cm.sum()), 4) if cm.sum() else None,
        "per_class_iou": per_iou,
        "per_class_precision": per_prec,
        "per_class_recall": per_rec,
        "affected_binary": affected_binary,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", type=Path, default=Path(r"C:\Users\Ahmed\av_work\av_subset_2020"))
    ap.add_argument("--model", type=Path, default=Path("deep_learning/weights/av_segmentation_2020.onnx"))
    ap.add_argument("--out", type=Path, default=Path("results/analysis/flyover"))
    ap.add_argument("--grid", type=int, default=4)      # 4x4 = 16 real patches
    ap.add_argument("--problem-tiles", type=int, default=None,
                    help="how many of the grid^2 tiles are problem-rich patches "
                         "(default grid^2//4 for a realistic sparse field; set = grid^2 "
                         "to reproduce the old dense field)")
    ap.add_argument("--win", type=int, default=512)
    ap.add_argument("--step", type=int, default=384)    # 25% overlap
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--save-field", action="store_true",
                    help="also write datasets/real_field/{field_texture,field_mask}.png "
                         "from the SAME patches, so the MATLAB flight sim has a real GT mask "
                         "aligned to the texture it flies over")
    ap.add_argument("--field-dir", type=Path, default=Path("datasets/real_field"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    field, gt = build_field(args.subset, args.grid, rng, args.problem_tiles)

    if args.save_field:
        # Texture and GT come from the SAME stitched patches, so they are pixel
        # aligned. MATLAB's build_field_model maps this onto the [0,15]x[0,10] m
        # field via imref2d; the raster is square, the map is 15:10, but GT and
        # detections both live in the 60x40 treatment grid so recall stays exact.
        # Mask stores av 0-based ids (0 background, 1 drydown, 2 weed, 3 double);
        # run_perception_loop adds 1 -> matlab affected ids [2 3 4].
        args.field_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.field_dir / "field_texture.png"), field)
        cv2.imwrite(str(args.field_dir / "field_mask.png"), gt)
        aff = int(np.isin(gt, (1, 2, 3)).sum())
        print(f"saved real field texture + GT mask -> {args.field_dir} "
              f"({field.shape[1]}x{field.shape[0]} px, {100*aff/gt.size:.1f}% affected)")

    sess = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    pred, path = flyover(field, sess, args.win, args.step)
    zones = extract_zones(pred, min_area=(field.shape[0] // 40) ** 2)
    order = revisit_route(zones)

    # figures
    cv2.imwrite(str(args.out / "field_rgb.png"), field)
    cv2.imwrite(str(args.out / "field_prediction.png"), overlay(field, pred))
    cv2.imwrite(str(args.out / "field_groundtruth.png"), overlay(field, gt))
    route_img = overlay(field, pred)
    for a, b in zip(order, order[1:]):
        cv2.line(route_img, tuple(zones[a]["centroid"]), tuple(zones[b]["centroid"]), (0, 255, 255), 4)
    for rank, i in enumerate(order):
        cv2.circle(route_img, tuple(zones[i]["centroid"]), 10, (0, 255, 255), -1)
        cv2.putText(route_img, str(rank+1), tuple(zones[i]["centroid"]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
    cv2.imwrite(str(args.out / "revisit_route.png"), route_img)

    total = pred.size
    cov = {CLASSES[c]: round(100*int((pred==c).sum())/total, 2) for c in range(len(CLASSES))}
    metrics = seg_metrics(pred, gt)
    rep = {"field_patches": args.grid**2, "frames_captured": len(path),
           "coverage_percent_by_class": cov, "num_affected_zones": len(zones),
           "real_field_metrics_vs_gt": metrics,
           "revisit_waypoints": [zones[i]["centroid"] for i in order],
           "zones": [zones[i] for i in order]}
    (args.out / "flyover_report.json").write_text(json.dumps(rep, indent=2))

    print("\n===== DRONE FLYOVER TEST (real field) =====")
    print(f"field: {args.grid}x{args.grid} real patches | frames flown: {len(path)}")
    print("detected coverage:", {k: f"{v}%" for k, v in cov.items() if k != 'background'})
    print(f"real-field vs GT: mIoU={metrics['mIoU_classes_present']} "
          f"pixel_acc={metrics['pixel_accuracy']} "
          f"affected(P/R/IoU)={metrics['affected_binary']['precision']}/"
          f"{metrics['affected_binary']['recall']}/{metrics['affected_binary']['iou']}")
    print(f"affected zones found: {len(zones)}")
    print(f"revisit route (order of centroids): {[zones[i]['centroid'] for i in order][:8]}{' ...' if len(order)>8 else ''}")
    print(f"figures + report -> {args.out}")


if __name__ == "__main__":
    main()
