"""Run the Day 3 deep-learning pipeline for the UAV perception extension.

The script performs a reproducible CPU/GPU-safe pass through:
1. Synthetic dataset availability check/generation.
2. U-Net training using segmentation_models_pytorch.
3. ONNX export.
4. ONNX Runtime inference on a MATLAB Day 2 camera frame.
5. Summary JSON creation for thesis traceability.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_command(command: list[str]) -> float:
    """Run a subprocess in the project root and return elapsed seconds."""
    print("\n" + " ".join(str(part) for part in command), flush=True)
    start = time.perf_counter()
    subprocess.run(command, cwd=ROOT, check=True)
    return time.perf_counter() - start


def count_files(path: Path, pattern: str) -> int:
    return sum(1 for _ in path.glob(pattern))


def ensure_dataset(count: int, seed: int) -> dict[str, int]:
    """Generate the synthetic dataset if the expected files are missing."""
    image_dir = ROOT / "datasets" / "synthetic" / "images"
    mask_dir = ROOT / "datasets" / "synthetic" / "masks"
    train_split = ROOT / "datasets" / "splits" / "train.txt"
    val_split = ROOT / "datasets" / "splits" / "val.txt"
    test_split = ROOT / "datasets" / "splits" / "test.txt"

    image_count = count_files(image_dir, "*.png")
    mask_count = count_files(mask_dir, "*.png")
    splits_exist = train_split.exists() and val_split.exists() and test_split.exists()

    if image_count < count or mask_count < count or not splits_exist:
        run_command(
            [
                sys.executable,
                "deep_learning/generate_synthetic_dataset.py",
                "--count",
                str(count),
                "--seed",
                str(seed),
            ]
        )

    return {
        "images": count_files(image_dir, "*.png"),
        "masks": count_files(mask_dir, "*.png"),
        "train": len(train_split.read_text().splitlines()) if train_split.exists() else 0,
        "val": len(val_split.read_text().splitlines()) if val_split.exists() else 0,
        "test": len(test_split.read_text().splitlines()) if test_split.exists() else 0,
    }


def choose_inference_frame() -> Path:
    """Prefer a MATLAB Day 2 frame; fall back to the synthetic validation split."""
    day2_frame = ROOT / "results" / "perception_logs" / "day2_lawnmower_PID" / "frames" / "frame_000050.png"
    if day2_frame.exists():
        return day2_frame

    val_split = ROOT / "datasets" / "splits" / "val.txt"
    if not val_split.exists():
        raise FileNotFoundError("No Day 2 frame and no validation split were found.")
    stem = next(line.strip() for line in val_split.read_text().splitlines() if line.strip())
    return ROOT / "datasets" / "synthetic" / "images" / f"{stem}.png"


def main(args: argparse.Namespace) -> None:
    output_root = ROOT / "results" / "perception_logs" / "day3_deep_learning"
    output_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "pretrained": args.pretrained,
        "python": sys.executable,
    }

    dataset_counts = ensure_dataset(args.dataset_count, args.seed)
    summary["dataset_counts"] = dataset_counts

    train_cmd = [
        sys.executable,
        "deep_learning/train.py",
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--image-size",
        str(args.image_size),
        "--early-stop-patience",
        str(args.early_stop_patience),
    ]
    if args.cpu:
        train_cmd.append("--cpu")
    if args.pretrained:
        train_cmd.append("--pretrained")
    summary["train_seconds"] = run_command(train_cmd)

    checkpoint = ROOT / "deep_learning" / "weights" / "crop_segmentation_best.pt"
    onnx_path = ROOT / "deep_learning" / "weights" / "crop_segmentation.onnx"
    export_cmd = [
        sys.executable,
        "deep_learning/export_onnx.py",
        "--opset",
        str(args.opset),
        "--checkpoint",
        str(checkpoint.relative_to(ROOT)),
        "--image-size",
        str(args.image_size),
    ]
    summary["export_seconds"] = run_command(export_cmd)

    inference_frame = choose_inference_frame()
    mask_path = output_root / f"{inference_frame.stem}_mask.png"
    infer_cmd = [
        sys.executable,
        "deep_learning/infer.py",
        "--image",
        str(inference_frame.relative_to(ROOT)),
        "--model",
        str(onnx_path.relative_to(ROOT)),
        "--output-mask",
        str(mask_path.relative_to(ROOT)),
        "--image-size",
        str(args.image_size),
    ]
    summary["inference_seconds"] = run_command(infer_cmd)
    summary["checkpoint"] = str(checkpoint.relative_to(ROOT))
    summary["onnx"] = str(onnx_path.relative_to(ROOT))
    summary["inference_frame"] = str(inference_frame.relative_to(ROOT))
    summary["inference_mask"] = str(mask_path.relative_to(ROOT))
    summary["training_log"] = "deep_learning/weights/training_log.json"

    summary_path = output_root / "day3_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nDay 3 deep-learning pipeline complete. Summary: {summary_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
