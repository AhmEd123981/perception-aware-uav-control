"""Run ONNX semantic segmentation inference on one UAV frame.

The script is intentionally CLI-friendly so MATLAB can call it with system()
when the ONNX Runtime bridge is selected. It writes a class-id mask PNG and an
optional compressed probability tensor for mapping.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


# Real Agriculture-Vision CVPR-2020 4-class scheme (0-based). Affected classes to
# map/spray are drydown/weed_cluster/double_plant; MATLAB adds 1 -> ids [2 3 4].
CLASSES = ["background", "drydown", "weed_cluster", "double_plant"]


def preprocess(image_bgr: np.ndarray, size: int) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = image_bgr.shape[:2]
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    tensor = resized.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
    return tensor, (h, w)


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def infer(args: argparse.Namespace) -> None:
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(args.image)

    tensor, original_hw = preprocess(image, args.image_size)
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: tensor})[0]
    probs = softmax(logits)[0]
    mask = probs.argmax(axis=0).astype(np.uint8)
    mask = cv2.resize(mask, (original_hw[1], original_hw[0]), interpolation=cv2.INTER_NEAREST)

    args.output_mask.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output_mask), mask)

    if args.output_probs:
        resized_probs = np.stack([
            cv2.resize(p, (original_hw[1], original_hw[0]), interpolation=cv2.INTER_LINEAR)
            for p in probs
        ])
        np.savez_compressed(args.output_probs, probs=resized_probs.astype(np.float16), classes=CLASSES)

    meta = {
        "image": str(args.image),
        "model": str(args.model),
        "mask": str(args.output_mask),
        "classes": CLASSES,
        "image_size": args.image_size,
    }
    args.output_mask.with_suffix(".json").write_text(json.dumps(meta, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("deep_learning/weights/av_segmentation_2020.onnx"))
    parser.add_argument("--output-mask", type=Path, required=True)
    parser.add_argument("--output-probs", type=Path)
    parser.add_argument("--image-size", type=int, default=512)
    return parser.parse_args()


if __name__ == "__main__":
    infer(parse_args())
