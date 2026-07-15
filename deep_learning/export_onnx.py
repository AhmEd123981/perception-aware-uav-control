"""Export the PyTorch crop segmentation checkpoint to ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from train import CLASSES, build_model


def export(args: argparse.Namespace) -> None:
    """Export an smp U-Net checkpoint with dynamic batch support."""
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    classes = checkpoint.get("classes", CLASSES)
    if list(classes) != CLASSES:
        raise ValueError(f"Checkpoint classes {classes} do not match expected classes {CLASSES}")

    model = build_model(pretrained=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    dummy = torch.randn(1, 3, args.image_size, args.image_size, dtype=torch.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        args.output,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={
            "image": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print(f"exported={args.output} checkpoint={args.checkpoint} opset={args.opset}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("deep_learning/weights/crop_segmentation_best.pt"),
    )
    parser.add_argument("--output", type=Path, default=Path("deep_learning/weights/crop_segmentation.onnx"))
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


if __name__ == "__main__":
    export(parse_args())
