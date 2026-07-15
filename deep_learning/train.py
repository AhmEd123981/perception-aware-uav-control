"""Train a U-Net crop semantic segmentation model.

The model is intentionally ONNX-friendly and CPU-deployable for MATLAB
simulation integration. Training can use ImageNet encoder weights when they are
available, but the default is fully offline.
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


CLASSES = ["soil", "healthy_crop", "diseased_crop", "weed_patch", "water_stress"]


def build_model(pretrained: bool = False) -> torch.nn.Module:
    """Build the thesis baseline segmentation model."""
    encoder_weights = "imagenet" if pretrained else None
    return smp.Unet(
        encoder_name="resnet18",
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=len(CLASSES),
    )


class CropSegDataset(Dataset):
    """PNG image/mask dataset with zero-based or MATLAB one-based masks."""

    def __init__(
        self,
        root: Path,
        split_file: Path,
        image_size: int = 512,
        train: bool = True,
    ) -> None:
        self.root = root
        self.split_file = split_file
        self.items = [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
        self.image_size = image_size
        self.transform = self._build_transform(train)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        stem = self.items[idx]
        image_path = self.root / "images" / f"{stem}.png"
        mask_path = self.root / "masks" / f"{stem}.png"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise FileNotFoundError(f"Missing image or mask for split item '{stem}'")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformed = self.transform(image=image, mask=mask)
        image = transformed["image"].astype(np.float32) / 255.0
        mask = transformed["mask"].astype(np.int64)
        if mask.min() >= 1:
            mask = mask - 1
        mask = np.clip(mask, 0, len(CLASSES) - 1)

        image_tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask))
        return image_tensor, mask_tensor

    def _build_transform(self, train: bool) -> A.Compose:
        transforms: list[Any] = [
            A.Resize(self.image_size, self.image_size, interpolation=cv2.INTER_LINEAR),
        ]
        if train:
            transforms.extend(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.2),
                    A.RandomRotate90(p=0.5),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.25,
                        contrast_limit=0.25,
                        p=0.6,
                    ),
                    A.MotionBlur(blur_limit=5, p=0.2),
                    A.Affine(
                        scale=(0.95, 1.05),
                        translate_percent=(-0.04, 0.04),
                        rotate=(-12, 12),
                        interpolation=cv2.INTER_LINEAR,
                        mask_interpolation=cv2.INTER_NEAREST,
                        fill=0,
                        fill_mask=0,
                        p=0.35,
                    ),
                ]
            )
        return A.Compose(transforms)


def update_confusion_matrix(
    confusion: np.ndarray,
    targets: torch.Tensor,
    predictions: torch.Tensor,
    num_classes: int,
) -> np.ndarray:
    """Accumulate a true-label by predicted-label confusion matrix."""
    target_np = targets.detach().cpu().numpy().astype(np.int64).ravel()
    pred_np = predictions.detach().cpu().numpy().astype(np.int64).ravel()
    valid = (target_np >= 0) & (target_np < num_classes)
    encoded = num_classes * target_np[valid] + pred_np[valid]
    counts = np.bincount(encoded, minlength=num_classes * num_classes)
    return confusion + counts.reshape(num_classes, num_classes)


def metrics_from_confusion(confusion: np.ndarray) -> dict[str, Any]:
    """Compute IoU, precision, recall, and mean IoU from a confusion matrix."""
    tp = np.diag(confusion).astype(np.float64)
    true_count = confusion.sum(axis=1).astype(np.float64)
    pred_count = confusion.sum(axis=0).astype(np.float64)
    union = true_count + pred_count - tp

    iou = np.divide(tp, union, out=np.full_like(tp, np.nan), where=union > 0)
    precision = np.divide(tp, pred_count, out=np.full_like(tp, np.nan), where=pred_count > 0)
    recall = np.divide(tp, true_count, out=np.full_like(tp, np.nan), where=true_count > 0)
    mean_iou = float(np.nanmean(iou)) if np.any(np.isfinite(iou)) else 0.0

    return {
        "per_class_iou": iou.tolist(),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "mean_iou": mean_iou,
        "confusion_matrix": confusion.astype(int).tolist(),
    }


def print_metrics_table(epoch: int, train_loss: float, metrics: dict[str, Any]) -> None:
    """Print a compact validation table after each epoch."""
    print(f"\nEpoch {epoch:03d} | train_loss={train_loss:.4f} | val_mIoU={metrics['mean_iou']:.4f}")
    print("-" * 76)
    print(f"{'class':<18} {'IoU':>10} {'precision':>12} {'recall':>12}")
    print("-" * 76)
    for idx, name in enumerate(CLASSES):
        iou = format_metric(metrics["per_class_iou"][idx])
        precision = format_metric(metrics["per_class_precision"][idx])
        recall = format_metric(metrics["per_class_recall"][idx])
        print(f"{name:<18} {iou:>10} {precision:>12} {recall:>12}")
    print("-" * 76)


def format_metric(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.4f}"


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> dict[str, Any]:
    """Evaluate segmentation metrics on the validation split."""
    model.eval()
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="val", leave=False):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            logits = model(images)
            predictions = torch.argmax(logits, dim=1)
            confusion = update_confusion_matrix(confusion, masks, predictions, num_classes)
    return metrics_from_confusion(confusion)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    epoch: int,
    mean_iou: float,
    args: argparse.Namespace,
) -> None:
    """Save a reproducible model checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "classes": CLASSES,
            "architecture": "smp.Unet",
            "encoder_name": "resnet18",
            "encoder_weights": "imagenet" if args.pretrained else None,
            "image_size": args.image_size,
            "epoch": epoch,
            "mean_iou": mean_iou,
            "seed": args.seed,
        },
        path,
    )


def json_ready(value: Any) -> Any:
    """Convert NaN values into JSON nulls for portable logs."""
    if isinstance(value, dict):
        return {k: json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    train_dataset = CropSegDataset(args.dataset_root, args.split_file, args.image_size, train=True)
    val_dataset = CropSegDataset(args.dataset_root, args.val_split_file, args.image_size, train=False)
    test_dataset = CropSegDataset(args.dataset_root, args.test_split_file, args.image_size, train=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = build_model(pretrained=args.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion: nn.Module = nn.CrossEntropyLoss()

    best_miou = -np.inf
    best_epoch = 0
    epochs_without_improvement = 0
    training_log: list[dict[str, Any]] = []
    best_output = args.output.parent / "crop_segmentation_best.pt"
    log_path = args.output.parent / "training_log.json"

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        progress = tqdm(train_loader, desc=f"epoch {epoch:03d} train")
        for images, masks in progress:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), masks)
            loss.backward()
            optimizer.step()
            batch_loss = float(loss.detach().cpu())
            total_loss += batch_loss
            progress.set_postfix(loss=f"{batch_loss:.4f}")

        train_loss = total_loss / max(1, len(train_loader))
        val_metrics = evaluate(model, val_loader, device, len(CLASSES))
        print_metrics_table(epoch, train_loss, val_metrics)

        save_checkpoint(args.output, model, epoch, val_metrics["mean_iou"], args)

        if val_metrics["mean_iou"] > best_miou:
            best_miou = val_metrics["mean_iou"]
            best_epoch = epoch
            epochs_without_improvement = 0
            if args.save_best:
                save_checkpoint(best_output, model, epoch, best_miou, args)
        else:
            epochs_without_improvement += 1

        epoch_log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val": val_metrics,
            "best_epoch": best_epoch,
            "best_mean_iou": None if not np.isfinite(best_miou) else float(best_miou),
        }
        training_log.append(json_ready(epoch_log))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(json.dumps(training_log, indent=2))

        if (
            args.early_stop_patience > 0
            and epoch >= args.min_epochs_before_early_stop
            and epochs_without_improvement >= args.early_stop_patience
        ):
            print(
                f"Early stopping at epoch {epoch}; best epoch {best_epoch} "
                f"with val_mIoU={best_miou:.4f}"
            )
            break

    best_output = args.output.parent / "crop_segmentation_best.pt"
    if best_output.exists():
        checkpoint = torch.load(best_output, map_location=device)
        model.load_state_dict(checkpoint["model"])
        if best_epoch == 0:
            best_epoch = int(checkpoint.get("epoch", 0))
            best_miou = float(checkpoint.get("mean_iou", best_miou))
    test_metrics = evaluate(model, test_loader, device, len(CLASSES))
    print_metrics_table(best_epoch, float("nan"), test_metrics)
    test_path = args.output.parent / "test_metrics.json"
    test_path.write_text(json.dumps(json_ready({"best_epoch": best_epoch, "test": test_metrics}), indent=2))
    print(f"held_out_test_mIoU={test_metrics['mean_iou']:.4f} metrics={test_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/synthetic"))
    parser.add_argument("--split-file", type=Path, default=Path("datasets/splits/train.txt"))
    parser.add_argument("--val-split-file", type=Path, default=Path("datasets/splits/val.txt"))
    parser.add_argument("--test-split-file", type=Path, default=Path("datasets/splits/test.txt"))
    parser.add_argument("--output", type=Path, default=Path("deep_learning/weights/crop_segmentation.pt"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--min-epochs-before-early-stop", type=int, default=30)
    parser.add_argument("--save-best", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
