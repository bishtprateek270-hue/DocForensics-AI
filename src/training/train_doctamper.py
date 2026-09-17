"""DocTamper Fine-Tuning and Training Script for DocForensics AI.

Trains or fine-tunes DualStreamForensicNet on DocTamper dataset (LMDB or extracted folders).
Preserves the locked baseline checkpoint and outputs to dedicated checkpoints.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.dual_stream_forensics import DualStreamForensicNet
from src.preprocessing.doctamper_loader import create_doctamper_dataloaders

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("DocTamperTrainer")


class DiceLoss(nn.Module):
    """Soft Dice Loss for binary segmentation."""

    def __init__(self, smooth: float = 1e-6) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        cardinality = probs_flat.sum() + targets_flat.sum()
        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice_score


class BCEDiceLoss(nn.Module):
    """Combined BCE and Dice Loss."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self.bce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return self.bce_weight * bce + self.dice_weight * dice


def compute_metrics(
    logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5
) -> Dict[str, float]:
    """Compute binary segmentation metrics (Dice, IoU, Precision, Recall)."""
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()
        targets = (targets >= 0.5).float()

        preds_flat = preds.view(-1)
        targets_flat = targets.view(-1)

        tp = (preds_flat * targets_flat).sum().item()
        fp = (preds_flat * (1 - targets_flat)).sum().item()
        fn = ((1 - preds_flat) * targets_flat).sum().item()

        dice = (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)
        iou = (tp + 1e-6) / (tp + fp + fn + 1e-6)
        precision = (tp + 1e-6) / (tp + fp + 1e-6)
        recall = (tp + 1e-6) / (tp + fn + 1e-6)

        return {
            "dice": dice,
            "iou": iou,
            "precision": precision,
            "recall": recall,
        }


def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    metrics_sum = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}
    num_batches = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(images)
            loss = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        batch_metrics = compute_metrics(logits, masks)
        total_loss += loss.item()
        for k in metrics_sum:
            metrics_sum[k] += batch_metrics[k]
        num_batches += 1

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "dice": f"{batch_metrics['dice']:.4f}"})

    avg_loss = total_loss / max(1, num_batches)
    avg_metrics = {k: v / max(1, num_batches) for k, v in metrics_sum.items()}
    avg_metrics["loss"] = avg_loss
    return avg_metrics


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0.0
    metrics_sum = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}
    num_batches = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(images)
            loss = criterion(logits, masks)

        batch_metrics = compute_metrics(logits, masks)
        total_loss += loss.item()
        for k in metrics_sum:
            metrics_sum[k] += batch_metrics[k]
        num_batches += 1

    avg_loss = total_loss / max(1, num_batches)
    avg_metrics = {k: v / max(1, num_batches) for k, v in metrics_sum.items()}
    avg_metrics["loss"] = avg_loss
    return avg_metrics


def run_training(
    data_path: str,
    epochs: int = 15,
    batch_size: int = 4,
    lr: float = 1e-4,
    pretrained_path: Optional[str] = "checkpoints/dual_stream_best.pth",
    output_checkpoint: str = "checkpoints/dual_stream_doctamper_best.pth",
    val_split: float = 0.15,
    max_samples: Optional[int] = None,
    num_workers: int = 0,
    device_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute complete DocTamper training loop."""
    device = torch.device(
        device_name if device_name else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info("Using device: %s (%s)", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU")

    # Ensure output dir exists
    out_path = Path(output_checkpoint)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    last_checkpoint_path = out_path.parent / f"{out_path.stem}_last{out_path.suffix}"

    # Load DataLoaders
    logger.info("Initializing DocTamper DataLoaders from %s...", data_path)
    train_loader, val_loader = create_doctamper_dataloaders(
        data_path=data_path,
        batch_size=batch_size,
        val_split=val_split,
        target_size=(512, 512),
        num_workers=num_workers,
        max_samples=max_samples,
    )
    logger.info("Train batches: %d, Val batches: %d", len(train_loader), len(val_loader) if val_loader else 0)

    # Initialize Model
    model = DualStreamForensicNet(pretrained_backbone=True)

    if pretrained_path and Path(pretrained_path).exists():
        logger.info("Loading pre-trained baseline weights from: %s", pretrained_path)
        checkpoint = torch.load(pretrained_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)
        logger.info("Baseline weights successfully transferred.")
    else:
        logger.info("Training from scratch / ImageNet backbone initialization.")

    model.to(device)

    # Criterion, Optimizer, Scheduler, Scaler
    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_dice = 0.0
    history = []

    logger.info("Starting DocTamper training for %d epochs...", epochs)
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, scaler, device)

        if val_loader is not None:
            val_metrics = evaluate_epoch(model, val_loader, criterion, device)
        else:
            val_metrics = train_metrics

        scheduler.step()
        epoch_duration = time.time() - epoch_start

        logger.info(
            "Epoch [%d/%d] (%.1fs) | Train Loss: %.4f, Dice: %.4f, IoU: %.4f | Val Loss: %.4f, Dice: %.4f, IoU: %.4f, Prec: %.4f, Rec: %.4f",
            epoch,
            epochs,
            epoch_duration,
            train_metrics["loss"],
            train_metrics["dice"],
            train_metrics["iou"],
            val_metrics["loss"],
            val_metrics["dice"],
            val_metrics["iou"],
            val_metrics["precision"],
            val_metrics["recall"],
        )

        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "lr": optimizer.param_groups[0]["lr"],
            "duration": epoch_duration,
        }
        history.append(record)

        # Save latest checkpoint
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": val_metrics["dice"],
                "val_iou": val_metrics["iou"],
                "history": history,
            },
            str(last_checkpoint_path),
        )

        # Save best checkpoint
        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            logger.info("★ New best validation Dice: %.4f. Saving to %s", best_val_dice, output_checkpoint)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_dice": val_metrics["dice"],
                    "val_iou": val_metrics["iou"],
                    "val_precision": val_metrics["precision"],
                    "val_recall": val_metrics["recall"],
                    "history": history,
                },
                str(out_path),
            )

    total_time = time.time() - start_time
    logger.info("Training complete in %.1f seconds. Best Val Dice: %.4f", total_time, best_val_dice)

    # Save summary report
    summary_path = PROJECT_ROOT / "reports" / "doctamper_training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_data = {
        "dataset_path": str(data_path),
        "total_epochs": epochs,
        "batch_size": batch_size,
        "initial_lr": lr,
        "pretrained_source": str(pretrained_path),
        "best_checkpoint": str(output_checkpoint),
        "best_val_dice": best_val_dice,
        "total_duration_sec": total_time,
        "history": history,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    return summary_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DocForensics AI on DocTamper Dataset")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/raw/doctamper",
        help="Path to DocTamper dataset (LMDB directory or extracted image folder)",
    )
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--pretrained",
        type=str,
        default="checkpoints/dual_stream_best.pth",
        help="Baseline checkpoint to initialize from",
    )
    parser.add_argument(
        "--output_checkpoint",
        type=str,
        default="checkpoints/dual_stream_doctamper_best.pth",
        help="Target output checkpoint path",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit number of dataset samples (useful for fast verification or subset fine-tuning)",
    )
    parser.add_argument("--val_split", type=float, default=0.15, help="Validation fraction")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu')")

    args = parser.parse_args()

    run_training(
        data_path=args.data_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        pretrained_path=args.pretrained,
        output_checkpoint=args.output_checkpoint,
        val_split=args.val_split,
        max_samples=args.max_samples,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
