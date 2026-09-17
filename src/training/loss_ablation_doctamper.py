"""Controlled Loss Ablation Study for Tiny-Text Document Tampering.

Compares:
A. BCE + Dice
B. Focal + Dice
C. Tversky (alpha=0.3, beta=0.7 -> High FN penalty)
D. Focal Tversky (alpha=0.3, beta=0.7, gamma=1.33)

Evaluates on identical training/validation splits with region-level and tiny-text metrics.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.region_metrics import evaluate_batch_region_metrics
from src.models.dual_stream_forensics import DualStreamForensicNet
from src.preprocessing.doctamper_patch_loader import create_patch_dataloaders
from src.training.losses import (
    BCEDiceLoss,
    FocalDiceLoss,
    FocalTverskyLoss,
    TverskyLoss,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LossAblation")


def evaluate_model_full_metrics(
    model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device
) -> Dict[str, Any]:
    """Run validation evaluation with region-level and area-stratified tracking."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = model(imgs)

            probs = torch.sigmoid(logits)
            all_preds.append(probs)
            all_targets.append(masks)

    cat_preds = torch.cat(all_preds, dim=0)
    cat_targets = torch.cat(all_targets, dim=0)

    metrics = evaluate_batch_region_metrics(cat_preds, cat_targets, threshold=0.5)
    return metrics


def run_single_loss_experiment(
    loss_name: str,
    criterion: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    checkpoint_init: str,
    epochs: int = 2,
    lr: float = 1e-4,
    device: torch.device = torch.device("cuda"),
) -> Dict[str, Any]:
    """Train for a controlled 2-epoch run and evaluate."""
    logger.info("=== Running Loss Ablation for [%s] ===", loss_name)

    # Initialize model from doctamper baseline
    model = DualStreamForensicNet(pretrained_backbone=False)
    ckpt = torch.load(checkpoint_init, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    history = []
    start_t = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        batches = 0

        for batch in tqdm(train_loader, desc=f"{loss_name} Ep {epoch}", leave=False):
            imgs = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = model(imgs)
                loss = criterion(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            batches += 1

        val_metrics = evaluate_model_full_metrics(model, val_loader, device)
        avg_train_loss = train_loss / max(1, batches)

        logger.info(
            "[%s] Ep %d | Train Loss: %.4f | Val Dice: %.4f, Recall: %.4f, RegRecall: %.4f, TinyRecall: %.4f, Prec: %.4f",
            loss_name,
            epoch,
            avg_train_loss,
            val_metrics["pixel_dice"],
            val_metrics["pixel_recall"],
            val_metrics["region_recall"],
            val_metrics["area_stratified"]["<0.5%"]["region_recall"],
            val_metrics["pixel_precision"],
        )

        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "metrics": val_metrics,
        })

    total_duration = time.time() - start_t
    final_metrics = history[-1]["metrics"]

    return {
        "loss_name": loss_name,
        "duration_sec": round(total_duration, 1),
        "final_pixel_dice": final_metrics["pixel_dice"],
        "final_pixel_iou": final_metrics["pixel_iou"],
        "final_pixel_precision": final_metrics["pixel_precision"],
        "final_pixel_recall": final_metrics["pixel_recall"],
        "final_region_recall": final_metrics["region_recall"],
        "final_region_precision": final_metrics["region_precision"],
        "final_fp_regions_per_doc": final_metrics["fp_regions_per_doc"],
        "tiny_region_recall": final_metrics["area_stratified"]["<0.5%"]["region_recall"],
        "tiny_region_dice": final_metrics["area_stratified"]["<0.5%"]["mean_dice"],
        "history": history,
    }


def run_loss_ablation_study(
    data_path: str = "data/raw/doctamper/DocTamper Training",
    checkpoint_init: str = "checkpoints/dual_stream_doctamper_best.pth",
    max_samples: int = 1200,
    epochs: int = 2,
    batch_size: int = 8,
) -> Dict[str, Any]:
    """Execute complete controlled loss ablation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Initializing identical ablation splits (max %d samples)...", max_samples)

    train_loader, val_loader = create_patch_dataloaders(
        data_path=data_path,
        batch_size=batch_size,
        val_split=0.20,
        target_size=(512, 512),
        crop_size=256,
        seed=42,
        max_samples=max_samples,
    )

    candidates = {
        "A_BCE_Dice": BCEDiceLoss(bce_weight=0.5, dice_weight=0.5),
        "B_Focal_Dice": FocalDiceLoss(focal_weight=0.5, dice_weight=0.5, gamma=2.0),
        "C_Tversky_FN_Favored": TverskyLoss(alpha=0.3, beta=0.7),
        "D_Focal_Tversky": FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=1.33),
    }

    results = {}
    for name, criterion in candidates.items():
        res = run_single_loss_experiment(
            loss_name=name,
            criterion=criterion,
            train_loader=train_loader,
            val_loader=val_loader,
            checkpoint_init=checkpoint_init,
            epochs=epochs,
            lr=1e-4,
            device=device,
        )
        results[name] = res

    # Select best loss based on composite score: Region Recall (40%) + Tiny Recall (30%) + Dice (30%)
    scores = {}
    for name, res in results.items():
        score = (
            0.40 * res["final_region_recall"]
            + 0.30 * res["tiny_region_recall"]
            + 0.30 * res["final_pixel_dice"]
        )
        scores[name] = score

    best_loss = max(scores, key=scores.get)
    logger.info("★ Loss Ablation Winner: [%s] (Composite Score: %.4f)", best_loss, scores[best_loss])

    output = {
        "ablation_results": results,
        "composite_scores": scores,
        "selected_loss": best_loss,
    }

    report_path = PROJECT_ROOT / "reports" / "doctamper_loss_ablation_summary.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    return output


if __name__ == "__main__":
    run_loss_ablation_study()
