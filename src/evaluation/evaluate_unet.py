"""
DocForensics AI — U-Net Test Benchmark Evaluator (Phase 5)
Loads best checkpoint and performs thorough evaluation on the untouched test set (122 samples).
Computes:
- Overall Test Loss, Dice Score, IoU, Precision, Recall
- Per-category metric breakdown
- Failure case analysis
Saves full report to reports/unet_test_metrics.json.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import BASE_DIR, CHECKPOINT_DIR, REPORTS_DIR, METADATA_CSV, cfg
from src.models.unet import UNet, get_unet_model
from src.preprocessing.dataset import DocForensicsDataset
from src.training.losses import BCEDiceLoss
from src.evaluation.metrics import compute_batch_metrics, MetricTracker


def evaluate_model_on_test_set(
    checkpoint_path: Path = CHECKPOINT_DIR / "unet_best.pth",
    metadata_path: Path = METADATA_CSV,
    batch_size: int = 8,
    threshold: float = 0.5,
    device: torch.device = None,
) -> Dict[str, Any]:
    """
    Evaluates trained U-Net checkpoint on untouched test set.
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    print("=" * 75)
    print(f"  DocForensics AI — Evaluating Best U-Net Checkpoint on Test Set  ")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Device    : {device}")
    print("=" * 75)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # 1. Load Dataset
    test_dataset = DocForensicsDataset(
        metadata_path=metadata_path,
        split="test",
        is_training=False,
        image_size=(512, 512),
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f"[*] Loaded test dataset: {len(test_dataset)} untouched samples.")

    # 2. Load Model & Weights
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = get_unet_model(in_channels=3, num_classes=1, base_channels=32, device=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    overall_tracker = MetricTracker()

    # Per-sample tracking for detailed diagnostics
    per_sample_results = []
    category_metrics_map: Dict[str, List[Dict[str, float]]] = {}

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating Test Set"):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            sample_ids = batch["sample_id"]
            is_tampered = batch["is_tampered"]
            manip_types = batch["manipulation_type"]
            source_types = batch["source_type"]

            logits = model(images)
            loss = criterion(logits, masks)

            batch_size_cur = images.size(0)
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()

            for i in range(batch_size_cur):
                p_mask = preds[i, 0].cpu()
                t_mask = masks[i, 0].cpu()

                tp = (p_mask * t_mask).sum().item()
                fp = (p_mask * (1.0 - t_mask)).sum().item()
                fn = ((1.0 - p_mask) * t_mask).sum().item()
                is_t = is_tampered[i].item()

                if is_t == 0:
                    dice = 1.0 if fp == 0 else 0.0
                    iou = 1.0 if fp == 0 else 0.0
                    prec = 1.0 if fp == 0 else 0.0
                    rec = 1.0
                else:
                    dice = (2.0 * tp + 1e-6) / (2.0 * tp + fp + fn + 1e-6)
                    iou = (tp + 1e-6) / (tp + fp + fn + 1e-6)
                    prec = (tp + 1e-6) / (tp + fp + 1e-6)
                    rec = (tp + 1e-6) / (tp + fn + 1e-6)

                m_type = manip_types[i]
                s_type = source_types[i]
                s_id = sample_ids[i]

                sample_res = {
                    "sample_id": s_id,
                    "source_type": s_type,
                    "manipulation_type": m_type,
                    "is_tampered": is_t,
                    "dice": round(dice, 4),
                    "iou": round(iou, 4),
                    "precision": round(prec, 4),
                    "recall": round(rec, 4),
                    "tp_pixels": int(tp),
                    "fp_pixels": int(fp),
                    "fn_pixels": int(fn),
                }
                per_sample_results.append(sample_res)

                if m_type not in category_metrics_map:
                    category_metrics_map[m_type] = []
                category_metrics_map[m_type].append(sample_res)

            batch_m = compute_batch_metrics(logits, masks, threshold=threshold)
            overall_tracker.update(loss.item(), batch_m, batch_size=batch_size_cur)

    summary_metrics = overall_tracker.compute()

    # Per-category summary
    category_summary = {}
    for cat, items in category_metrics_map.items():
        d_vals = [x["dice"] for x in items]
        i_vals = [x["iou"] for x in items]
        p_vals = [x["precision"] for x in items]
        r_vals = [x["recall"] for x in items]
        category_summary[cat] = {
            "sample_count": len(items),
            "mean_dice": round(float(np.mean(d_vals)), 4),
            "mean_iou": round(float(np.mean(i_vals)), 4),
            "mean_precision": round(float(np.mean(p_vals)), 4),
            "mean_recall": round(float(np.mean(r_vals)), 4),
        }

    # Identify Failure Cases (Lowest Dice on Tampered Documents)
    tampered_samples = [s for s in per_sample_results if s["is_tampered"] == 1]
    tampered_samples_sorted = sorted(tampered_samples, key=lambda x: x["dice"])
    worst_failures = tampered_samples_sorted[:8]

    final_report = {
        "model_name": "UNet",
        "best_epoch": checkpoint.get("epoch", -1),
        "device_used": str(device),
        "test_dataset_size": len(test_dataset),
        "test_metrics": {
            "test_loss": round(summary_metrics["loss"], 4),
            "test_dice": round(summary_metrics["dice"], 4),
            "test_iou": round(summary_metrics["iou"], 4),
            "test_precision": round(summary_metrics["precision"], 4),
            "test_recall": round(summary_metrics["recall"], 4),
        },
        "per_category_metrics": category_summary,
        "worst_failure_cases": worst_failures,
    }

    # Save to reports/unet_test_metrics.json
    out_path = REPORTS_DIR / "unet_test_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    # Print Formatted Test Results
    print("\n" + "=" * 75)
    print("             DOCFORENSICS AI — U-NET TEST SET BENCHMARK RESULTS       ")
    print("=" * 75)
    print(f" Test Set Size            : {len(test_dataset):,} samples")
    print(f" Test Dice Score (F1)     : {final_report['test_metrics']['test_dice']:.4f}")
    print(f" Test IoU (Jaccard Index) : {final_report['test_metrics']['test_iou']:.4f}")
    print(f" Test Precision           : {final_report['test_metrics']['test_precision']:.4f}")
    print(f" Test Recall              : {final_report['test_metrics']['test_recall']:.4f}")
    print(f" Test Loss (BCE + Dice)   : {final_report['test_metrics']['test_loss']:.4f}")
    print("-" * 75)
    print(" Performance Breakdown by Category:")
    for cat, c_data in category_summary.items():
        print(f"  • {cat:<28} ({c_data['sample_count']:>2} samples) | Dice: {c_data['mean_dice']:.4f} | IoU: {c_data['mean_iou']:.4f} | Prec: {c_data['mean_precision']:.4f} | Rec: {c_data['mean_recall']:.4f}")
    print("=" * 75 + "\n")

    return final_report


if __name__ == "__main__":
    evaluate_model_on_test_set()
