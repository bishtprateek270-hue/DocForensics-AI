"""
DocForensics AI — Phase 10 Evaluation, Threshold Calibration & Area-Bucketing Benchmark
Evaluates segmentation models across:
1. Overall Metrics: Dice, IoU, Precision, Recall, Pixel-FPR
2. Area Buckets: Tiny (<0.5%), Small (0.5-1.5%), Medium (1.5-4.0%), Large (>4.0%)
3. Category Metrics: text, number, date, stamp, signature, copy-move, inpainting, hard negatives
4. Multi-Scale Inference Modes: Full-Page, Patch-Only, Multi-Scale Fusion
5. Validation Threshold Calibration (0.20 - 0.80)
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_target_device, REPORTS_DIR, CHECKPOINT_DIR
from src.preprocessing.dataset import DocForensicsDataset, create_dataloaders
from src.forensics.patch_engine import PatchEngine, fuse_multiscale_predictions
from src.evaluation.metrics import compute_binary_metrics


def categorize_tampered_area(area_pct: float) -> str:
    """Classifies sample into empirical dataset area quantile bucket."""
    if area_pct <= 0.0:
        return "authentic"
    elif area_pct < 0.50:
        return "tiny"
    elif area_pct < 1.50:
        return "small"
    elif area_pct < 4.00:
        return "medium"
    else:
        return "large"


def evaluate_model_comprehensive(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.50,
    use_patch_engine: bool = False,
    patch_overlap: float = 0.25,
    multiscale_fusion: bool = False,
    alpha: float = 0.50,
) -> Dict[str, Any]:
    """
    Executes comprehensive evaluation computing overall, per-category, and per-area bucket metrics.
    """
    model.eval()
    patch_engine = PatchEngine(patch_size=512, overlap=patch_overlap) if (use_patch_engine or multiscale_fusion) else None

    total_samples = len(dataloader.dataset)
    all_dices, all_ious, all_precs, all_recalls = [], [], [], []
    latencies = []

    category_buckets: Dict[str, Dict[str, List[float]]] = {}
    area_buckets: Dict[str, Dict[str, List[float]]] = {
        "tiny": {"dice": [], "iou": [], "precision": [], "recall": []},
        "small": {"dice": [], "iou": [], "precision": [], "recall": []},
        "medium": {"dice": [], "iou": [], "precision": [], "recall": []},
        "large": {"dice": [], "iou": [], "precision": [], "recall": []},
        "authentic": {"dice": [], "iou": [], "precision": [], "recall": []},
    }

    hard_negative_fps = 0
    hard_negative_total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Phase 10 Model", leave=False):
            images = batch["image"].to(device)  # (B, 3, 512, 512)
            masks = batch["mask"].numpy()       # (B, 1, 512, 512)
            meta = batch["metadata"]
            batch_size = images.size(0)

            t0 = time.perf_counter()

            # 1. Full-Page Forward Pass
            with torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
                logits = model(images)
                full_page_probs = torch.sigmoid(logits).cpu().numpy()[:, 0]  # (B, 512, 512)

            t_end = time.perf_counter()
            latencies.append((t_end - t0) * 1000.0 / batch_size)

            for i in range(batch_size):
                gt_mask = masks[i, 0]  # (512, 512)
                full_prob = full_page_probs[i]

                # 2. Multi-Scale / Patch Inference if enabled
                if use_patch_engine and not multiscale_fusion:
                    # Patch only
                    img_np = (images[i].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
                    patches, coords, orig_shape = patch_engine.extract_patches(img_np)
                    patch_tensors = torch.from_numpy(np.array(patches)).permute(0, 3, 1, 2).float().to(device) / 255.0
                    with torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
                        p_logits = model(patch_tensors)
                        p_probs = torch.sigmoid(p_logits).cpu().numpy()[:, 0]
                    final_prob = patch_engine.reconstruct_probability_map(p_probs, coords, (512, 512))
                elif multiscale_fusion:
                    img_np = (images[i].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
                    patches, coords, orig_shape = patch_engine.extract_patches(img_np)
                    patch_tensors = torch.from_numpy(np.array(patches)).permute(0, 3, 1, 2).float().to(device) / 255.0
                    with torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
                        p_logits = model(patch_tensors)
                        p_probs = torch.sigmoid(p_logits).cpu().numpy()[:, 0]
                    patch_prob = patch_engine.reconstruct_probability_map(p_probs, coords, (512, 512))
                    final_prob = fuse_multiscale_predictions(full_prob, patch_prob, alpha=alpha)
                else:
                    final_prob = full_prob

                # Binary segmentation
                pred_binary = (final_prob >= threshold).astype(np.uint8)

                # Compute sample metrics
                cat = meta["manipulation_type"][i]
                area_pct = float(meta["tampered_area_percentage"][i])
                area_bucket = categorize_tampered_area(area_pct)

                sample_metrics = compute_binary_metrics(pred_binary, gt_mask)
                d = sample_metrics["dice"]
                iou = sample_metrics["iou"]
                prec = sample_metrics["precision"]
                rec = sample_metrics["recall"]

                all_dices.append(d)
                all_ious.append(iou)
                all_precs.append(prec)
                all_recalls.append(rec)

                # Record category
                if cat not in category_buckets:
                    category_buckets[cat] = {"dice": [], "iou": [], "precision": [], "recall": []}
                category_buckets[cat]["dice"].append(d)
                category_buckets[cat]["iou"].append(iou)
                category_buckets[cat]["precision"].append(prec)
                category_buckets[cat]["recall"].append(rec)

                # Record area bucket
                area_buckets[area_bucket]["dice"].append(d)
                area_buckets[area_bucket]["iou"].append(iou)
                area_buckets[area_bucket]["precision"].append(prec)
                area_buckets[area_bucket]["recall"].append(rec)

                # Hard Negative False Positive Tracking
                if cat in ["none", "authentic_hard_negative"] or area_pct == 0.0:
                    hard_negative_total += 1
                    if np.sum(pred_binary) > 0:
                        hard_negative_fps += 1

    # Aggregate summaries
    mean_dice = float(np.mean(all_dices))
    mean_iou = float(np.mean(all_ious))
    mean_prec = float(np.mean(all_precs))
    mean_rec = float(np.mean(all_recalls))
    hn_fpr = float(hard_negative_fps / max(1, hard_negative_total))

    per_category_summary = {}
    for c, val_dict in category_buckets.items():
        per_category_summary[c] = {
            "sample_count": len(val_dict["dice"]),
            "mean_dice": float(round(np.mean(val_dict["dice"]), 4)),
            "mean_iou": float(round(np.mean(val_dict["iou"]), 4)),
            "mean_precision": float(round(np.mean(val_dict["precision"]), 4)),
            "mean_recall": float(round(np.mean(val_dict["recall"]), 4)),
        }

    per_area_summary = {}
    for a, val_dict in area_buckets.items():
        if len(val_dict["dice"]) > 0:
            per_area_summary[a] = {
                "sample_count": len(val_dict["dice"]),
                "mean_dice": float(round(np.mean(val_dict["dice"]), 4)),
                "mean_iou": float(round(np.mean(val_dict["iou"]), 4)),
                "mean_precision": float(round(np.mean(val_dict["precision"]), 4)),
                "mean_recall": float(round(np.mean(val_dict["recall"]), 4)),
            }

    return {
        "mean_dice": round(mean_dice, 4),
        "mean_iou": round(mean_iou, 4),
        "mean_precision": round(mean_prec, 4),
        "mean_recall": round(mean_rec, 4),
        "hard_negative_false_positive_rate": round(hn_fpr, 4),
        "avg_latency_ms": round(float(np.mean(latencies)), 2),
        "threshold_used": threshold,
        "per_category_metrics": per_category_summary,
        "per_area_bucket_metrics": per_area_summary,
    }


def calibrate_threshold_on_validation(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    threshold_range: Tuple[float, float, float] = (0.20, 0.80, 0.02)
) -> Tuple[float, Dict[str, Any]]:
    """
    Sweeps thresholds on validation data only to find the optimal operating point.
    Objective: Maximize Validation Dice while bounding false positive rate on hard negatives.
    """
    model.eval()
    start_th, end_th, step_th = threshold_range
    thresholds = np.arange(start_th, end_th + 1e-5, step_th)

    best_th = 0.50
    best_score = -1.0
    sweep_history = []

    print("[*] Running Validation Threshold Calibration Sweep...")
    for th in thresholds:
        th = round(float(th), 2)
        res = evaluate_model_comprehensive(model, val_loader, device, threshold=th)
        dice = res["mean_dice"]
        fpr = res["hard_negative_false_positive_rate"]
        # Score penalizes high FPR
        composite_score = dice - (0.5 * fpr)

        sweep_history.append({
            "threshold": th,
            "dice": dice,
            "iou": res["mean_iou"],
            "precision": res["mean_precision"],
            "recall": res["mean_recall"],
            "hard_negative_fpr": fpr,
        })

        if composite_score > best_score:
            best_score = composite_score
            best_th = th

    print(f"[+] Optimal Validation Threshold calibrated: {best_th:.2f} (Val Dice: {sweep_history[int((best_th-start_th)/step_th)]['dice']:.4f})")
    return best_th, {"best_threshold": best_th, "sweep_results": sweep_history}
