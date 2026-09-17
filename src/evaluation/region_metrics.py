"""Region-Level and Area-Stratified Metrics for Document Tampering Detection.

Computes:
1. Pixel-level metrics: Dice, IoU, Precision, Recall
2. Connected-component Region-level metrics: Region Recall, Region Precision, FP regions / doc
3. Area-stratified performance:
   - <0.5% (Tiny text / digits)
   - 0.5% - 1.0% (Small words / dates)
   - 1.0% - 2.0% (Sentences / table cells)
   - >2.0% (Paragraphs / large modifications)
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch


def compute_connected_component_regions(
    binary_mask: np.ndarray, min_area: int = 10
) -> List[Tuple[int, int, int, int, int]]:
    """Extract connected components as (x, y, w, h, area)."""
    mask_u8 = (binary_mask > 0.5).astype(np.uint8) * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_u8, connectivity=8
    )
    regions = []
    for i in range(1, num_labels):  # Skip background
        x, y, w, h, area = stats[i]
        if area >= min_area:
            regions.append((x, y, w, h, area))
    return regions


def evaluate_batch_region_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    iou_match_threshold: float = 0.20,
) -> Dict[str, Any]:
    """Evaluate pixel and region-level metrics on a batch of predictions and ground-truth masks."""
    preds_np = (preds >= threshold).cpu().numpy().squeeze(1)  # [B, H, W]
    targets_np = (targets >= 0.5).cpu().numpy().squeeze(1)    # [B, H, W]
    batch_size = preds_np.shape[0]

    pixel_tp, pixel_fp, pixel_fn = 0, 0, 0
    gt_regions_total = 0
    detected_regions_total = 0
    pred_regions_total = 0
    false_alarm_regions_total = 0

    # Area-stratified buckets: (gt_count, detected_count, dice_sum)
    area_buckets = {
        "<0.5%": {"gt_count": 0, "detected": 0, "dice_sum": 0.0, "sample_count": 0},
        "0.5-1%": {"gt_count": 0, "detected": 0, "dice_sum": 0.0, "sample_count": 0},
        "1-2%": {"gt_count": 0, "detected": 0, "dice_sum": 0.0, "sample_count": 0},
        ">2%": {"gt_count": 0, "detected": 0, "dice_sum": 0.0, "sample_count": 0},
    }

    for b in range(batch_size):
        p_mask = preds_np[b]
        t_mask = targets_np[b]

        # 1. Pixel stats
        tp = np.sum((p_mask == 1) & (t_mask == 1))
        fp = np.sum((p_mask == 1) & (t_mask == 0))
        fn = np.sum((p_mask == 0) & (t_mask == 1))
        pixel_tp += tp
        pixel_fp += fp
        pixel_fn += fn

        # Sample Dice
        sample_dice = (2.0 * tp + 1e-6) / (2.0 * tp + fp + fn + 1e-6)
        tampered_pct = (np.sum(t_mask == 1) / t_mask.size) * 100

        if tampered_pct < 0.5:
            bucket_key = "<0.5%"
        elif tampered_pct < 1.0:
            bucket_key = "0.5-1%"
        elif tampered_pct < 2.0:
            bucket_key = "1-2%"
        else:
            bucket_key = ">2%"

        area_buckets[bucket_key]["dice_sum"] += sample_dice
        area_buckets[bucket_key]["sample_count"] += 1

        # 2. Connected Component Regions
        gt_regions = compute_connected_component_regions(t_mask)
        pred_regions = compute_connected_component_regions(p_mask)

        gt_regions_total += len(gt_regions)
        pred_regions_total += len(pred_regions)
        area_buckets[bucket_key]["gt_count"] += len(gt_regions)

        # Match GT regions with predicted regions
        matched_gt_indices = set()
        matched_pred_indices = set()

        for g_idx, (gx, gy, gw, gh, garea) in enumerate(gt_regions):
            # Direct pixel overlap inside GT bounding box
            gt_crop_pred = p_mask[gy : gy + gh, gx : gx + gw]
            actual_overlap = int(np.sum(gt_crop_pred == 1))
            overlap_ratio = actual_overlap / max(1, garea)

            if overlap_ratio >= 0.15 or actual_overlap >= 5:
                matched_gt_indices.add(g_idx)

            for p_idx, (px, py, pw, ph, parea) in enumerate(pred_regions):
                # Fast coordinate intersection IoU
                ix1 = max(gx, px)
                iy1 = max(gy, py)
                ix2 = min(gx + gw, px + pw)
                iy2 = min(gy + gh, py + ph)
                inter_w = max(0, ix2 - ix1)
                inter_h = max(0, iy2 - iy1)
                inter_area = inter_w * inter_h
                union_area = (gw * gh) + (pw * ph) - inter_area
                box_iou = inter_area / max(1, union_area)

                if box_iou >= iou_match_threshold or (inter_area / max(1, parea)) >= 0.20:
                    matched_gt_indices.add(g_idx)
                    matched_pred_indices.add(p_idx)

        detected_count = len(matched_gt_indices)
        detected_regions_total += detected_count
        area_buckets[bucket_key]["detected"] += detected_count

        fa_count = len(pred_regions) - len(matched_pred_indices)
        false_alarm_regions_total += max(0, fa_count)

    # Calculate final aggregate metrics
    dice = (2.0 * pixel_tp + 1e-6) / (2.0 * pixel_tp + pixel_fp + pixel_fn + 1e-6)
    iou = (pixel_tp + 1e-6) / (pixel_tp + pixel_fp + pixel_fn + 1e-6)
    precision = (pixel_tp + 1e-6) / (pixel_tp + pixel_fp + 1e-6)
    recall = (pixel_tp + 1e-6) / (pixel_tp + pixel_fn + 1e-6)

    region_recall = (detected_regions_total + 1e-6) / (gt_regions_total + 1e-6)
    region_precision = (
        (len(matched_pred_indices) + 1e-6) / (pred_regions_total + 1e-6)
        if pred_regions_total > 0
        else 0.0
    )
    fp_per_doc = false_alarm_regions_total / max(1, batch_size)

    stratified_results = {}
    for k, v in area_buckets.items():
        cnt = v["sample_count"]
        gt_cnt = v["gt_count"]
        stratified_results[k] = {
            "sample_count": cnt,
            "mean_dice": round(v["dice_sum"] / max(1, cnt), 4),
            "region_recall": round((v["detected"] + 1e-6) / (gt_cnt + 1e-6), 4)
            if gt_cnt > 0
            else 0.0,
            "total_gt_regions": gt_cnt,
            "detected_gt_regions": v["detected"],
        }

    return {
        "pixel_dice": float(dice),
        "pixel_iou": float(iou),
        "pixel_precision": float(precision),
        "pixel_recall": float(recall),
        "region_recall": float(region_recall),
        "region_precision": float(region_precision),
        "fp_regions_per_doc": float(fp_per_doc),
        "total_gt_regions": gt_regions_total,
        "total_detected_regions": detected_regions_total,
        "area_stratified": stratified_results,
    }
