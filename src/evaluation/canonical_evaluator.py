"""
DocForensics AI — Phase 11 Canonical Evaluator Engine
The single authoritative evaluation implementation for all DocForensics AI models.
Controls:
1. Deterministic image loading, EXIF orientation correction, RGB conversion, resizing, and normalization.
2. Exact FP32 forward inference.
3. Probability thresholding, connected-component analysis, and morphological noise filtering.
4. Pixel-level metrics (Dice, IoU, Precision, Recall, Pixel-FPR).
5. Region-level metrics (Detected, Partially Detected, Missed with explicit IoU overlap criterion).
6. Document-level metrics (Tampered Detection Rate, Authentic Document False-Positive Rate).
"""

import os
import sys
import time
import math
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_target_device, CHECKPOINT_DIR
from src.evaluation.metrics import compute_binary_metrics


class CanonicalEvaluator:
    """
    Authoritative evaluation engine ensuring 100% reproducible and standardized metrics.
    """

    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None,
        target_size: Tuple[int, int] = (512, 512),
        threshold: float = 0.50,
        min_component_area: int = 16,
        region_iou_threshold: float = 0.25,
    ):
        self.device = device or get_target_device("auto")
        self.target_size = target_size
        self.threshold = threshold
        self.min_component_area = min_component_area
        self.region_iou_threshold = region_iou_threshold

        if model is not None:
            self.model = model.to(self.device)
        elif checkpoint_path is not None:
            from src.models.dual_stream_forensics import get_dual_stream_model
            ckpt_p = Path(checkpoint_path)
            if not ckpt_p.exists():
                raise FileNotFoundError(f"Checkpoint not found at: {ckpt_p}")
            self.model = get_dual_stream_model(
                in_channels=3,
                num_classes=1,
                pretrained_backbone=False,
                fusion_type="baseline",
                device=self.device,
            )
            ckpt = torch.load(ckpt_p, map_location=self.device, weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)
            self.model.load_state_dict(state_dict, strict=False)
        else:
            raise ValueError("Either model or checkpoint_path must be provided.")

        self.model.eval()

    def preprocess_image(self, image_input: Union[str, Path, np.ndarray, Image.Image]) -> Tuple[torch.Tensor, np.ndarray, Tuple[int, int]]:
        """
        Loads and standardizes input image with EXIF orientation handling and ImageNet normalization.
        
        Returns:
            tensor: [1, 3, 512, 512] float tensor on self.device
            rgb_orig: [H, W, 3] uint8 numpy array of original image
            orig_shape: (H, W) tuple
        """
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(image_input)
            pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
            rgb_np = np.array(pil_img)
        elif isinstance(image_input, Image.Image):
            pil_img = ImageOps.exif_transpose(image_input).convert("RGB")
            rgb_np = np.array(pil_img)
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 2:
                rgb_np = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            elif image_input.shape[2] == 4:
                rgb_np = cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
            else:
                rgb_np = image_input.copy()
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        orig_h, orig_w = rgb_np.shape[:2]

        # Resize to target 512x512 with bilinear interpolation
        resized = cv2.resize(rgb_np, self.target_size, interpolation=cv2.INTER_LINEAR)
        img_float = resized.astype(np.float32) / 255.0

        # ImageNet standardization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (img_float - mean) / std

        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        return tensor, rgb_np, (orig_h, orig_w)

    def filter_connected_components(self, binary_mask: np.ndarray) -> np.ndarray:
        """Removes isolated speckles and noise regions smaller than min_component_area."""
        if np.sum(binary_mask) == 0:
            return binary_mask
        
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask.astype(np.uint8), connectivity=8)
        filtered = np.zeros_like(binary_mask, dtype=np.uint8)
        for label_idx in range(1, num_labels):
            area = stats[label_idx, cv2.CC_STAT_AREA]
            if area >= self.min_component_area:
                filtered[labels == label_idx] = 1
        return filtered

    def extract_regions(self, binary_mask: np.ndarray) -> List[Dict[str, Any]]:
        """Extracts bounding boxes and pixel coordinates for connected components."""
        if np.sum(binary_mask) == 0:
            return []
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask.astype(np.uint8), connectivity=8)
        regions = []
        for i in range(1, num_labels):
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            area = int(stats[i, cv2.CC_STAT_AREA])
            mask_roi = (labels == i).astype(np.uint8)
            regions.append({
                "bbox": [x, y, x + w, y + h],
                "area": area,
                "centroid": [float(centroids[i][0]), float(centroids[i][1])],
                "mask": mask_roi
            })
        return regions

    def compute_region_overlap_metrics(
        self,
        pred_regions: List[Dict[str, Any]],
        gt_regions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Computes region-level recall and precision.
        - Detected: IoU >= region_iou_threshold (0.25)
        - Partially Detected: 0.05 <= IoU < region_iou_threshold
        - Missed: IoU < 0.05
        """
        if len(gt_regions) == 0:
            # Authentic sample
            return {
                "gt_region_count": 0,
                "pred_region_count": len(pred_regions),
                "detected_count": 0,
                "partially_detected_count": 0,
                "missed_count": 0,
                "false_positive_regions": len(pred_regions),
                "region_recall": 1.0,
                "region_precision": 1.0 if len(pred_regions) == 0 else 0.0
            }

        detected = 0
        partial = 0
        missed = 0
        matched_preds = set()

        for gt in gt_regions:
            gt_mask = gt["mask"]
            best_iou = 0.0
            best_p_idx = -1

            for p_idx, pred in enumerate(pred_regions):
                pred_mask = pred["mask"]
                intersection = np.sum(gt_mask & pred_mask)
                union = np.sum(gt_mask | pred_mask)
                iou = float(intersection) / max(1, union)
                if iou > best_iou:
                    best_iou = iou
                    best_p_idx = p_idx

            if best_iou >= self.region_iou_threshold:
                detected += 1
                if best_p_idx >= 0:
                    matched_preds.add(best_p_idx)
            elif best_iou >= 0.05:
                partial += 1
                if best_p_idx >= 0:
                    matched_preds.add(best_p_idx)
            else:
                missed += 1

        false_regions = len(pred_regions) - len(matched_preds)
        reg_recall = float(detected + 0.5 * partial) / len(gt_regions)
        reg_prec = float(len(matched_preds)) / max(1, len(pred_regions)) if len(pred_regions) > 0 else 0.0

        return {
            "gt_region_count": len(gt_regions),
            "pred_region_count": len(pred_regions),
            "detected_count": detected,
            "partially_detected_count": partial,
            "missed_count": missed,
            "false_positive_regions": false_regions,
            "region_recall": round(reg_recall, 4),
            "region_precision": round(reg_prec, 4)
        }

    @torch.no_grad()
    def evaluate_sample(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image],
        ground_truth_mask: Optional[np.ndarray] = None,
        is_tampered: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Executes complete canonical evaluation on a single sample.
        """
        t0 = time.perf_counter()
        tensor, rgb_orig, orig_shape = self.preprocess_image(image_input)

        # Exact deterministic FP32 inference
        logits = self.model(tensor)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()  # (512, 512)
        t_inference_ms = (time.perf_counter() - t0) * 1000.0

        raw_binary = (probs >= self.threshold).astype(np.uint8)
        filtered_binary = self.filter_connected_components(raw_binary)

        # Resize probability heatmap back to original dimensions for reporting
        prob_orig = cv2.resize(probs, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_LINEAR)
        pred_mask_orig = cv2.resize(filtered_binary, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST)

        pred_regions = self.extract_regions(filtered_binary)
        highest_region_score = float(np.max(probs)) if probs.size > 0 else 0.0

        res: Dict[str, Any] = {
            "inference_latency_ms": round(t_inference_ms, 2),
            "highest_region_score": round(highest_region_score, 4),
            "predicted_tampered_pixels": int(np.sum(filtered_binary)),
            "predicted_tampered_area_pct": round(float(np.sum(filtered_binary)) / (512 * 512) * 100.0, 4),
            "predicted_region_count": len(pred_regions),
            "predicted_regions": pred_regions,
            "is_flagged_suspicious": bool(np.sum(filtered_binary) >= self.min_component_area and highest_region_score >= self.threshold),
            "probability_map": prob_orig,
            "binary_mask": pred_mask_orig,
        }

        # If ground truth mask provided, compute metrics
        if ground_truth_mask is not None:
            # Resize gt_mask to 512x512
            if ground_truth_mask.shape != (512, 512):
                gt_512 = cv2.resize(ground_truth_mask.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST)
            else:
                gt_512 = ground_truth_mask.astype(np.uint8)

            gt_regions = self.extract_regions(gt_512)
            pixel_metrics = compute_binary_metrics(filtered_binary, gt_512)
            region_metrics = self.compute_region_overlap_metrics(pred_regions, gt_regions)

            res["metrics"] = {
                "dice": round(pixel_metrics["dice"], 4),
                "iou": round(pixel_metrics["iou"], 4),
                "precision": round(pixel_metrics["precision"], 4),
                "recall": round(pixel_metrics["recall"], 4),
                **region_metrics
            }

        return res

    def evaluate_dataloader(self, dataloader, desc: str = "Canonical Evaluation") -> Dict[str, Any]:
        """Evaluates an entire dataset and returns structured overall, category, and region metrics."""
        from tqdm import tqdm
        all_dices, all_ious, all_precs, all_recalls = [], [], [], []
        all_reg_recalls, all_reg_precs = [], []
        latencies = []
        tampered_correct = 0
        tampered_total = 0
        authentic_fps = 0
        authentic_total = 0

        category_map: Dict[str, Dict[str, List[float]]] = {}

        for batch in tqdm(dataloader, desc=desc, leave=False):
            images = batch["image"].to(self.device)
            masks = batch["mask"].numpy()[:, 0]
            batch_size = images.size(0)

            t0 = time.perf_counter()
            with torch.no_grad():
                logits = self.model(images)
                probs = torch.sigmoid(logits).cpu().numpy()[:, 0]
            t_end = time.perf_counter()
            latencies.append((t_end - t0) * 1000.0 / batch_size)

            for i in range(batch_size):
                gt = masks[i].astype(np.uint8)
                prob = probs[i]
                raw_bin = (prob >= self.threshold).astype(np.uint8)
                filt_bin = self.filter_connected_components(raw_bin)

                gt_regions = self.extract_regions(gt)
                pred_regions = self.extract_regions(filt_bin)

                pix_m = compute_binary_metrics(filt_bin, gt)
                reg_m = self.compute_region_overlap_metrics(pred_regions, gt_regions)

                d, iou, prec, rec = pix_m["dice"], pix_m["iou"], pix_m["precision"], pix_m["recall"]
                all_dices.append(d)
                all_ious.append(iou)
                all_precs.append(prec)
                all_recalls.append(rec)
                all_reg_recalls.append(reg_m["region_recall"])
                all_reg_precs.append(reg_m["region_precision"])

                is_t = int(np.sum(gt) > 0)
                is_flagged = int(np.sum(filt_bin) >= self.min_component_area)

                if is_t == 1:
                    tampered_total += 1
                    if is_flagged == 1:
                        tampered_correct += 1
                else:
                    authentic_total += 1
                    if is_flagged == 1:
                        authentic_fps += 1

                # Record category if metadata present
                cat = "standard"
                if "metadata" in batch and "manipulation_type" in batch["metadata"]:
                    cat = batch["metadata"]["manipulation_type"][i]
                elif "manipulation_type" in batch:
                    cat = batch["manipulation_type"][i]

                if cat not in category_map:
                    category_map[cat] = {"dice": [], "iou": [], "precision": [], "recall": []}
                category_map[cat]["dice"].append(d)
                category_map[cat]["iou"].append(iou)
                category_map[cat]["precision"].append(prec)
                category_map[cat]["recall"].append(rec)

        cat_summary = {}
        for c, v in category_map.items():
            cat_summary[c] = {
                "sample_count": len(v["dice"]),
                "mean_dice": round(float(np.mean(v["dice"])), 4),
                "mean_iou": round(float(np.mean(v["iou"])), 4),
                "mean_precision": round(float(np.mean(v["precision"])), 4),
                "mean_recall": round(float(np.mean(v["recall"])), 4),
            }

        return {
            "total_samples": len(all_dices),
            "mean_dice": round(float(np.mean(all_dices)), 4),
            "mean_iou": round(float(np.mean(all_ious)), 4),
            "mean_precision": round(float(np.mean(all_precs)), 4),
            "mean_recall": round(float(np.mean(all_recalls)), 4),
            "region_recall": round(float(np.mean(all_reg_recalls)), 4),
            "region_precision": round(float(np.mean(all_reg_precs)), 4),
            "tampered_document_detection_rate": round(float(tampered_correct / max(1, tampered_total)), 4),
            "authentic_document_fpr": round(float(authentic_fps / max(1, authentic_total)), 4),
            "avg_latency_ms": round(float(np.mean(latencies)), 2),
            "per_category_metrics": cat_summary
        }
