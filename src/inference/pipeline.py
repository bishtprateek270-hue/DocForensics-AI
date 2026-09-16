"""
DocForensics AI — End-to-End Forensic Inference & Report Generation Pipeline (Phase 8)

Pipeline Flow:
1. Document Input Validation & Original Geometry Preservation (JPG, PNG)
2. Dual-Stream RGB + SRM Model Forward Pass on CUDA with AMP
3. Full-Resolution Probability Heatmap & Binary Thresholding
4. Connected Component Suspicious Region Extraction
5. OCR Text Detection & Geometric Region-Text Association
6. Conservative Visual Region Heuristic Classification
7. Structured Forensic JSON Report Generation
8. Multi-Panel Visual Report & High-Resolution Region Crops
"""

import os
import sys
import time
import json
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timezone

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from config import (
    BASE_DIR,
    CHECKPOINT_DIR,
    REPORTS_DIR,
    OUTPUT_DIR,
    cfg,
    get_target_device,
)
from src.models.dual_stream_forensics import get_dual_stream_model, DualStreamForensicNet
from src.ocr.ocr_engine import OCREngine, OCREntry, get_ocr_engine
from src.training.gpu_utils import get_vram_usage


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def calculate_bbox_intersection(boxA: List[int], boxB: List[int]) -> float:
    """
    Computes the Intersection area over the area of boxB (boxA is region, boxB is OCR text).
    Coordinates: [x1, y1, x2, y2]
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    inter_area = inter_width * inter_height

    boxB_area = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    return inter_area / float(boxB_area)


def classify_region_heuristic(
    bbox: List[int],
    doc_shape: Tuple[int, int],
    has_text: bool,
    crop_rgb: np.ndarray,
) -> str:
    """
    Assigns a conservative descriptive heuristic classification to a suspicious region.
    """
    if has_text:
        return "text_or_number_region"

    h_crop, w_crop = crop_rgb.shape[:2]
    doc_h, doc_w = doc_shape
    aspect_ratio = w_crop / max(1, h_crop)
    area_ratio = (h_crop * w_crop) / float(doc_h * doc_w)

    # Convert to HSV to detect colored stamps/seals
    hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
    # Saturation and distinct hue check (stamps/seals are often blue/red/violet)
    sat_mean = np.mean(hsv[:, :, 1])

    if sat_mean > 55 and aspect_ratio > 0.6 and aspect_ratio < 1.8:
        return "stamp_or_seal_like_region"

    # Photo / portrait box heuristic (typically prominent rectangular aspect ratio in ID/certificates)
    if area_ratio > 0.04 and 0.6 < aspect_ratio < 1.5:
        return "photo_or_image_like_region"

    # Elongated strokes without OCR text (potential signature line)
    if aspect_ratio > 2.2 and h_crop < 120:
        return "signature_like_region"

    return "unknown_visual_region"


class DocForensicsPipeline:
    """
    End-to-End Forensic Tampering Localization, OCR Association, and Report Generation Pipeline.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        threshold: float = 0.50,
        min_region_pixels: int = 64,
        device: Optional[torch.device] = None,
        use_ocr: bool = True,
    ):
        self.device = device or get_target_device(cfg.training.device)
        self.checkpoint_path = checkpoint_path or (CHECKPOINT_DIR / "dual_stream_best.pth")
        self.threshold = threshold
        self.min_region_pixels = min_region_pixels
        self.use_ocr = use_ocr

        print(f"[*] Initializing DocForensics Pipeline on {self.device}...")
        print(f"[*] Loading model checkpoint from: {self.checkpoint_path}")

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.checkpoint_path}")

        # Load Checkpoint & Determine Fusion Type
        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        fusion_type = "gated" if "fusion.gate_conv.0.weight" in ckpt["model_state_dict"] else "baseline"

        self.model = get_dual_stream_model(
            in_channels=3,
            num_classes=1,
            pretrained_backbone=False,
            fusion_type=fusion_type,
            device=self.device,
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        self.ocr_engine = get_ocr_engine(use_gpu=(self.device.type == "cuda")) if self.use_ocr else None
        print("[+] DocForensics Pipeline initialized successfully.")

    def preprocess_image(self, image_rgb: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Preprocesses original image into normalized 512x512 tensor.
        Preserves original dimensions (H0, W0).
        """
        h0, w0 = image_rgb.shape[:2]
        # Resize to model input size (512, 512)
        resized = cv2.resize(image_rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
        img_norm = resized.astype(np.float32) / 255.0

        # Standard ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_norm = (img_norm - mean) / std

        tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float()
        return tensor.to(self.device), (h0, w0)

    def analyze_document(
        self,
        image_path: Union[str, Path],
        document_id: Optional[str] = None,
        save_visual_report: bool = True,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Executes complete forensic analysis on a document image.
        
        Args:
            image_path: Path to document image.
            document_id: Optional identifier (defaults to filename or UUID).
            save_visual_report: Whether to generate visual overlay PNGs and crops.
            output_dir: Destination folder for reports.
            
        Returns:
            Structured JSON forensic report dictionary.
        """
        path_obj = Path(image_path)
        doc_id = document_id or path_obj.stem
        start_total = time.perf_counter()

        # 1. Input Validation
        if not path_obj.exists():
            raise FileNotFoundError(f"Document file does not exist: {path_obj}")

        if path_obj.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file format '{path_obj.suffix}'. Supported: {SUPPORTED_EXTENSIONS}")

        img_bgr = cv2.imread(str(path_obj))
        if img_bgr is None or img_bgr.size == 0:
            raise ValueError(f"Failed to read image file (corrupted or unreadable): {path_obj}")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h0, w0 = img_rgb.shape[:2]

        # Reset GPU memory stats
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        # 2. Dual-Stream Model Inference
        t_model_start = time.perf_counter()
        input_tensor, _ = self.preprocess_image(img_rgb)

        with torch.no_grad():
            with torch.amp.autocast(self.device.type, enabled=(self.device.type == "cuda")):
                logits = self.model(input_tensor)
                prob_map_512 = torch.sigmoid(logits)[0, 0]

            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)

        t_model_end = time.perf_counter()
        model_latency_ms = (t_model_end - t_model_start) * 1000.0

        # 3. Restore Probability Heatmap to Original Resolution (H0, W0)
        prob_map_512_cpu = prob_map_512.float().cpu().numpy()
        prob_heatmap_orig = cv2.resize(prob_map_512_cpu, (w0, h0), interpolation=cv2.INTER_LINEAR)
        prob_heatmap_orig = np.clip(prob_heatmap_orig, 0.0, 1.0)

        # 4. Binary Thresholding & Post-Processing
        t_post_start = time.perf_counter()
        binary_mask_raw = (prob_heatmap_orig >= self.threshold).astype(np.uint8)

        # Connected Components Extraction
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask_raw, connectivity=8)

        suspicious_regions: List[Dict[str, Any]] = []
        clean_binary_mask = np.zeros((h0, w0), dtype=np.uint8)
        region_counter = 1

        for label_idx in range(1, num_labels):
            area = stats[label_idx, cv2.CC_STAT_AREA]
            # Discard tiny isolated noise specks
            if area < self.min_region_pixels:
                continue

            x = stats[label_idx, cv2.CC_STAT_LEFT]
            y = stats[label_idx, cv2.CC_STAT_TOP]
            w = stats[label_idx, cv2.CC_STAT_WIDTH]
            h = stats[label_idx, cv2.CC_STAT_HEIGHT]
            cx, cy = centroids[label_idx]

            region_mask = (labels == label_idx)
            clean_binary_mask[region_mask] = 1

            region_probs = prob_heatmap_orig[region_mask]
            mean_score = float(np.mean(region_probs))
            max_score = float(np.max(region_probs))
            pct_area = (area / float(h0 * w0)) * 100.0

            bbox = [int(x), int(y), int(x + w), int(y + h)]
            crop_rgb = img_rgb[y : y + h, x : x + w]

            suspicious_regions.append({
                "region_id": region_counter,
                "bbox": bbox,
                "center": [round(float(cx), 1), round(float(cy), 1)],
                "pixel_area": int(area),
                "percentage_of_document_area": round(pct_area, 3),
                "mean_tampering_score": round(mean_score, 4),
                "max_tampering_score": round(max_score, 4),
                "crop_rgb": crop_rgb,
            })
            region_counter += 1

        t_post_end = time.perf_counter()
        post_latency_ms = (t_post_end - t_post_start) * 1000.0

        # 5. OCR Execution & Region Association
        ocr_latency_ms = 0.0
        ocr_entries: List[OCREntry] = []

        if self.ocr_engine is not None:
            ocr_entries, ocr_latency_ms = self.ocr_engine.extract_text(img_rgb)

        # Associate OCR text with each suspicious region
        for region in suspicious_regions:
            r_box = region["bbox"]
            overlapping_texts = []
            overlapping_confs = []

            for ocr_item in ocr_entries:
                overlap_ratio = calculate_bbox_intersection(r_box, ocr_item.bbox)
                if overlap_ratio >= 0.25:  # At least 25% of OCR text box inside region
                    overlapping_texts.append(ocr_item.text)
                    overlapping_confs.append(ocr_item.confidence)

            if overlapping_texts:
                combined_text = " ".join(overlapping_texts)
                mean_ocr_conf = float(np.mean(overlapping_confs))
                region["ocr_text"] = combined_text
                region["ocr_confidence"] = round(mean_ocr_conf, 4)
                region["has_associated_text"] = True
            else:
                region["ocr_text"] = "No OCR text associated with this suspicious region"
                region["ocr_confidence"] = 0.0
                region["has_associated_text"] = False

            # Determine conservative visual region heuristic
            region["region_type"] = classify_region_heuristic(
                bbox=region["bbox"],
                doc_shape=(h0, w0),
                has_text=region["has_associated_text"],
                crop_rgb=region["crop_rgb"],
            )

        # 6. Conservative Document-Level Assessment
        total_suspicious_pixels = sum(r["pixel_area"] for r in suspicious_regions)
        total_suspicious_pct = round((total_suspicious_pixels / float(h0 * w0)) * 100.0, 3)
        highest_score = max([r["max_tampering_score"] for r in suspicious_regions], default=0.0)

        if len(suspicious_regions) == 0:
            analysis_status = "no_significant_tampering_evidence_detected"
            assessment_summary = "No significant visual tampering evidence detected by the model."
        elif highest_score >= 0.70 or total_suspicious_pct >= 0.50:
            analysis_status = "suspicious_visual_manipulation_detected"
            assessment_summary = f"Detected {len(suspicious_regions)} suspicious localized region(s) with high tampering likelihood."
        else:
            analysis_status = "manual_review_recommended"
            assessment_summary = f"Detected {len(suspicious_regions)} borderline suspicious region(s); expert manual inspection recommended."

        total_pipeline_latency_ms = (time.perf_counter() - start_total) * 1000.0
        peak_vram_mb = round(torch.cuda.max_memory_allocated(self.device) / (1024 ** 2), 2) if self.device.type == "cuda" else 0.0

        # 7. Build Output Directories
        base_out = output_dir or (OUTPUT_DIR / "phase8_visual_reports")
        reports_json_dir = REPORTS_DIR / "phase8_forensic_reports"
        base_out.mkdir(parents=True, exist_ok=True)
        reports_json_dir.mkdir(parents=True, exist_ok=True)

        crops_dir = base_out / "regions" / doc_id
        if save_visual_report and suspicious_regions:
            crops_dir.mkdir(parents=True, exist_ok=True)

        # 8. Save Visual Forensic Composite and Crops
        visual_report_path = None
        saved_crop_paths = []

        if save_visual_report:
            visual_report_path = base_out / f"{doc_id}_forensic_report.png"
            self._generate_visual_report(
                image_rgb=img_rgb,
                heatmap=prob_heatmap_orig,
                regions=suspicious_regions,
                output_path=visual_report_path,
                doc_id=doc_id,
                status_text=analysis_status,
            )

            # Save individual region crops
            for r in suspicious_regions:
                crop_path = crops_dir / f"region_{r['region_id']}.png"
                crop_bgr = cv2.cvtColor(r["crop_rgb"], cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(crop_path), crop_bgr)
                saved_crop_paths.append(str(crop_path))

        # Clean crop_rgb from JSON report payload (NumPy arrays are not JSON serializable)
        serialized_regions = []
        for r in suspicious_regions:
            r_dict = {k: v for k, v in r.items() if k != "crop_rgb"}
            serialized_regions.append(r_dict)

        # 9. Assemble Structured JSON Report
        report = {
            "document_id": doc_id,
            "filename": path_obj.name,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "analysis_status": analysis_status,
            "assessment_summary": assessment_summary,
            "model_architecture": "dual_stream_rgb_srm_forensic",
            "checkpoint_loaded": str(self.checkpoint_path),
            "inference_threshold": self.threshold,
            "original_resolution": [w0, h0],
            "suspicious_region_count": len(serialized_regions),
            "total_suspicious_area_percent": total_suspicious_pct,
            "highest_tampering_score": round(highest_score, 4),
            "performance_latency": {
                "model_inference_ms": round(model_latency_ms, 2),
                "ocr_processing_ms": round(ocr_latency_ms, 2),
                "post_processing_ms": round(post_latency_ms, 2),
                "total_pipeline_latency_ms": round(total_pipeline_latency_ms, 2),
                "peak_vram_mb": peak_vram_mb,
            },
            "suspicious_regions": serialized_regions,
            "visual_report_path": str(visual_report_path) if visual_report_path else None,
            "region_crop_paths": saved_crop_paths,
        }

        # Save JSON Report
        json_report_path = reports_json_dir / f"{doc_id}_report.json"
        with open(json_report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        return report

    def _generate_visual_report(
        self,
        image_rgb: np.ndarray,
        heatmap: np.ndarray,
        regions: List[Dict[str, Any]],
        output_path: Path,
        doc_id: str,
        status_text: str,
    ):
        """Generates a 3-panel professional forensic composite."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
        plt.subplots_adjust(wspace=0.08)

        # 1. Original Document
        axes[0].imshow(image_rgb)
        axes[0].set_title("Original Document", fontsize=12, fontweight="bold", pad=10)
        axes[0].axis("off")

        # 2. Tampering Heatmap (Jet Colormap)
        im_hm = axes[1].imshow(heatmap, cmap="jet", vmin=0.0, vmax=1.0)
        axes[1].set_title("Forensic Probability Heatmap", fontsize=12, fontweight="bold", pad=10)
        axes[1].axis("off")
        cbar = plt.colorbar(im_hm, ax=axes[1], fraction=0.046, pad=0.04)
        cbar.set_label("Tampering Score", fontsize=9)

        # 3. Localization Overlay with Bounding Boxes & Region IDs
        overlay = image_rgb.copy()
        for r in regions:
            x1, y1, x2, y2 = r["bbox"]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), max(2, int(image_rgb.shape[1] / 350)))

            label = f"Region #{r['region_id']} ({r['mean_tampering_score']:.2f})"
            font_scale = max(0.4, image_rgb.shape[1] / 1200)
            thickness = max(1, int(image_rgb.shape[1] / 600))
            (w_lbl, h_lbl), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            
            # Label background box
            cv2.rectangle(overlay, (x1, max(0, y1 - h_lbl - 8)), (x1 + w_lbl + 6, max(0, y1)), (255, 0, 0), -1)
            cv2.putText(
                overlay,
                label,
                (x1 + 3, max(0, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

        axes[2].imshow(overlay)
        axes[2].set_title(f"Localization Overlay ({status_text})", fontsize=12, fontweight="bold", pad=10)
        axes[2].axis("off")

        plt.suptitle(
            f"DocForensics AI — Forensic Tampering Report: {doc_id}",
            fontsize=14,
            fontweight="bold",
            y=0.98,
        )

        plt.tight_layout()
        plt.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close()


def run_forensic_pipeline(
    image_path: Union[str, Path],
    checkpoint_path: Optional[Path] = None,
    threshold: float = 0.50,
    use_ocr: bool = True,
) -> Dict[str, Any]:
    """Convenience functional interface to execute forensic pipeline on a single document."""
    pipeline = DocForensicsPipeline(
        checkpoint_path=checkpoint_path,
        threshold=threshold,
        use_ocr=use_ocr,
    )
    return pipeline.analyze_document(image_path)
