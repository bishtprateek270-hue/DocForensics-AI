"""Dual-Specialist Evidence Fusion Engine for DocForensics AI.

Fuses:
1. Model A (Phase 7 Physical/Image Forensic Model): Photos, signatures, stamps, splicing, sensor noise.
2. Model B (Tiny-Text Digital Forensic Model): Post-processed numbers, dates, character modifications.
3. Content Consistency Engine: Arithmetic/logical consistency checks from OCR tokens.
4. Reference Verification: Optional authoritative database verification.

Outputs strictly structured, conservative forensic evidence without synthetic authenticity percentages.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

from src.forensics.tinytext_postprocessor import TinyTextPostProcessor

logger = logging.getLogger(__name__)


class EvidenceFusionEngine:
    """Orchestrates multi-specialist inference and spatial evidence fusion."""

    def __init__(
        self,
        model_a_physical: Optional[nn.Module] = None,
        model_b_tinytext: Optional[nn.Module] = None,
        tinytext_postprocessor: Optional[TinyTextPostProcessor] = None,
        iou_fusion_threshold: float = 0.20,
    ) -> None:
        self.model_a = model_a_physical
        self.model_b = model_b_tinytext
        self.postprocessor_b = (
            tinytext_postprocessor
            if tinytext_postprocessor is not None
            else TinyTextPostProcessor(threshold=0.45, min_component_area=15)
        )
        self.iou_fusion_threshold = iou_fusion_threshold

    def extract_physical_regions(
        self, prob_map: np.ndarray, threshold: float = 0.50, min_area: int = 50
    ) -> List[Dict[str, Any]]:
        """Extract physical/image manipulation regions from Model A probability map."""
        binary = (prob_map >= threshold).astype(np.uint8) * 255
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        regions = []

        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area < min_area:
                continue

            comp_mask = (labels == i)
            comp_probs = prob_map[comp_mask]
            mean_p = float(np.mean(comp_probs))
            max_p = float(np.max(comp_probs))

            regions.append({
                "region_id": f"physical_region_{i}",
                "bbox": [int(x), int(y), int(w), int(h)],
                "area_pixels": int(area),
                "region_evidence_score": round(0.60 * mean_p + 0.40 * max_p, 4),
                "evidence_source": "Phase 7 Physical Forensic Model",
                "explanation": "Suspicious visual manipulation evidence detected (splicing/sensor artifact).",
            })

        return regions

    def fuse_overlapping_regions(
        self,
        physical_regions: List[Dict[str, Any]],
        text_regions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Merge spatial detections flagged by both Model A and Model B."""
        fused = []
        used_text_indices = set()

        for phys in physical_regions:
            px, py, pw, ph = phys["bbox"]
            matched_text = []

            for t_idx, txt in enumerate(text_regions):
                if t_idx in used_text_indices:
                    continue
                tx, ty, tw, th = txt["bbox"]

                # Check IoU
                ix1 = max(px, tx)
                iy1 = max(py, ty)
                ix2 = min(px + pw, tx + tw)
                iy2 = min(py + ph, ty + th)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                union = (pw * ph) + (tw * th) - inter
                iou = inter / max(1, union)

                if iou >= self.iou_fusion_threshold or (inter / max(1, tw * th)) >= 0.30:
                    matched_text.append((t_idx, txt))
                    used_text_indices.add(t_idx)

            if matched_text:
                # Combined bounding box
                all_boxes = [phys["bbox"]] + [t["bbox"] for _, t in matched_text]
                min_x = min(b[0] for b in all_boxes)
                min_y = min(b[1] for b in all_boxes)
                max_x = max(b[0] + b[2] for b in all_boxes)
                max_y = max(b[1] + b[3] for b in all_boxes)

                ocr_texts = [t.get("associated_ocr_text", "") for _, t in matched_text]
                combined_text = " ".join([t for t in ocr_texts if t])

                fused.append({
                    "region_id": f"fused_region_{len(fused)+1}",
                    "bbox": [min_x, min_y, max_x - min_x, max_y - min_y],
                    "evidence_sources": [
                        "Phase 7 Physical Forensic Model",
                        "Tiny-Text Digital Forensic Model",
                    ],
                    "physical_evidence_score": phys["region_evidence_score"],
                    "text_evidence_score": max(t["region_evidence_score"] for _, t in matched_text),
                    "associated_ocr_text": combined_text,
                    "explanation": "High-confidence multi-specialist evidence detected (both physical and text-region visual patterns present).",
                })
            else:
                fused.append({
                    "region_id": phys["region_id"],
                    "bbox": phys["bbox"],
                    "evidence_sources": [phys["evidence_source"]],
                    "physical_evidence_score": phys["region_evidence_score"],
                    "text_evidence_score": None,
                    "associated_ocr_text": "",
                    "explanation": phys["explanation"],
                })

        # Add remaining text-only regions
        for t_idx, txt in enumerate(text_regions):
            if t_idx not in used_text_indices:
                fused.append({
                    "region_id": txt["region_id"],
                    "bbox": txt["bbox"],
                    "evidence_sources": [txt["evidence_source"]],
                    "physical_evidence_score": None,
                    "text_evidence_score": txt["region_evidence_score"],
                    "associated_ocr_text": txt.get("associated_ocr_text", ""),
                    "explanation": txt["explanation"],
                })

        return fused

    def analyze(
        self,
        image_rgb: np.ndarray,
        ocr_results: Optional[List[Dict[str, Any]]] = None,
        content_consistency_findings: Optional[List[Dict[str, Any]]] = None,
        reference_findings: Optional[List[Dict[str, Any]]] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        """Execute complete multi-evidence analysis on input document image."""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        h_orig, w_orig = image_rgb.shape[:2]

        # 1. Preprocess input for neural models (512x512, ImageNet normalized)
        img_512 = cv2.resize(image_rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
        img_norm = (img_512.astype(np.float32) / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        tensor_in = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

        # 2. Run Model A (Physical Forensics)
        phys_prob_map = np.zeros((512, 512), dtype=np.float32)
        if self.model_a is not None:
            self.model_a.eval()
            with torch.no_grad():
                with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    out_a = self.model_a(tensor_in)
                phys_prob_map = torch.sigmoid(out_a)[0, 0].cpu().numpy()

        # 3. Run Model B (Tiny-Text Forensics)
        text_prob_map = np.zeros((512, 512), dtype=np.float32)
        if self.model_b is not None:
            self.model_b.eval()
            with torch.no_grad():
                with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    out_b = self.model_b(tensor_in)
                text_prob_map = torch.sigmoid(out_b)[0, 0].cpu().numpy()

        # 4. Scale probability maps and OCR to original document resolution
        scale_x = w_orig / 512.0
        scale_y = h_orig / 512.0

        phys_prob_orig = cv2.resize(phys_prob_map, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
        text_prob_orig = cv2.resize(text_prob_map, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

        # 5. Extract regions
        physical_regions = self.extract_physical_regions(phys_prob_orig, threshold=0.50)
        text_regions = self.postprocessor_b.process(text_prob_orig, ocr_results=ocr_results)

        # 6. Fuse overlapping visual evidence
        fused_visual_regions = self.fuse_overlapping_regions(physical_regions, text_regions)

        # 7. Aggregate evidence channels
        has_physical_ev = len(physical_regions) > 0
        has_text_ev = len(text_regions) > 0
        has_content_inconsistency = bool(content_consistency_findings and len(content_consistency_findings) > 0)
        has_ref_mismatch = bool(reference_findings and len(reference_findings) > 0)

        # Conservative summary statement
        if has_physical_ev or has_text_ev or has_content_inconsistency or has_ref_mismatch:
            status_msg = "Potential visual manipulation evidence or content inconsistency detected."
        else:
            status_msg = "No significant visual manipulation evidence detected."

        return {
            "visual_forensics": {
                "physical_regions": physical_regions,
                "text_regions": text_regions,
                "fused_regions": fused_visual_regions,
                "physical_evidence_detected": has_physical_ev,
                "text_evidence_detected": has_text_ev,
            },
            "content_consistency": {
                "findings": content_consistency_findings or [],
                "inconsistency_detected": has_content_inconsistency,
            },
            "reference_verification": {
                "available": reference_findings is not None,
                "findings": reference_findings or [],
                "mismatch_detected": has_ref_mismatch,
            },
            "summary": {
                "physical_visual_evidence": has_physical_ev,
                "digital_text_visual_evidence": has_text_ev,
                "content_inconsistency": has_content_inconsistency,
                "reference_mismatch": has_ref_mismatch,
                "status_message": status_msg,
                "disclaimer": "These findings indicate potential visual or content inconsistencies and should not be interpreted as definitive proof of document authenticity or fraud.",
            },
        }
