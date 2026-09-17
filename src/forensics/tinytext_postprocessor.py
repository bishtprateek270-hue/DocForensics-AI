"""Tiny-Text Forensic Post-Processing and False-Positive Reduction Pipeline.

Applies:
1. Connected-component extraction and feature profiling
2. Minimum component-size filtering (removes isolated sub-15px noise)
3. OCR-aware text-line association and spatial constraint filtering
4. Horizontal/vertical spatial region merging along same OCR line
5. Region Evidence Score computation
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class TinyTextPostProcessor:
    """Dedicated post-processing pipeline for Tiny-Text Digital Forensic detections."""

    def __init__(
        self,
        threshold: float = 0.50,
        min_component_area: int = 15,
        ocr_expansion_ratio: float = 0.15,
        max_ocr_distance: float = 35.0,
        merge_horizontal_distance: int = 30,
        merge_vertical_distance: int = 10,
        require_ocr_association: bool = True,
    ) -> None:
        self.threshold = threshold
        self.min_component_area = min_component_area
        self.ocr_expansion_ratio = ocr_expansion_ratio
        self.max_ocr_distance = max_ocr_distance
        self.merge_horizontal_distance = merge_horizontal_distance
        self.merge_vertical_distance = merge_vertical_distance
        self.require_ocr_association = require_ocr_association

    def _build_ocr_text_line_regions(
        self, ocr_results: List[Dict[str, Any]], image_shape: Tuple[int, int]
    ) -> List[Dict[str, Any]]:
        """Construct expanded text-line bounding boxes from OCR tokens."""
        h, w = image_shape[:2]
        text_lines = []

        for ocr_item in ocr_results:
            bbox = ocr_item.get("bbox", [])
            text = ocr_item.get("text", "")
            if len(bbox) == 4:
                bx, by, bw, bh = bbox
            elif len(bbox) == 8:  # 4 corner points
                xs = bbox[0::2]
                ys = bbox[1::2]
                bx, by = min(xs), min(ys)
                bw, bh = max(xs) - bx, max(ys) - by
            else:
                continue

            # Expand box by margin
            exp_w = int(bw * self.ocr_expansion_ratio)
            exp_h = int(bh * self.ocr_expansion_ratio)
            x1 = max(0, bx - exp_w)
            y1 = max(0, by - exp_h)
            x2 = min(w, bx + bw + exp_w)
            y2 = min(h, by + bh + exp_h)

            text_lines.append({
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "text": text,
                "confidence": ocr_item.get("confidence", 1.0),
            })

        return text_lines

    def _extract_raw_components(
        self, prob_map: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Extract connected components passing binary threshold."""
        binary = (prob_map >= self.threshold).astype(np.uint8) * 255
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        components = []

        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area < self.min_component_area:
                continue

            comp_mask = (labels == i)
            comp_probs = prob_map[comp_mask]

            components.append({
                "raw_id": i,
                "bbox": [int(x), int(y), int(w), int(h)],
                "area_pixels": int(area),
                "mean_prob": float(np.mean(comp_probs)),
                "max_prob": float(np.max(comp_probs)),
                "p90_prob": float(np.percentile(comp_probs, 90)),
                "centroid": [float(centroids[i][0]), float(centroids[i][1])],
            })

        return components

    def _associate_with_ocr(
        self,
        components: List[Dict[str, Any]],
        text_lines: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Filter and enrich components with associated OCR text lines."""
        associated = []

        for comp in components:
            cx, cy, cw, ch = comp["bbox"]
            center_x, center_y = comp["centroid"]

            matched_line = None
            min_dist = float("inf")

            for line in text_lines:
                lx, ly, lw, lh = line["bbox"]

                # Check bounding box overlap
                ix1 = max(cx, lx)
                iy1 = max(cy, ly)
                ix2 = min(cx + cw, lx + lw)
                iy2 = min(cy + ch, ly + lh)
                inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)

                if inter_area > 0:
                    matched_line = line
                    min_dist = 0.0
                    break

                # Measure distance from component center to line box
                dx = max(0, max(lx - center_x, center_x - (lx + lw)))
                dy = max(0, max(ly - center_y, center_y - (ly + lh)))
                dist = float(np.sqrt(dx * dx + dy * dy))

                if dist < min_dist:
                    min_dist = dist
                    if dist <= self.max_ocr_distance:
                        matched_line = line

            if self.require_ocr_association and len(text_lines) > 0:
                if matched_line is None or min_dist > self.max_ocr_distance:
                    # Suppress isolated non-text prediction
                    continue

            comp_copy = dict(comp)
            comp_copy["associated_ocr_text"] = matched_line["text"] if matched_line else ""
            comp_copy["ocr_distance"] = round(min_dist, 1)
            associated.append(comp_copy)

        return associated

    def _merge_adjacent_regions(
        self, components: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Merge components on the same text line separated by small character gaps."""
        if not components:
            return []

        if len(components) == 1:
            c = components[0]
            ocr_text = c.get("associated_ocr_text", "")
            mean_prob = c.get("mean_prob", 0.5)
            max_prob = c.get("max_prob", 0.5)
            evidence_score = round(0.50 * mean_prob + 0.30 * max_prob + 0.20 * (1.0 if ocr_text else 0.5), 4)
            return [{
                "bbox": c["bbox"],
                "area_pixels": c["area_pixels"],
                "mean_prob": round(mean_prob, 4),
                "max_prob": round(max_prob, 4),
                "region_evidence_score": evidence_score,
                "associated_ocr_text": ocr_text,
                "evidence_source": "Tiny-Text Digital Forensic Model",
                "explanation": "Suspicious text-region visual pattern detected.",
            }]

        # Group components by horizontal/vertical proximity
        merged: List[Dict[str, Any]] = []
        visited = [False] * len(components)

        for i in range(len(components)):
            if visited[i]:
                continue
            group = [components[i]]
            visited[i] = True

            for j in range(i + 1, len(components)):
                if visited[j]:
                    continue
                c1 = components[i]
                c2 = components[j]

                # Check proximity
                x1, y1, w1, h1 = c1["bbox"]
                x2, y2, w2, h2 = c2["bbox"]

                v_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                h_dist = max(0, max(x1, x2) - min(x1 + w1, x2 + w2))
                v_dist = max(0, max(y1, y2) - min(y1 + h1, y2 + h2))

                same_line = (v_overlap > 0 or v_dist <= self.merge_vertical_distance)
                near_horiz = (h_dist <= self.merge_horizontal_distance)

                if same_line and near_horiz:
                    group.append(c2)
                    visited[j] = True

            # Merge all in group
            min_x = min(c["bbox"][0] for c in group)
            min_y = min(c["bbox"][1] for c in group)
            max_x = max(c["bbox"][0] + c["bbox"][2] for c in group)
            max_y = max(c["bbox"][1] + c["bbox"][3] for c in group)
            total_area = sum(c["area_pixels"] for c in group)
            mean_prob = float(np.mean([c["mean_prob"] for c in group]))
            max_prob = float(np.max([c["max_prob"] for c in group]))

            # Aggregate associated text
            texts = [c["associated_ocr_text"] for c in group if c.get("associated_ocr_text")]
            ocr_text = " ".join(list(dict.fromkeys(texts)))

            # Calculate Region Evidence Score
            evidence_score = round(0.50 * mean_prob + 0.30 * max_prob + 0.20 * (1.0 if ocr_text else 0.5), 4)

            merged.append({
                "bbox": [min_x, min_y, max_x - min_x, max_y - min_y],
                "area_pixels": total_area,
                "mean_prob": round(mean_prob, 4),
                "max_prob": round(max_prob, 4),
                "region_evidence_score": evidence_score,
                "associated_ocr_text": ocr_text,
                "evidence_source": "Tiny-Text Digital Forensic Model",
                "explanation": "Suspicious text-region visual pattern detected.",
            })

        return merged

    def process(
        self,
        prob_map: np.ndarray,
        ocr_results: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Run complete post-processing pipeline on Tiny-Text probability map.

        Args:
            prob_map: [H, W] float array in [0.0, 1.0].
            ocr_results: Optional list of OCR detections with 'bbox' and 'text'.

        Returns:
            List of filtered, merged suspicious text regions.
        """
        if ocr_results is None:
            ocr_results = []

        # 1. Connected components above threshold
        components = self._extract_raw_components(prob_map)
        if not components:
            return []

        # 2. OCR text-line region construction
        text_lines = self._build_ocr_text_line_regions(ocr_results, prob_map.shape)

        # 3. OCR text-line association & spatial constraint filtering
        associated_comps = self._associate_with_ocr(components, text_lines)
        if not associated_comps:
            return []

        # 4. Spatial region merging
        final_regions = self._merge_adjacent_regions(associated_comps)

        # 5. Number regions
        for idx, r in enumerate(final_regions, 1):
            r["region_id"] = f"text_region_{idx}"

        return final_regions
