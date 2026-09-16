"""
DocForensics AI — OCR Engine Module (Phase 8)
Extracts text, bounding boxes, and recognition confidence scores from document images.
Supports PyTorch-accelerated EasyOCR with GPU/CPU support and PyMuPDF fallback.
"""

import os
import sys
import time
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False


class OCREntry:
    """Represents a single detected text unit with coordinates and confidence."""

    def __init__(self, text: str, bbox: List[int], confidence: float):
        self.text = text.strip()
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.confidence = round(float(confidence), 4)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "bbox": self.bbox,
            "confidence": self.confidence,
        }

    def __repr__(self) -> str:
        return f"OCREntry(text='{self.text}', bbox={self.bbox}, conf={self.confidence:.2f})"


class OCREngine:
    """
    Modular OCR Extractor for Document Forensics.
    """

    def __init__(self, lang_list: Optional[List[str]] = None, use_gpu: bool = True):
        self.lang_list = lang_list or ["en"]
        self.use_gpu = use_gpu and EASYOCR_AVAILABLE
        self.reader = None

        if EASYOCR_AVAILABLE:
            try:
                # Initialize EasyOCR reader (loads CRAFT text detector + CRNN recognizer)
                self.reader = easyocr.Reader(
                    self.lang_list,
                    gpu=self.use_gpu,
                    verbose=False,
                )
            except Exception as e:
                print(f"[!] Warning: Failed to initialize EasyOCR on GPU ({e}), falling back to CPU.")
                try:
                    self.reader = easyocr.Reader(self.lang_list, gpu=False, verbose=False)
                    self.use_gpu = False
                except Exception as e2:
                    print(f"[!] EasyOCR CPU fallback failed: {e2}")
                    self.reader = None

    def extract_text(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image],
        min_confidence: float = 0.20,
    ) -> Tuple[List[OCREntry], float]:
        """
        Extracts OCR text bounding boxes from an image.
        
        Args:
            image_input: Filepath, NumPy array (RGB or BGR), or PIL Image.
            min_confidence: Threshold to discard low-confidence noisy detections.
            
        Returns:
            entries: List of OCREntry objects with [x1, y1, x2, y2] bboxes.
            latency_ms: OCR execution duration in milliseconds.
        """
        t0 = time.perf_counter()

        # Normalize input to uint8 RGB numpy array
        if isinstance(image_input, (str, Path)):
            path_str = str(image_input)
            if not os.path.exists(path_str):
                raise FileNotFoundError(f"Image not found at {path_str}")
            img_bgr = cv2.imread(path_str)
            if img_bgr is None:
                raise ValueError(f"Failed to read image at {path_str}")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, Image.Image):
            img_rgb = np.array(image_input.convert("RGB"))
        elif isinstance(image_input, np.ndarray):
            if image_input.dtype != np.uint8:
                if image_input.max() <= 1.0:
                    img_rgb = (image_input * 255.0).clip(0, 255).astype(np.uint8)
                else:
                    img_rgb = image_input.clip(0, 255).astype(np.uint8)
            else:
                img_rgb = image_input
            if len(img_rgb.shape) == 2:
                img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        entries: List[OCREntry] = []

        if self.reader is not None:
            results = self.reader.readtext(img_rgb)
            for item in results:
                # EasyOCR returns: [box_points, text, confidence]
                # box_points is [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                pts, text, conf = item
                if conf < min_confidence or not text.strip():
                    continue

                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                x1, x2 = int(min(xs)), int(max(xs))
                y1, y2 = int(min(ys)), int(max(ys))

                # Boundary clamping
                h, w = img_rgb.shape[:2]
                x1 = max(0, min(w - 1, x1))
                x2 = max(0, min(w - 1, x2))
                y1 = max(0, min(h - 1, y1))
                y2 = max(0, min(h - 1, y2))

                if x2 > x1 and y2 > y1:
                    entries.append(OCREntry(text=text, bbox=[x1, y1, x2, y2], confidence=conf))

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return entries, latency_ms


def get_ocr_engine(use_gpu: bool = True) -> OCREngine:
    """Factory builder for OCREngine."""
    return OCREngine(use_gpu=use_gpu)
