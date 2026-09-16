"""
DocForensics AI — PDF Rendering & Page Management (Phase 9)
Safely renders individual pages from PDF documents into high-resolution RGB NumPy arrays and PNGs.
Preserves page geometry and coordinate mappings for the downstream Phase 8 forensic pipeline.
"""

import io
from typing import Tuple, List, Optional
import fitz  # PyMuPDF
import numpy as np
from PIL import Image


def get_pdf_page_count(pdf_bytes: bytes) -> int:
    """Returns total number of pages in a PDF document."""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        count = len(doc)
        doc.close()
        return max(1, count)
    except Exception as e:
        raise ValueError(f"Invalid or corrupted PDF file: {e}")


def render_pdf_page_to_rgb(
    pdf_bytes: bytes,
    page_number: int = 1,
    dpi: int = 150,
) -> Tuple[np.ndarray, int, int, int]:
    """
    Renders a specific PDF page to an RGB NumPy array.
    
    Args:
        pdf_bytes: Raw binary content of the PDF.
        page_number: 1-indexed page number to render.
        dpi: Rendering resolution (default 150 DPI for sharp document text).
        
    Returns:
        img_rgb: [H, W, 3] uint8 RGB image array.
        total_pages: Total number of pages in the PDF document.
        width: Rendered width in pixels.
        height: Rendered height in pixels.
    """
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = len(doc)
        if total_pages == 0:
            raise ValueError("PDF document contains 0 pages.")

        page_idx = max(0, min(total_pages - 1, page_number - 1))
        page = doc[page_idx]

        # Calculate matrix scaling for requested DPI (72 DPI is 1.0x)
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)

        # Convert pixmap to RGB numpy array
        img_rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, 3))
        h, w = img_rgb.shape[:2]
        doc.close()

        return img_rgb, total_pages, w, h
    except Exception as e:
        raise ValueError(f"Failed to render PDF page {page_number}: {e}")
