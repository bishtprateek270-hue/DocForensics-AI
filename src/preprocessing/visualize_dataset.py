"""
DocForensics AI — Multi-Panel Forensic Dataset Visualizer (Phase 3)
Renders side-by-side forensic verification grids:
1. ORIGINAL (Clean Source Document)
2. TAMPERED (Manipulated Document Image)
3. GROUND-TRUTH MASK (Binary Tampering Mask, 0=untouched, 255=tampered)
4. MASK OVERLAY (Semi-transparent red forensic highlight with contour borders)

Saves verification grid to outputs/sample_dataset_verification.png.
"""

import ast
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import cv2

from config import BASE_DIR, METADATA_CSV, OUTPUT_DIR, cfg
from src.preprocessing.tampering_engine import MANIPULATION_CATEGORIES


def create_forensic_overlay(image_arr: np.ndarray, mask_arr: np.ndarray, bbox: list = None) -> np.ndarray:
    """Overlay semi-transparent red highlight over tampered regions with contour boundaries."""
    overlay = image_arr.copy()
    if mask_arr.max() == 0:
        cv2.putText(overlay, "AUTHENTIC", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (22, 163, 74), 3)
        return overlay

    red_mask = np.zeros_like(image_arr)
    red_mask[:, :] = (220, 38, 38)

    tampered_pixels = mask_arr > 0
    alpha = 0.55
    overlay[tampered_pixels] = (
        alpha * red_mask[tampered_pixels] + (1 - alpha) * image_arr[tampered_pixels]
    ).astype(np.uint8)

    contours, _ = cv2.findContours(mask_arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (220, 38, 38), 3)
    
    if bbox and len(bbox) == 4 and any(b > 0 for b in bbox):
        bx1, by1, bx2, by2 = bbox
        cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (255, 165, 0), 2)

    cv2.putText(overlay, "TAMPERED", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (220, 38, 38), 3)
    return overlay


def visualize_dataset_samples(
    metadata_path: Path = METADATA_CSV,
    output_path: Path = OUTPUT_DIR / "sample_dataset_verification.png",
    num_samples: int = 10,
    seed: int = 42,
) -> Path:
    """
    Renders 4-panel visual verification grid:
    ORIGINAL | TAMPERED | GROUND-TRUTH MASK | MASK OVERLAY
    covering diverse manipulation categories.
    """
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file {metadata_path} not found. Run dataset builder first.")

    df = pd.read_csv(metadata_path)
    random.seed(seed)

    # Pick samples across different manipulation categories
    sampled_records = []
    syn_tampered = df[(df["source_type"] == "synthetic") & (df["is_tampered"] == 1)]
    
    for cat in MANIPULATION_CATEGORIES:
        cat_matches = syn_tampered[syn_tampered["manipulation_type"] == cat]
        if len(cat_matches) > 0:
            sampled_records.append(cat_matches.sample(1, random_state=seed).iloc[0])

    # Also add 1 authentic sample and 1 public sample
    syn_auth = df[(df["source_type"] == "synthetic") & (df["is_tampered"] == 0)]
    if len(syn_auth) > 0:
        sampled_records.append(syn_auth.sample(1, random_state=seed).iloc[0])

    pub_samples = df[df["source_type"] == "public"]
    if len(pub_samples) > 0:
        sampled_records.append(pub_samples.sample(1, random_state=seed).iloc[0])

    sampled_df = pd.DataFrame(sampled_records)

    rows = len(sampled_df)
    cols = 4  # [ORIGINAL, TAMPERED, GROUND-TRUTH MASK, MASK OVERLAY]
    fig, axes = plt.subplots(rows, cols, figsize=(22, 5.0 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, 0)

    fig.suptitle(
        "DocForensics AI — Phase 3 Synthetic & Public Tampering Verification Grid\n"
        "[ORIGINAL | TAMPERED | GROUND-TRUTH MASK | MASK OVERLAY]",
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )

    col_titles = [
        "1. ORIGINAL (Clean Source)",
        "2. TAMPERED (Manipulated Image)",
        "3. GROUND-TRUTH MASK (0=Clean, 255=Tampered)",
        "4. FORENSIC MASK OVERLAY",
    ]

    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title, fontsize=13, fontweight="bold", pad=12)

    for r_idx, (_, row) in enumerate(sampled_df.iterrows()):
        orig_p = BASE_DIR / str(row.get("original_path", ""))
        tamp_p = BASE_DIR / str(row.get("tampered_path", row.get("image_path", "")))
        mask_p = BASE_DIR / str(row["mask_path"])

        # 1. Original Image
        if orig_p.exists():
            orig_bgr = cv2.imread(str(orig_p))
            orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
        else:
            orig_rgb = np.zeros((600, 800, 3), dtype=np.uint8)

        # 2. Tampered Image
        tamp_bgr = cv2.imread(str(tamp_p))
        tamp_rgb = cv2.cvtColor(tamp_bgr, cv2.COLOR_BGR2RGB)

        # 3. Ground-Truth Mask
        mask_gray = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)

        # Bounding box
        bbox = None
        if "bounding_box" in row and pd.notna(row["bounding_box"]):
            try:
                b_val = row["bounding_box"]
                bbox = ast.literal_eval(b_val) if isinstance(b_val, str) else b_val
            except Exception:
                bbox = None

        # 4. Forensic Overlay
        overlay = create_forensic_overlay(tamp_rgb, mask_gray, bbox=bbox)

        # Plot 1: ORIGINAL
        axes[r_idx, 0].imshow(orig_rgb)
        axes[r_idx, 0].axis("off")
        orig_label = f"DOC ID: {row['source_document_id']}\nType: {row.get('doc_type', 'N/A')}"
        axes[r_idx, 0].text(10, orig_rgb.shape[0] - 20, orig_label, color="white", fontsize=8.5,
                            bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.75))

        # Plot 2: TAMPERED
        axes[r_idx, 1].imshow(tamp_rgb)
        axes[r_idx, 1].axis("off")
        tamp_label = f"MANIPULATION: {row['manipulation_type'].upper()}\nSplit: {row.get('dataset_split', 'train').upper()}"
        axes[r_idx, 1].text(10, tamp_rgb.shape[0] - 20, tamp_label, color="yellow", fontsize=8.5,
                            bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.75))

        # Plot 3: MASK
        axes[r_idx, 2].imshow(mask_gray, cmap="gray", vmin=0, vmax=255)
        axes[r_idx, 2].axis("off")
        pix_cnt = row.get("tampered_pixel_count", int(np.sum(mask_gray > 0)))
        area_pct = row.get("tampered_area_percentage", 0.0)
        mask_label = f"Pixels: {pix_cnt:,} ({area_pct:.2f}%)\nValues: [{mask_gray.min()}, {mask_gray.max()}]"
        axes[r_idx, 2].text(10, mask_gray.shape[0] - 20, mask_label, color="cyan", fontsize=8.5,
                            bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.75))

        # Plot 4: OVERLAY
        axes[r_idx, 3].imshow(overlay)
        axes[r_idx, 3].axis("off")
        over_label = f"BBox: {bbox if bbox else 'N/A'}"
        axes[r_idx, 3].text(10, overlay.shape[0] - 20, over_label, color="white", fontsize=8.5,
                            bbox=dict(boxstyle="round,pad=0.3", fc="darkred", alpha=0.75))

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()

    print(f"[+] Multi-panel forensic verification grid saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    visualize_dataset_samples()
