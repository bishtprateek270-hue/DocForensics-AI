"""
DocForensics AI — Real-World Augmentation Visualizer (Phase 4)
Renders side-by-side multi-sample verification grid:
ORIGINAL → AUGMENTED → AUGMENTED MASK → OVERLAY

Allows visual verification of:
1. Realistic real-world distortions (JPEG, blur, noise, shadow, perspective, scan)
2. Exact geometric alignment of document content and ground-truth mask
3. Binary mask preservation ({0, 255})
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import random
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from config import BASE_DIR, METADATA_CSV, OUTPUT_DIR, cfg
from src.preprocessing.augmentations import (
    get_train_augmentation_pipeline,
    apply_augmentation,
)
from src.preprocessing.tampering_engine import MANIPULATION_CATEGORIES


def create_forensic_overlay(image_rgb: np.ndarray, mask_gray: np.ndarray) -> np.ndarray:
    """Creates semi-transparent red forensic overlay with contour borders."""
    overlay = image_rgb.copy()
    if mask_gray.max() == 0:
        cv2.putText(overlay, "AUTHENTIC (CLEAN)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (22, 163, 74), 2)
        return overlay

    red_mask = np.zeros_like(image_rgb)
    red_mask[:, :] = (220, 38, 38)

    tampered_pixels = mask_gray > 0
    alpha = 0.55
    overlay[tampered_pixels] = (
        alpha * red_mask[tampered_pixels] + (1 - alpha) * image_rgb[tampered_pixels]
    ).astype(np.uint8)

    contours, _ = cv2.findContours(mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (220, 38, 38), 2)
    cv2.putText(overlay, "TAMPERED OVERLAY", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (220, 38, 38), 2)
    return overlay


def visualize_augmented_samples(
    metadata_path: Path = METADATA_CSV,
    output_path: Path = OUTPUT_DIR / "sample_augmentation_verification.png",
    num_samples: int = 8,
    seed: int = 42,
) -> Path:
    """
    Renders 4-column visual verification grid:
    1. ORIGINAL (Clean Master Document)
    2. AUGMENTED (Simulated Real-World Document with JPEG/Noise/Shadow/Perspective)
    3. AUGMENTED MASK (Pixel-Aligned Binary Mask)
    4. OVERLAY (Forensic Highlight on Augmented Image)
    """
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file {metadata_path} not found.")

    df = pd.read_csv(metadata_path)
    random.seed(seed)
    np.random.seed(seed)

    # Sample balanced set across manipulation categories + authentic
    sampled_records = []
    syn_tampered = df[(df["source_type"] == "synthetic") & (df["is_tampered"] == 1)]
    
    # Pick 6 distinct manipulation types
    chosen_categories = MANIPULATION_CATEGORIES[:6]
    for cat in chosen_categories:
        cat_matches = syn_tampered[syn_tampered["manipulation_type"] == cat]
        if len(cat_matches) > 0:
            sampled_records.append(cat_matches.sample(1, random_state=seed).iloc[0])

    # 1 authentic sample
    auth_matches = df[df["is_tampered"] == 0]
    if len(auth_matches) > 0:
        sampled_records.append(auth_matches.sample(1, random_state=seed).iloc[0])

    # 1 public sample
    pub_matches = df[df["source_type"] == "public"]
    if len(pub_matches) > 0:
        sampled_records.append(pub_matches.sample(1, random_state=seed).iloc[0])

    sampled_df = pd.DataFrame(sampled_records)
    pipeline = get_train_augmentation_pipeline(image_size=(512, 512))

    rows = len(sampled_df)
    cols = 4  # [Original, Augmented, Augmented Mask, Overlay]
    fig, axes = plt.subplots(rows, cols, figsize=(22, 5.2 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, 0)

    fig.suptitle(
        "DocForensics AI — Phase 4 Real-World Data Augmentation Verification Grid\n"
        "[ORIGINAL → AUGMENTED → AUGMENTED MASK → OVERLAY]",
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )

    col_titles = [
        "1. ORIGINAL (Master Image)",
        "2. AUGMENTED (JPEG / Noise / Shadow / Perspective / Scan)",
        "3. AUGMENTED MASK (Aligned Binary Mask {0, 255})",
        "4. OVERLAY (Forensic Ground-Truth Highlight)",
    ]

    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title, fontsize=13, fontweight="bold", pad=12)

    for r_idx, (_, row) in enumerate(sampled_df.iterrows()):
        orig_p = BASE_DIR / str(row.get("original_path", ""))
        tamp_p = BASE_DIR / str(row.get("tampered_path", row.get("image_path", "")))
        mask_p = BASE_DIR / str(row["mask_path"])

        if orig_p.exists():
            orig_bgr = cv2.imread(str(orig_p))
            orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
            orig_rgb = cv2.resize(orig_rgb, (512, 512))
        else:
            orig_rgb = np.zeros((512, 512, 3), dtype=np.uint8)

        tamp_bgr = cv2.imread(str(tamp_p))
        tamp_rgb = cv2.cvtColor(tamp_bgr, cv2.COLOR_BGR2RGB)
        mask_gray = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)

        # Apply augmentation with identical geometric warp on image and mask
        aug_rgb, aug_mask = apply_augmentation(
            tamp_rgb,
            mask_gray,
            pipeline=pipeline,
            is_training=True,
            image_size=(512, 512),
        )

        overlay = create_forensic_overlay(aug_rgb, aug_mask)

        # Plot 1: Original
        axes[r_idx, 0].imshow(orig_rgb)
        axes[r_idx, 0].axis("off")
        orig_txt = f"ID: {row['source_document_id']}\nType: {row.get('doc_type', 'N/A')}"
        axes[r_idx, 0].text(10, 490, orig_txt, color="white", fontsize=8.5,
                            bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.75))

        # Plot 2: Augmented
        axes[r_idx, 1].imshow(aug_rgb)
        axes[r_idx, 1].axis("off")
        tamp_txt = f"Manipulation: {row['manipulation_type'].upper()}\nAugmentation: Real-World Degradation"
        axes[r_idx, 1].text(10, 490, tamp_txt, color="yellow", fontsize=8.5,
                            bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.75))

        # Plot 3: Mask
        axes[r_idx, 2].imshow(aug_mask, cmap="gray", vmin=0, vmax=255)
        axes[r_idx, 2].axis("off")
        pos_pix = int(np.sum(aug_mask > 0))
        area_pct = float(np.round((pos_pix / (512 * 512)) * 100, 2))
        mask_txt = f"Tampered Pixels: {pos_pix:,} ({area_pct}%)\nValues: [{aug_mask.min()}, {aug_mask.max()}]"
        axes[r_idx, 2].text(10, 490, mask_txt, color="cyan", fontsize=8.5,
                            bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.75))

        # Plot 4: Overlay
        axes[r_idx, 3].imshow(overlay)
        axes[r_idx, 3].axis("off")
        over_txt = f"Aligned Contours: {len(cv2.findContours(aug_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])} region(s)"
        axes[r_idx, 3].text(10, 490, over_txt, color="white", fontsize=8.5,
                            bbox=dict(boxstyle="round,pad=0.3", fc="darkred", alpha=0.75))

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()

    print(f"[+] Real-world augmentation verification grid saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    visualize_augmented_samples()
