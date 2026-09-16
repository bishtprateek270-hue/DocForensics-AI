"""
DocForensics AI — Multi-Source Forensic Visualizer
Renders side-by-side verification figures across both Public (RealText-V2) and Synthetic benchmarks:
1. Document Image (RGB)
2. Ground-Truth Binary Mask
3. Forensic Tampering Overlay
4. Error Level Analysis (ELA)
Saves visual inspection grid to outputs/sample_dataset_verification.png.
"""

import io
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageChops, ImageEnhance
import cv2

from config import BASE_DIR, METADATA_CSV, OUTPUT_DIR, cfg


def compute_ela_image(image_path: Path, quality: int = 90, scale: float = 15.0) -> np.ndarray:
    """Compute Error Level Analysis (ELA) image to highlight compression discrepancies."""
    orig = Image.open(image_path).convert("RGB")
    buf = io.BytesIO()
    orig.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    resaved = Image.open(buf)

    diff = ImageChops.difference(orig, resaved)
    extrema = diff.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale_factor = 255.0 / max_diff * (scale / 10.0)
    diff = ImageEnhance.Brightness(diff).enhance(scale_factor)
    return np.array(diff)


def create_forensic_overlay(image_arr: np.ndarray, mask_arr: np.ndarray) -> np.ndarray:
    """Overlay semi-transparent red highlight over tampered regions."""
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
    cv2.putText(overlay, "TAMPERED", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (220, 38, 38), 3)
    return overlay


def visualize_dataset_samples(
    metadata_path: Path = METADATA_CSV,
    output_path: Path = OUTPUT_DIR / "sample_dataset_verification.png",
    seed: int = 42,
) -> Path:
    """Sample documents from both public and synthetic sources and render verification grid."""
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file {metadata_path} not found. Run dataset builder first.")

    df = pd.read_csv(metadata_path)
    random.seed(seed)

    # Sample balanced set:
    # 2 Public Tampered + 1 Public Authentic + 2 Synthetic Tampered + 1 Synthetic Authentic
    pub_tamp = df[(df["data_source"] == "public_realtext_v2") & (df["is_tampered"] == 1)].sample(min(2, len(df[(df["data_source"] == "public_realtext_v2") & (df["is_tampered"] == 1)])), random_state=seed)
    pub_auth = df[(df["data_source"] == "public_realtext_v2") & (df["is_tampered"] == 0)].sample(min(1, len(df[(df["data_source"] == "public_realtext_v2") & (df["is_tampered"] == 0)])), random_state=seed)
    syn_tamp = df[(df["data_source"] == "synthetic_docforensics") & (df["is_tampered"] == 1)].sample(min(2, len(df[(df["data_source"] == "synthetic_docforensics") & (df["is_tampered"] == 1)])), random_state=seed)
    syn_auth = df[(df["data_source"] == "synthetic_docforensics") & (df["is_tampered"] == 0)].sample(min(1, len(df[(df["data_source"] == "synthetic_docforensics") & (df["is_tampered"] == 0)])), random_state=seed)

    sampled_df = pd.concat([pub_tamp, pub_auth, syn_tamp, syn_auth]).reset_index(drop=True)

    rows = len(sampled_df)
    cols = 4  # [Original, Mask, Forensic Overlay, ELA Residual]
    fig, axes = plt.subplots(rows, cols, figsize=(20, 4.5 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, 0)

    fig.suptitle("DocForensics AI — Public & Synthetic Dataset Forensic Inspection", fontsize=18, fontweight="bold", y=0.995)

    col_titles = [
        "1. Document Image (RGB)",
        "2. Ground-Truth Binary Mask",
        "3. Forensic Tamper Overlay",
        "4. Error Level Analysis (ELA)",
    ]

    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title, fontsize=13, fontweight="bold", pad=12)

    for r_idx, row in sampled_df.iterrows():
        img_p = BASE_DIR / row["image_path"]
        mask_p = BASE_DIR / row["mask_path"]

        img_bgr = cv2.imread(str(img_p))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mask_gray = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)

        overlay = create_forensic_overlay(img_rgb, mask_gray)
        ela_img = compute_ela_image(img_p)

        # Plot 1: RGB
        axes[r_idx, 0].imshow(img_rgb)
        axes[r_idx, 0].axis("off")
        src_label = f"SOURCE: {row['data_source'].upper()} | Split: {row['split'].upper()}\nID: {row['sample_id']}"
        axes[r_idx, 0].text(10, img_rgb.shape[0] - 20, src_label, color="white", fontsize=8.5,
                            bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.75))

        # Plot 2: Mask
        axes[r_idx, 1].imshow(mask_gray, cmap="gray", vmin=0, vmax=255)
        axes[r_idx, 1].axis("off")
        tech_lbl = f"Technique: {row['tampering_type'].upper()}" if row['is_tampered'] == 1 else "AUTHENTIC (Zero Mask)"
        axes[r_idx, 1].text(10, mask_gray.shape[0] - 20, tech_lbl, color="cyan", fontsize=8.5,
                            bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.75))

        # Plot 3: Overlay
        axes[r_idx, 2].imshow(overlay)
        axes[r_idx, 2].axis("off")

        # Plot 4: ELA
        axes[r_idx, 3].imshow(ela_img)
        axes[r_idx, 3].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()

    print(f"[+] Multi-source forensic verification visual saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    visualize_dataset_samples()
