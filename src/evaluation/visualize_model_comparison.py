"""
DocForensics AI — Multi-Model Prediction & Metric Comparison Visualizer (Phase 6)
Renders:
1. outputs/model_comparison_predictions.png: 5-column visual inspection grid
   [ Document | Ground Truth | U-Net Prediction | DeepLabV3+ Prediction | SegFormer Prediction ]
2. outputs/model_comparison_curves.png: Comparative validation Dice and loss curves over epochs
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from PIL import Image

from config import BASE_DIR, CHECKPOINT_DIR, OUTPUT_DIR, REPORTS_DIR, METADATA_CSV, cfg, get_target_device
from src.models.unet import get_unet_model
from src.models.deeplabv3plus import get_deeplabv3plus_model
from src.models.segformer import get_segformer_model
from src.preprocessing.dataset import DocForensicsDataset


def create_colored_overlay(image_rgb: np.ndarray, mask_bin: np.ndarray, color: tuple = (220, 38, 38)) -> np.ndarray:
    """Overlays colored highlight over detected tampering region."""
    overlay = image_rgb.copy()
    tampered_pixels = mask_bin > 0
    if not np.any(tampered_pixels):
        return overlay

    alpha = 0.55
    color_mask = np.zeros_like(image_rgb)
    color_mask[:, :] = color
    overlay[tampered_pixels] = (
        alpha * color_mask[tampered_pixels] + (1 - alpha) * image_rgb[tampered_pixels]
    ).astype(np.uint8)

    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, 2)
    return overlay


def plot_model_comparison_curves(output_path: Path = OUTPUT_DIR / "model_comparison_curves.png") -> Path:
    """Plots comparative validation curves for U-Net, DeepLabV3+, and SegFormer."""
    histories = {}
    for m_name in ["unet", "deeplabv3plus", "segformer"]:
        hist_p = REPORTS_DIR / f"{m_name}_training_history.json"
        if hist_p.exists():
            with open(hist_p, "r", encoding="utf-8") as f:
                histories[m_name] = json.load(f)

    if not histories:
        print("[!] No training histories found for comparison curves.")
        return output_path

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    fig.suptitle("DocForensics AI — Multi-Model Architecture Training Dynamics", fontsize=15, fontweight="bold", y=0.98)

    colors = {"unet": "#2563eb", "deeplabv3plus": "#059669", "segformer": "#7c3aed"}
    markers = {"unet": "o", "deeplabv3plus": "s", "segformer": "^"}
    labels = {"unet": "U-Net Baseline", "deeplabv3plus": "DeepLabV3+ (ResNet34+ASPP)", "segformer": "SegFormer-B0 (Transformer)"}

    for m_name, data in histories.items():
        hist = data["history"]
        epochs = hist["epoch"]
        c = colors.get(m_name, "#000000")
        m = markers.get(m_name, "o")
        lbl = labels.get(m_name, m_name)

        # Plot 1: Val Loss
        axes[0].plot(epochs, hist["val_loss"], label=lbl, color=c, marker=m, linewidth=2.0, markersize=4)

        # Plot 2: Val Dice
        axes[1].plot(epochs, hist["val_dice"], label=f"{lbl} (Best: {max(hist['val_dice']):.4f})", color=c, marker=m, linewidth=2.0, markersize=4)

    axes[0].set_title("1. Validation Loss (BCE + Dice)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=10)
    axes[0].set_ylabel("Loss", fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(fontsize=9.5)

    axes[1].set_title("2. Validation Dice Score (F1)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=10)
    axes[1].set_ylabel("Dice Score", fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(fontsize=9.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()

    print(f"[+] Comparative training curves saved to: {output_path}")
    return output_path


def plot_model_comparison_grid(
    output_path: Path = OUTPUT_DIR / "model_comparison_predictions.png",
    num_samples: int = 8,
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> Path:
    """
    Renders 5-column multi-model visual comparison grid:
    [ Document | Ground Truth | U-Net Prediction | DeepLabV3+ Prediction | SegFormer Prediction ]
    """
    dev = device or get_target_device(cfg.training.device)

    # Load models
    models = {}
    builders = {
        "unet": (get_unet_model(in_channels=3, num_classes=1, base_channels=32, device=dev), CHECKPOINT_DIR / "unet_best.pth"),
        "deeplabv3plus": (get_deeplabv3plus_model(in_channels=3, num_classes=1, device=dev), CHECKPOINT_DIR / "deeplabv3plus_best.pth"),
        "segformer": (get_segformer_model(in_channels=3, num_classes=1, variant="b0", device=dev), CHECKPOINT_DIR / "segformer_best.pth"),
    }

    for m_name, (m_instance, ckpt_p) in builders.items():
        if ckpt_p.exists():
            ckpt = torch.load(ckpt_p, map_location=dev, weights_only=False)
            m_instance.load_state_dict(ckpt["model_state_dict"])
            m_instance.eval()
            models[m_name] = m_instance

    test_dataset = DocForensicsDataset(
        metadata_path=METADATA_CSV,
        split="test",
        is_training=False,
        image_size=cfg.preprocessing.image_size,
        normalize=True,
    )

    np.random.seed(seed)
    indices = np.random.choice(len(test_dataset), size=min(num_samples, len(test_dataset)), replace=False)

    rows = len(indices)
    cols = 5  # [Doc, GT, UNet, DeepLab, SegFormer]
    fig, axes = plt.subplots(rows, cols, figsize=(24, 4.8 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, 0)

    fig.suptitle(
        "DocForensics AI — Multi-Model Architecture Comparison on Untouched Test Set\n"
        "[ Document Image | Ground Truth Mask | U-Net Baseline | DeepLabV3+ (CNN) | SegFormer (Transformer) ]",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    col_titles = [
        "1. Document Image (RGB)",
        "2. Ground Truth Mask (GT)",
        "3. U-Net Prediction",
        "4. DeepLabV3+ Prediction",
        "5. SegFormer Prediction",
    ]

    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title, fontsize=12, fontweight="bold", pad=10)

    with torch.no_grad():
        for r_idx, sample_idx in enumerate(indices):
            item = test_dataset[sample_idx]
            img_tensor = item["image"].unsqueeze(0).to(dev)
            gt_mask_np = (item["mask"][0].numpy() * 255).astype(np.uint8)

            # Unnormalize image
            mean = np.array(cfg.preprocessing.mean, dtype=np.float32).reshape(3, 1, 1)
            std = np.array(cfg.preprocessing.std, dtype=np.float32).reshape(3, 1, 1)
            img_unnorm = (item["image"].numpy() * std + mean).transpose(1, 2, 0)
            img_rgb = np.clip(img_unnorm * 255.0, 0, 255).astype(np.uint8)

            # 1. Document Image
            axes[r_idx, 0].imshow(img_rgb)
            axes[r_idx, 0].axis("off")
            axes[r_idx, 0].text(8, 490, f"ID: {item['sample_id']}\nManip: {item['manipulation_type']}",
                                color="white", fontsize=8.0, bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.75))

            # 2. GT Mask
            gt_overlay = create_colored_overlay(img_rgb, gt_mask_np, color=(220, 38, 38))
            axes[r_idx, 1].imshow(gt_overlay)
            axes[r_idx, 1].axis("off")
            gt_pixels = int(np.sum(gt_mask_np > 0))
            axes[r_idx, 1].text(8, 490, f"GT Pixels: {gt_pixels:,}\nStatus: {'TAMPERED' if gt_pixels > 0 else 'AUTHENTIC'}",
                                color="yellow", fontsize=8.0, bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.75))

            # Predictions for each model
            for c_idx, m_key in enumerate(["unet", "deeplabv3plus", "segformer"], start=2):
                if m_key in models:
                    logits = models[m_key](img_tensor)
                    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
                    pred_bin = ((prob >= 0.5).astype(np.uint8)) * 255

                    # Compute dice
                    tp = np.sum((gt_mask_np > 0) & (pred_bin > 0))
                    fp = np.sum((gt_mask_np == 0) & (pred_bin > 0))
                    fn = np.sum((gt_mask_np > 0) & (pred_bin == 0))
                    dice = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-6) if np.sum(gt_mask_np > 0) > 0 else (1.0 if fp == 0 else 0.0)

                    color = (5, 150, 105) if dice > 0.6 else ((217, 119, 6) if dice > 0.2 else (220, 38, 38))
                    pred_overlay = create_colored_overlay(img_rgb, pred_bin, color=color)
                    axes[r_idx, c_idx].imshow(pred_overlay)
                    axes[r_idx, c_idx].text(8, 490, f"Dice: {dice:.4f} | Pixels: {int(np.sum(pred_bin > 0)):,}",
                                            color="white", fontsize=8.0, bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.75))
                else:
                    axes[r_idx, c_idx].text(0.5, 0.5, f"[{m_key.upper()} N/A]", ha="center", va="center", fontsize=12)
                axes[r_idx, c_idx].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()

    print(f"[+] Multi-model comparison prediction grid saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    plot_model_comparison_curves()
    plot_model_comparison_grid()
