"""
DocForensics AI — U-Net Training Curves & Prediction Visualizer (Phase 5)
Generates:
1. outputs/unet_training_curves.png: Train/Val Loss, Dice, IoU, Precision, Recall vs Epoch
2. outputs/unet_test_predictions.png: 4-column visual comparison on untouched test set
   [ Document | Ground Truth Mask | Predicted Mask | Overlay ]
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional

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

from config import BASE_DIR, CHECKPOINT_DIR, OUTPUT_DIR, REPORTS_DIR, METADATA_CSV, cfg
from src.models.unet import get_unet_model
from src.preprocessing.dataset import DocForensicsDataset


def plot_training_curves(
    history_path: Path = REPORTS_DIR / "unet_training_history.json",
    output_path: Path = OUTPUT_DIR / "unet_training_curves.png",
) -> Path:
    """Plots training and validation loss and metric curves over epochs."""
    if not history_path.exists():
        raise FileNotFoundError(f"History file not found: {history_path}")

    with open(history_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    hist = data["history"]
    epochs = hist["epoch"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle(
        f"DocForensics AI — U-Net Baseline Training Dynamics (Best Epoch: {data.get('best_epoch')})",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    # 1. Loss Curve
    axes[0].plot(epochs, hist["train_loss"], label="Train Loss (BCE+Dice)", color="#2563eb", linewidth=2.2, marker="o", markersize=4)
    axes[0].plot(epochs, hist["val_loss"], label="Val Loss (BCE+Dice)", color="#dc2626", linewidth=2.2, marker="s", markersize=4)
    axes[0].set_title("1. Training vs Validation Loss", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=10)
    axes[0].set_ylabel("Loss", fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(fontsize=10)

    # 2. Dice & IoU Curves
    axes[1].plot(epochs, hist["val_dice"], label="Val Dice Score (F1)", color="#059669", linewidth=2.2, marker="^", markersize=4)
    axes[1].plot(epochs, hist["val_iou"], label="Val IoU (Jaccard)", color="#7c3aed", linewidth=2.2, marker="d", markersize=4)
    axes[1].set_title("2. Validation Dice & IoU Score", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=10)
    axes[1].set_ylabel("Score", fontsize=10)
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(fontsize=10)

    # 3. Precision & Recall Curves
    axes[2].plot(epochs, hist["val_precision"], label="Val Precision", color="#d97706", linewidth=2.2, marker="p", markersize=4)
    axes[2].plot(epochs, hist["val_recall"], label="Val Recall", color="#0284c7", linewidth=2.2, marker="v", markersize=4)
    axes[2].set_title("3. Validation Precision & Recall", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Epoch", fontsize=10)
    axes[2].set_ylabel("Score", fontsize=10)
    axes[2].set_ylim(0.0, 1.05)
    axes[2].grid(True, linestyle="--", alpha=0.6)
    axes[2].legend(fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()

    print(f"[+] Training curves saved to: {output_path}")
    return output_path


def create_prediction_overlay(image_rgb: np.ndarray, gt_mask: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    """
    Creates visual overlay:
    - Green = True Positives (Overlap where both GT and Pred are 1)
    - Red = False Positives (Predicted as tampered, but actually authentic)
    - Blue = False Negatives (Actual tampered, but missed by prediction)
    """
    overlay = image_rgb.copy()
    gt_bin = gt_mask > 127
    pred_bin = pred_mask > 127

    tp = gt_bin & pred_bin
    fp = (~gt_bin) & pred_bin
    fn = gt_bin & (~pred_bin)

    alpha = 0.55
    # TP = Green (22, 163, 74)
    if np.any(tp):
        overlay[tp] = (alpha * np.array([22, 163, 74]) + (1 - alpha) * overlay[tp]).astype(np.uint8)
    # FP = Red (220, 38, 38)
    if np.any(fp):
        overlay[fp] = (alpha * np.array([220, 38, 38]) + (1 - alpha) * overlay[fp]).astype(np.uint8)
    # FN = Blue / Purple (79, 70, 229)
    if np.any(fn):
        overlay[fn] = (alpha * np.array([79, 70, 229]) + (1 - alpha) * overlay[fn]).astype(np.uint8)

    # Draw contour around predicted regions
    contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (220, 38, 38), 2)

    return overlay


def plot_test_predictions(
    checkpoint_path: Path = CHECKPOINT_DIR / "unet_best.pth",
    output_path: Path = OUTPUT_DIR / "unet_test_predictions.png",
    num_samples: int = 8,
    seed: int = 42,
    device: torch.device = None,
) -> Path:
    """
    Generates 4-panel visual comparison on test set:
    Document | Ground Truth Mask | Predicted Mask | Overlay
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = get_unet_model(in_channels=3, num_classes=1, base_channels=32, device=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_dataset = DocForensicsDataset(
        metadata_path=METADATA_CSV,
        split="test",
        is_training=False,
        image_size=(512, 512),
        normalize=True,
    )

    np.random.seed(seed)
    indices = np.random.choice(len(test_dataset), size=min(num_samples, len(test_dataset)), replace=False)

    rows = len(indices)
    cols = 4  # [Document, GT Mask, Predicted Mask, Overlay]
    fig, axes = plt.subplots(rows, cols, figsize=(22, 5.2 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, 0)

    fig.suptitle(
        "DocForensics AI — U-Net Test Set Localization Visualizations\n"
        "[ Document | Ground Truth Mask | Predicted Mask | Overlay (Green=TP, Red=FP, Blue=FN) ]",
        fontsize=17,
        fontweight="bold",
        y=0.995,
    )

    col_titles = [
        "1. Document Image (RGB)",
        "2. Ground Truth Mask (GT)",
        "3. U-Net Predicted Mask (Sigmoid >= 0.5)",
        "4. Forensic Diagnostic Overlay",
    ]

    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title, fontsize=13, fontweight="bold", pad=12)

    with torch.no_grad():
        for r_idx, sample_idx in enumerate(indices):
            item = test_dataset[sample_idx]
            img_tensor = item["image"].unsqueeze(0).to(device)  # [1, 3, 512, 512]
            gt_mask_tensor = item["mask"][0].numpy()  # [512, 512] in {0.0, 1.0}
            gt_mask_uint8 = (gt_mask_tensor * 255).astype(np.uint8)

            logits = model(img_tensor)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
            pred_mask_uint8 = ((prob >= 0.5).astype(np.uint8)) * 255

            # Unnormalize image for visualization
            mean = np.array(cfg.preprocessing.mean, dtype=np.float32).reshape(3, 1, 1)
            std = np.array(cfg.preprocessing.std, dtype=np.float32).reshape(3, 1, 1)
            img_unnorm = (item["image"].numpy() * std + mean).transpose(1, 2, 0)
            img_rgb = np.clip(img_unnorm * 255.0, 0, 255).astype(np.uint8)

            overlay = create_prediction_overlay(img_rgb, gt_mask_uint8, pred_mask_uint8)

            # Compute sample metrics
            tp = np.sum((gt_mask_uint8 > 0) & (pred_mask_uint8 > 0))
            fp = np.sum((gt_mask_uint8 == 0) & (pred_mask_uint8 > 0))
            fn = np.sum((gt_mask_uint8 > 0) & (pred_mask_uint8 == 0))
            
            if np.sum(gt_mask_uint8 > 0) == 0:
                dice = 1.0 if fp == 0 else 0.0
            else:
                dice = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-6)

            # Plot 1: Document Image
            axes[r_idx, 0].imshow(img_rgb)
            axes[r_idx, 0].axis("off")
            axes[r_idx, 0].text(10, 490, f"ID: {item['sample_id']}\nManip: {item['manipulation_type']}",
                                color="white", fontsize=8.5, bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.75))

            # Plot 2: GT Mask
            axes[r_idx, 1].imshow(gt_mask_uint8, cmap="gray", vmin=0, vmax=255)
            axes[r_idx, 1].axis("off")
            axes[r_idx, 1].text(10, 490, f"GT Tampered Pixels: {int(np.sum(gt_mask_uint8 > 0)):,}",
                                color="cyan", fontsize=8.5, bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.75))

            # Plot 3: Predicted Mask
            axes[r_idx, 2].imshow(pred_mask_uint8, cmap="gray", vmin=0, vmax=255)
            axes[r_idx, 2].axis("off")
            axes[r_idx, 2].text(10, 490, f"Pred Pixels: {int(np.sum(pred_mask_uint8 > 0)):,}\nDice: {dice:.4f}",
                                color="yellow", fontsize=8.5, bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.75))

            # Plot 4: Diagnostic Overlay
            axes[r_idx, 3].imshow(overlay)
            axes[r_idx, 3].axis("off")
            status_text = "AUTHENTIC" if item["is_tampered"] == 0 else f"Dice: {dice:.4f}"
            axes[r_idx, 3].text(10, 490, f"Overlay | {status_text}",
                                color="white", fontsize=8.5, bbox=dict(boxstyle="round,pad=0.3", fc="darkred", alpha=0.75))

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()

    print(f"[+] U-Net test set predictions saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    plot_training_curves()
    plot_test_predictions()
