"""
DocForensics AI — Phase 7 Visual Comparisons & Localization Maps
Generates:
1. Multi-Model 5-Column Grid: Document | Ground Truth | DeepLabV3+ | Dual-Stream Prediction | Overlay
2. Dedicated Challenging Cases: Copy-Move, Inpainting, and RealText-V2 comparisons.
Saved to outputs/phase7_model_comparison.png and outputs/phase7_challenging_cases.png.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

from config import BASE_DIR, CHECKPOINT_DIR, OUTPUT_DIR, METADATA_CSV, cfg, get_target_device
from src.preprocessing.dataset import DocForensicsDataset
from src.models.deeplabv3plus import get_deeplabv3plus_model
from src.models.dual_stream_forensics import get_dual_stream_model


def create_overlay(image_rgb: np.ndarray, mask: np.ndarray, color=(255, 0, 0), alpha=0.45) -> np.ndarray:
    """Blends a binary prediction mask as a semi-transparent colored overlay over the RGB image."""
    overlay = image_rgb.copy()
    mask_bool = mask > 0.5
    for c in range(3):
        overlay[:, :, c] = np.where(
            mask_bool,
            (1 - alpha) * overlay[:, :, c] + alpha * color[c],
            overlay[:, :, c],
        )
    return np.clip(overlay, 0, 255).astype(np.uint8)


def generate_phase7_visualizations(
    output_comparison_path: Optional[Path] = None,
    output_challenging_path: Optional[Path] = None,
    threshold: float = 0.5,
    device: Optional[torch.device] = None,
):
    """
    Generates comparative visualizations showing DeepLabV3+ vs Dual-Stream on test samples.
    """
    dev = device or get_target_device(cfg.training.device)
    out_cmp = output_comparison_path or (OUTPUT_DIR / "phase7_model_comparison.png")
    out_chg = output_challenging_path or (OUTPUT_DIR / "phase7_challenging_cases.png")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Dataset
    test_dataset = DocForensicsDataset(
        metadata_path=METADATA_CSV,
        split="test",
        is_training=False,
        image_size=cfg.preprocessing.image_size,
        normalize=True,
    )

    # 2. Load DeepLabV3+ Baseline
    deeplab_ckpt = CHECKPOINT_DIR / "deeplabv3plus_best.pth"
    deeplab_model = get_deeplabv3plus_model(in_channels=3, num_classes=1, device=dev)
    if deeplab_ckpt.exists():
        ckpt_d = torch.load(deeplab_ckpt, map_location=dev, weights_only=False)
        deeplab_model.load_state_dict(ckpt_d["model_state_dict"])
    deeplab_model.eval()

    # 3. Load Dual-Stream Forensic Model
    dual_ckpt = CHECKPOINT_DIR / "dual_stream_best.pth"
    if not dual_ckpt.exists():
        print(f"[!] Warning: Dual-Stream checkpoint not found at {dual_ckpt}")
        return

    ckpt_dual = torch.load(dual_ckpt, map_location=dev, weights_only=False)
    fusion_type = "gated" if "fusion.gate_conv.0.weight" in ckpt_dual["model_state_dict"] else "baseline"
    dual_model = get_dual_stream_model(in_channels=3, num_classes=1, fusion_type=fusion_type, device=dev)
    dual_model.load_state_dict(ckpt_dual["model_state_dict"])
    dual_model.eval()

    # 4. Select Diverse Samples
    sample_indices = []
    # Find representative samples across categories
    categories_sought = [
        "text_replacement",
        "date_replacement",
        "signature_manipulation",
        "stamp_seal_manipulation",
        "copy_move_intra",
        "local_inpainting",
        "public_realtext_tampered",
        "none",
    ]

    for cat in categories_sought:
        for idx in range(len(test_dataset)):
            m_type = test_dataset.df.iloc[idx]["manipulation_type"]
            if m_type == cat and idx not in sample_indices:
                sample_indices.append(idx)
                break

    # If some categories missing, fill with other samples
    while len(sample_indices) < 6 and len(sample_indices) < len(test_dataset):
        for idx in range(len(test_dataset)):
            if idx not in sample_indices:
                sample_indices.append(idx)
                break

    # 5. Generate 5-Column Grid: Document | Ground Truth | DeepLabV3+ | Dual-Stream | Overlay
    fig, axes = plt.subplots(len(sample_indices), 5, figsize=(20, 3.8 * len(sample_indices)))
    plt.subplots_adjust(wspace=0.08, hspace=0.25)

    headers = [
        "Document Input",
        "Ground Truth Mask",
        "DeepLabV3+ (RGB Only)",
        "Dual-Stream (RGB+Forensic)",
        "Dual-Stream Overlay",
    ]

    for col, title in enumerate(headers):
        axes[0, col].set_title(title, fontsize=13, fontweight="bold", pad=12)

    with torch.no_grad():
        for row, idx in enumerate(sample_indices):
            sample = test_dataset[idx]
            img_t = sample["image"].unsqueeze(0).to(dev)
            mask_t = sample["mask"].numpy()[0]
            m_type = sample["manipulation_type"]
            s_id = sample["sample_id"]

            # Predictions
            out_deeplab = torch.sigmoid(deeplab_model(img_t))[0, 0].cpu().numpy()
            pred_deeplab = (out_deeplab >= threshold).astype(np.float32)

            out_dual = torch.sigmoid(dual_model(img_t))[0, 0].cpu().numpy()
            pred_dual = (out_dual >= threshold).astype(np.float32)

            # Unnormalize image for visualization
            mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
            img_orig = sample["image"].numpy() * std + mean
            img_orig = np.clip(img_orig * 255.0, 0, 255).transpose(1, 2, 0).astype(np.uint8)

            overlay = create_overlay(img_orig, pred_dual, color=(255, 30, 30), alpha=0.5)

            # Plot row
            axes[row, 0].imshow(img_orig)
            axes[row, 0].set_ylabel(f"[{m_type}]\n{s_id[:16]}", fontsize=9, fontweight="semibold")
            axes[row, 0].set_xticks([])
            axes[row, 0].set_yticks([])

            axes[row, 1].imshow(mask_t, cmap="gray", vmin=0, vmax=1)
            axes[row, 1].set_xticks([])
            axes[row, 1].set_yticks([])

            axes[row, 2].imshow(pred_deeplab, cmap="magma", vmin=0, vmax=1)
            axes[row, 2].set_xticks([])
            axes[row, 2].set_yticks([])

            axes[row, 3].imshow(pred_dual, cmap="magma", vmin=0, vmax=1)
            axes[row, 3].set_xticks([])
            axes[row, 3].set_yticks([])

            axes[row, 4].imshow(overlay)
            axes[row, 4].set_xticks([])
            axes[row, 4].set_yticks([])

    plt.tight_layout()
    plt.savefig(out_cmp, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved Phase 7 Model Comparison visual grid to: {out_cmp}")

    # 6. Generate Challenging Cases Visual Grid (Copy-move, Inpainting, RealText-V2)
    challenging_cats = ["copy_move_intra", "local_inpainting", "public_realtext_tampered"]
    chg_indices = []
    for cat in challenging_cats:
        for idx in range(len(test_dataset)):
            if test_dataset.df.iloc[idx]["manipulation_type"] == cat:
                chg_indices.append(idx)
                if len([i for i in chg_indices if test_dataset.df.iloc[i]["manipulation_type"] == cat]) >= 2:
                    break

    if len(chg_indices) > 0:
        fig, axes = plt.subplots(len(chg_indices), 5, figsize=(20, 3.8 * len(chg_indices)))
        if len(chg_indices) == 1:
            axes = np.expand_dims(axes, 0)
        plt.subplots_adjust(wspace=0.08, hspace=0.25)

        for col, title in enumerate(headers):
            axes[0, col].set_title(title, fontsize=13, fontweight="bold", pad=12)

        with torch.no_grad():
            for row, idx in enumerate(chg_indices):
                sample = test_dataset[idx]
                img_t = sample["image"].unsqueeze(0).to(dev)
                mask_t = sample["mask"].numpy()[0]
                m_type = sample["manipulation_type"]
                s_id = sample["sample_id"]

                out_deeplab = torch.sigmoid(deeplab_model(img_t))[0, 0].cpu().numpy()
                pred_deeplab = (out_deeplab >= threshold).astype(np.float32)

                out_dual = torch.sigmoid(dual_model(img_t))[0, 0].cpu().numpy()
                pred_dual = (out_dual >= threshold).astype(np.float32)

                mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
                std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
                img_orig = sample["image"].numpy() * std + mean
                img_orig = np.clip(img_orig * 255.0, 0, 255).transpose(1, 2, 0).astype(np.uint8)

                overlay = create_overlay(img_orig, pred_dual, color=(255, 30, 30), alpha=0.5)

                axes[row, 0].imshow(img_orig)
                axes[row, 0].set_ylabel(f"CHALLENGING:\n[{m_type}]\n{s_id[:14]}", fontsize=9, fontweight="bold", color="darkred")
                axes[row, 0].set_xticks([])
                axes[row, 0].set_yticks([])

                axes[row, 1].imshow(mask_t, cmap="gray", vmin=0, vmax=1)
                axes[row, 1].set_xticks([])
                axes[row, 1].set_yticks([])

                axes[row, 2].imshow(pred_deeplab, cmap="magma", vmin=0, vmax=1)
                axes[row, 2].set_xticks([])
                axes[row, 2].set_yticks([])

                axes[row, 3].imshow(pred_dual, cmap="magma", vmin=0, vmax=1)
                axes[row, 3].set_xticks([])
                axes[row, 3].set_yticks([])

                axes[row, 4].imshow(overlay)
                axes[row, 4].set_xticks([])
                axes[row, 4].set_yticks([])

        plt.tight_layout()
        plt.savefig(out_chg, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[+] Saved Phase 7 Challenging Cases visual grid to: {out_chg}")


if __name__ == "__main__":
    generate_phase7_visualizations()
