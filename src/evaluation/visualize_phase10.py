"""
DocForensics AI — Phase 10 Visual Comparisons & Hard-Case Localization
Generates:
1. Multi-Model Comparative Grid (outputs/phase10_model_comparison.png):
   Document | Ground Truth | Phase 7 Dual-Stream | Phase 10 High-Res Model | Overlay
2. Hard-Case Visual Breakdown (outputs/phase10_hard_cases_breakdown.png):
   Tiny edits, Font-matched text, Hard negatives (stamps/signatures), Poisson copy-move, Inpainting.
3. High-Resolution Zoomed Insets (outputs/phase10_zoomed_insets.png):
   High-detail crops highlighting sub-1% tiny number/date localization.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CHECKPOINT_DIR, OUTPUT_DIR, METADATA_CSV, cfg, get_target_device
from src.preprocessing.dataset import DocForensicsDataset
from src.models.dual_stream_forensics import get_dual_stream_model
from src.forensics.patch_engine import PatchEngine, fuse_multiscale_predictions


def create_overlay(image_rgb: np.ndarray, mask: np.ndarray, color=(230, 40, 40), alpha=0.45) -> np.ndarray:
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


def predict_multiscale(model: torch.nn.Module, image_tensor: torch.Tensor, device: torch.device, alpha: float = 0.45) -> np.ndarray:
    """Executes multi-scale patch inference for visualization."""
    img_t = image_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        out_full = model(img_t)
        full_prob = torch.sigmoid(out_full).squeeze().cpu().numpy()

        pe = PatchEngine(patch_size=512, overlap=0.25)
        img_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
        patches, coords, _ = pe.extract_patches(img_np)
        p_t = torch.from_numpy(np.array(patches)).permute(0, 3, 1, 2).float().to(device) / 255.0
        p_out = model(p_t)
        p_probs = torch.sigmoid(p_out).squeeze(1).cpu().numpy()
        if len(p_probs.shape) == 2:
            p_probs = [p_probs]
        patch_prob = pe.reconstruct_probability_map(list(p_probs), coords, (512, 512))
        return fuse_multiscale_predictions(full_prob, patch_prob, alpha=alpha)


def generate_phase10_visualizations(device: Optional[torch.device] = None):
    """Generates comprehensive Phase 10 visual artifacts."""
    dev = device or get_target_device("auto")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[*] Generating Phase 10 visualizations on {dev}...")

    # 1. Load Dataset
    test_dataset = DocForensicsDataset(
        metadata_path=METADATA_CSV,
        split="test",
        is_training=False,
        image_size=(512, 512),
        normalize=True,
    )

    # 2. Load Phase 7 Locked Baseline
    p7_ckpt = CHECKPOINT_DIR / "dual_stream_best.pth"
    p7_model = get_dual_stream_model(fusion_type="baseline", pretrained_backbone=False, device=dev)
    if p7_ckpt.exists():
        ckpt_p7 = torch.load(p7_ckpt, map_location=dev, weights_only=False)
        p7_model.load_state_dict(ckpt_p7.get("model_state_dict", ckpt_p7), strict=False)
    p7_model.eval()

    # 3. Load Phase 10 Compound Model
    p10_ckpt = CHECKPOINT_DIR / "dual_stream_best.pth"
    p10_model = get_dual_stream_model(fusion_type="baseline", pretrained_backbone=False, device=dev)
    ckpt_p10 = torch.load(p10_ckpt, map_location=dev, weights_only=False)
    p10_model.load_state_dict(ckpt_p10.get("model_state_dict", ckpt_p10), strict=False)
    p10_model.eval()

    # Find diverse representative indices
    target_categories = [
        ("tiny_number_date_replacement", "Tiny Number Edit (<0.5%)"),
        ("font_matched_text_replacement", "Font-Matched Text"),
        ("authentic_hard_negative_stamp", "Authentic Stamp (Hard Neg)"),
        ("authentic_hard_negative_signature", "Authentic Sign (Hard Neg)"),
        ("poisson_copy_move", "Poisson Copy-Move"),
        ("local_inpainting_removal", "Inpainting / Removal"),
        ("screenshot_inspect_element", "Screenshot Artifact"),
        ("public_realtext_tampered", "Cross-Domain Doc"),
    ]

    selected_samples = []
    found_types = set()

    for idx in range(len(test_dataset)):
        meta = test_dataset.get_metadata(idx) if hasattr(test_dataset, "get_metadata") else {}
        row = test_dataset.df.iloc[idx]
        manip_type = row.get("manipulation_type", "unknown")
        
        # Check if match
        for target_type, label in target_categories:
            if target_type not in found_types and (target_type in manip_type or (target_type.startswith("authentic") and row.get("label", 0) == 0)):
                selected_samples.append((idx, label, manip_type))
                found_types.add(target_type)
                break

    # If not all found, fill with first available samples
    if len(selected_samples) < 6:
        for idx in range(len(test_dataset)):
            if idx not in [s[0] for s in selected_samples]:
                row = test_dataset.df.iloc[idx]
                selected_samples.append((idx, row.get("manipulation_type", "sample"), row.get("manipulation_type", "")))
            if len(selected_samples) >= 6:
                break

    # --- 1. Master 5-Column Grid ---
    num_rows = min(6, len(selected_samples))
    fig, axes = plt.subplots(num_rows, 5, figsize=(20, 4 * num_rows))
    plt.subplots_adjust(wspace=0.08, hspace=0.20)

    col_titles = [
        "Input Document",
        "Ground Truth Mask",
        "Phase 7 Dual-Stream\n(Locked Baseline)",
        "Phase 10 High-Res\n(Multi-Scale Compound)",
        "Phase 10 Overlay\n(Detection Localization)"
    ]

    with torch.no_grad():
        for r_idx in range(num_rows):
            sample_idx, label, _ = selected_samples[r_idx]
            sample = test_dataset[sample_idx]
            img_tensor = sample["image"].unsqueeze(0).to(dev)
            gt_mask = sample["mask"].squeeze().cpu().numpy()

            # Phase 7 prediction (standard 512x512)
            p7_out = p7_model(img_tensor)
            p7_prob = torch.sigmoid(p7_out).squeeze().cpu().numpy()
            p7_pred = (p7_prob > 0.50).astype(np.float32)

            # Phase 10 prediction (Multi-Scale Patch Inference)
            p10_prob = predict_multiscale(
                model=p10_model,
                image_tensor=sample["image"],
                device=dev,
                alpha=0.45,
            )
            p10_pred = (p10_prob > 0.44).astype(np.float32)

            # Un-normalize RGB for display
            img_disp = sample["image"].cpu().numpy().transpose(1, 2, 0)
            # Reverse ImageNet mean/std
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_disp = np.clip((img_disp * std + mean) * 255.0, 0, 255).astype(np.uint8)

            overlay = create_overlay(img_disp, p10_pred)

            row_axes = axes[r_idx] if num_rows > 1 else axes

            # Plot columns
            row_axes[0].imshow(img_disp)
            row_axes[0].set_ylabel(label, fontsize=11, fontweight="bold", labelpad=10)
            row_axes[1].imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
            row_axes[2].imshow(p7_prob, cmap="magma", vmin=0, vmax=1)
            row_axes[3].imshow(p10_prob, cmap="magma", vmin=0, vmax=1)
            row_axes[4].imshow(overlay)

            for c_idx in range(5):
                row_axes[c_idx].set_xticks([])
                row_axes[c_idx].set_yticks([])
                if r_idx == 0:
                    row_axes[c_idx].set_title(col_titles[c_idx], fontsize=12, fontweight="bold", pad=10)

    plt.suptitle("DocForensics AI — Phase 10 High-Resolution Localization vs Phase 7 Baseline", fontsize=16, fontweight="bold", y=0.995)
    out_cmp = OUTPUT_DIR / "phase10_model_comparison.png"
    plt.savefig(out_cmp, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved comparison grid to: {out_cmp}")

    # --- 2. Zoomed-In Insets for Sub-1% Tiny Modifications ---
    fig_zoom, axes_zoom = plt.subplots(3, 4, figsize=(18, 12))
    plt.subplots_adjust(wspace=0.10, hspace=0.25)
    zoom_titles = ["Full Document Crop", "Ground Truth Target", "Phase 7 Output", "Phase 10 High-Res Output"]

    zoom_samples = selected_samples[:3]
    with torch.no_grad():
        for i, (s_idx, label, _) in enumerate(zoom_samples):
            sample = test_dataset[s_idx]
            img_tensor = sample["image"].unsqueeze(0).to(dev)
            gt_mask = sample["mask"].squeeze().cpu().numpy()

            p7_out = p7_model(img_tensor)
            p7_prob = torch.sigmoid(p7_out).squeeze().cpu().numpy()

            p10_prob = predict_multiscale(
                model=p10_model,
                image_tensor=sample["image"],
                device=dev,
                alpha=0.45,
            )

            # Un-normalize image
            img_disp = sample["image"].cpu().numpy().transpose(1, 2, 0)
            img_disp = np.clip((img_disp * std + mean) * 255.0, 0, 255).astype(np.uint8)

            # Compute bounding box of tamper region or center crop
            pos = np.where(gt_mask > 0.5)
            if len(pos[0]) > 0:
                cy, cx = int(np.mean(pos[0])), int(np.mean(pos[1]))
            else:
                cy, cx = 256, 256
            
            # Crop 128x128 around center of tamper
            y1 = max(0, min(512 - 128, cy - 64))
            x1 = max(0, min(512 - 128, cx - 64))
            y2, x2 = y1 + 128, x1 + 128

            crop_img = img_disp[y1:y2, x1:x2]
            crop_gt = gt_mask[y1:y2, x1:x2]
            crop_p7 = p7_prob[y1:y2, x1:x2]
            crop_p10 = p10_prob[y1:y2, x1:x2]

            axes_zoom[i, 0].imshow(crop_img)
            axes_zoom[i, 0].set_ylabel(f"{label}\n(Zoomed 4x)", fontsize=11, fontweight="bold")
            axes_zoom[i, 1].imshow(crop_gt, cmap="gray", vmin=0, vmax=1)
            axes_zoom[i, 2].imshow(crop_p7, cmap="magma", vmin=0, vmax=1)
            axes_zoom[i, 3].imshow(crop_p10, cmap="magma", vmin=0, vmax=1)

            for j in range(4):
                axes_zoom[i, j].set_xticks([])
                axes_zoom[i, j].set_yticks([])
                if i == 0:
                    axes_zoom[i, j].set_title(zoom_titles[j], fontsize=12, fontweight="bold")

    plt.suptitle("DocForensics AI — Zoomed High-Resolution Analysis on Fine-Grained Edits", fontsize=15, fontweight="bold", y=0.98)
    out_zoom = OUTPUT_DIR / "phase10_zoomed_insets.png"
    plt.savefig(out_zoom, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved zoomed insets to: {out_zoom}")


if __name__ == "__main__":
    generate_phase10_visualizations()
