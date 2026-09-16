"""
DocForensics AI — Dual-Stream Training & Validation Ablation (Phase 7)
Trains and validates Dual-Stream Forensic Models on RTX 5050 / DGX Spark.
Executes controlled validation ablation:
  1. RGB-Only (DeepLabV3+ Baseline)
  2. RGB + SRM Residual Stream (Baseline Concat Fusion)
  3. RGB + SRM Residual Stream (Gated Attention Fusion)
Saves the best model checkpoint to checkpoints/dual_stream_best.pth based on validation performance.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (
    BASE_DIR,
    CHECKPOINT_DIR,
    REPORTS_DIR,
    METADATA_CSV,
    cfg,
    get_target_device,
)
from src.models.dual_stream_forensics import get_dual_stream_model
from src.preprocessing.dataset import DocForensicsDataset
from src.training.trainer import Trainer
from src.training.losses import BCEDiceLoss
from src.training.gpu_utils import get_hardware_diagnostics, get_vram_usage, set_seed


def run_ablation_and_training(
    epochs: int = 20,
    batch_size: int = 4,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 6,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Executes controlled validation ablation and trains the finalized Dual-Stream model.
    """
    set_seed(cfg.training.seed)
    dev = device or get_target_device(cfg.training.device)
    diag = get_hardware_diagnostics()
    print(f"[*] Hardware Environment: GPU={diag['gpu_name']} | VRAM={diag['total_memory_gb']}GB | PyTorch={diag['pytorch_version']}")

    print("\n" + "=" * 75)
    print("      DocForensics AI — Phase 7 Dual-Stream Forensic Training & Ablation      ")
    print("=" * 75)

    # 1. Prepare Datasets & DataLoaders
    train_dataset = DocForensicsDataset(
        metadata_path=METADATA_CSV,
        split="train",
        is_training=True,
        image_size=cfg.preprocessing.image_size,
        normalize=True,
    )
    val_dataset = DocForensicsDataset(
        metadata_path=METADATA_CSV,
        split="val",
        is_training=False,
        image_size=cfg.preprocessing.image_size,
        normalize=True,
    )

    num_workers = cfg.training.num_workers
    pin_mem = cfg.training.pin_memory and (dev.type == "cuda")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
    )

    print(f"[*] Training Samples: {len(train_dataset)} | Val Samples: {len(val_dataset)}")
    print(f"[*] Batch Size: {batch_size} | AMP Enabled: {cfg.training.mixed_precision}")

    ablation_results = {}

    # Check existing DeepLabV3+ baseline validation results
    deeplab_history_p = REPORTS_DIR / "deeplabv3plus_training_history.json"
    if deeplab_history_p.exists():
        with open(deeplab_history_p, "r", encoding="utf-8") as f:
            d_hist = json.load(f)
            ablation_results["1_rgb_only_deeplabv3plus"] = {
                "description": "RGB-only single stream (DeepLabV3+)",
                "fusion_type": "none",
                "best_val_epoch": d_hist.get("best_epoch"),
                "best_val_dice": d_hist.get("best_val_dice"),
                "total_training_time": d_hist.get("total_training_time_seconds"),
            }

    # Configuration 2: Dual-Stream with Baseline Concat Fusion
    print("\n" + "-" * 75)
    print("  [Ablation 2] Training Dual-Stream with Baseline Concat Fusion...")
    print("-" * 75)

    set_seed(cfg.training.seed)
    model_baseline = get_dual_stream_model(
        in_channels=3,
        num_classes=1,
        pretrained_backbone=False,
        fusion_type="baseline",
        device=dev,
    )

    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer_baseline = torch.optim.AdamW(model_baseline.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler_baseline = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_baseline, mode="max", factor=0.5, patience=2, min_lr=1e-6
    )

    trainer_baseline = Trainer(
        model=model_baseline,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer_baseline,
        scheduler=scheduler_baseline,
        device=dev,
        mixed_precision=cfg.training.mixed_precision,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        early_stopping_patience=patience,
        model_name="dual_stream_baseline",
        seed=cfg.training.seed,
    )

    res_baseline = trainer_baseline.fit(num_epochs=epochs)
    ablation_results["2_dual_stream_concat_fusion"] = {
        "description": "RGB + SRM Forensic Stream (Concat + Conv Fusion)",
        "fusion_type": "baseline",
        "best_val_epoch": res_baseline["best_epoch"],
        "best_val_dice": round(res_baseline["best_val_dice"], 4),
        "total_training_time": round(res_baseline["training_time"], 2),
        "checkpoint": res_baseline["best_checkpoint"],
    }

    # Configuration 3: Dual-Stream with Gated Attention Fusion
    print("\n" + "-" * 75)
    print("  [Ablation 3] Training Dual-Stream with Gated Attention Fusion...")
    print("-" * 75)

    set_seed(cfg.training.seed)
    model_gated = get_dual_stream_model(
        in_channels=3,
        num_classes=1,
        pretrained_backbone=False,
        fusion_type="gated",
        device=dev,
    )

    optimizer_gated = torch.optim.AdamW(model_gated.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler_gated = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_gated, mode="max", factor=0.5, patience=2, min_lr=1e-6
    )

    trainer_gated = Trainer(
        model=model_gated,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer_gated,
        scheduler=scheduler_gated,
        device=dev,
        mixed_precision=cfg.training.mixed_precision,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        early_stopping_patience=patience,
        model_name="dual_stream_gated",
        seed=cfg.training.seed,
    )

    res_gated = trainer_gated.fit(num_epochs=epochs)
    ablation_results["3_dual_stream_gated_attention"] = {
        "description": "RGB + SRM Forensic Stream (Gated Attention Fusion)",
        "fusion_type": "gated",
        "best_val_epoch": res_gated["best_epoch"],
        "best_val_dice": round(res_gated["best_val_dice"], 4),
        "total_training_time": round(res_gated["training_time"], 2),
        "checkpoint": res_gated["best_checkpoint"],
    }

    # Compare validation performance and select the best Phase 7 checkpoint
    best_config_key = max(
        ["2_dual_stream_concat_fusion", "3_dual_stream_gated_attention"],
        key=lambda k: ablation_results[k]["best_val_dice"],
    )
    best_config = ablation_results[best_config_key]
    print("\n" + "=" * 75)
    print(f"  Validation Ablation Selection: {best_config_key.upper()}")
    print(f"  Selected Fusion: {best_config['fusion_type']} (Val Dice: {best_config['best_val_dice']:.4f})")
    print("=" * 75)

    # Copy the winning checkpoint to dual_stream_best.pth
    winning_ckpt_p = Path(best_config["checkpoint"])
    final_best_p = CHECKPOINT_DIR / "dual_stream_best.pth"
    if winning_ckpt_p.exists():
        import shutil
        shutil.copyfile(winning_ckpt_p, final_best_p)
        print(f"[+] Finalized Phase 7 best checkpoint saved to: {final_best_p}")

    ablation_results["selected_phase7_model"] = {
        "config_key": best_config_key,
        "fusion_type": best_config["fusion_type"],
        "best_val_dice": best_config["best_val_dice"],
        "final_checkpoint": str(final_best_p),
    }

    # Save ablation study report
    ablation_report_p = REPORTS_DIR / "phase7_ablation_study.json"
    with open(ablation_report_p, "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2)
    print(f"[+] Ablation report saved to: {ablation_report_p}")

    return ablation_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Phase 7 Dual-Stream Model")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=6, help="Early stopping patience")
    args = parser.parse_args()

    run_ablation_and_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
    )
