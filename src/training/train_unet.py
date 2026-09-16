"""
DocForensics AI — U-Net Baseline Training Pipeline (Phase 5 & GPU Acceleration)
Supports configurable execution across:
- Windows Laptop with NVIDIA GeForce RTX 5050 (8GB VRAM)
- NVIDIA DGX Spark (Multi-GPU / High-VRAM)
- Standard CPU fallback
"""

import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    BASE_DIR,
    METADATA_CSV,
    CHECKPOINT_DIR,
    OUTPUT_DIR,
    REPORTS_DIR,
    cfg,
    get_target_device,
)
from src.models.unet import get_unet_model
from src.preprocessing.dataset import DocForensicsDataset
from src.training.losses import BCEDiceLoss
from src.training.trainer import Trainer
from src.training.gpu_utils import set_seed, get_hardware_diagnostics
from src.evaluation.evaluate_unet import evaluate_model_on_test_set
from src.evaluation.visualize_predictions import plot_training_curves, plot_test_predictions


def run_training_pipeline(
    profile: Optional[str] = None,
    device_str: Optional[str] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[int] = None,
    base_channels: int = 32,
    patience: Optional[int] = None,
    seed: int = 42,
    mixed_precision: Optional[bool] = None,
    num_workers: Optional[int] = None,
) -> dict:
    """
    Executes complete training and evaluation pipeline for U-Net baseline.
    Configurable for laptop RTX 5050 and DGX Spark.
    """
    # Apply hardware profile if requested
    if profile:
        cfg.apply_hardware_profile(profile)

    # Resolve parameters
    target_device = get_target_device(device_str or cfg.training.device)
    epochs_val = epochs or cfg.training.num_epochs
    batch_size_val = batch_size or cfg.training.batch_size
    lr_val = learning_rate or cfg.training.learning_rate
    patience_val = patience or cfg.training.early_stopping_patience
    use_amp = mixed_precision if mixed_precision is not None else cfg.training.mixed_precision
    workers_val = num_workers if num_workers is not None else cfg.training.num_workers
    pin_mem = cfg.training.pin_memory if target_device.type == "cuda" else False
    grad_accum = cfg.training.gradient_accumulation_steps

    set_seed(seed)
    diag = get_hardware_diagnostics()

    print("=" * 75)
    print("        DocForensics AI — U-Net Training & Hardware Configuration     ")
    print(f"  Hardware Profile  : {cfg.hardware_profile}")
    print(f"  Target Device     : {target_device} ({diag.get('gpu_name', 'N/A')})")
    print(f"  Total VRAM        : {diag.get('total_memory_gb', 0)} GB")
    print(f"  Batch Size        : {batch_size_val} (Grad Accum: {grad_accum} -> Effective: {batch_size_val * grad_accum})")
    print(f"  AMP Enabled       : {use_amp}")
    print(f"  Data Workers      : {workers_val} | Pin Memory: {pin_mem}")
    print(f"  Learning Rate     : {lr_val}")
    print(f"  Base Channels     : {base_channels}")
    print(f"  Max Epochs        : {epochs_val} (Patience={patience_val})")
    print("=" * 75)

    # 1. Prepare Datasets & DataLoaders
    print("\n--- 1. Initializing PyTorch Datasets & DataLoaders ---")
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_val,
        shuffle=True,
        num_workers=workers_val,
        pin_memory=pin_mem,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size_val,
        shuffle=False,
        num_workers=workers_val,
        pin_memory=pin_mem,
    )

    print(f"  Train Set : {len(train_dataset)} samples ({len(train_loader)} batches)")
    print(f"  Val Set   : {len(val_dataset)} samples ({len(val_loader)} batches)")

    # 2. Instantiate Model, Loss, Optimizer & Scheduler
    print("\n--- 2. Instantiating U-Net Model & Optimizer ---")
    model = get_unet_model(
        in_channels=3,
        num_classes=1,
        base_channels=base_channels,
        device=target_device,
    )

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total Trainable Parameters: {param_count:,}")

    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_val, weight_decay=cfg.training.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_val, eta_min=1e-5)

    # 3. Fit Model (Check if resuming from checkpoint)
    print("\n--- 3. Starting Training Loop ---")
    resume_ckpt = CHECKPOINT_DIR / "unet_latest.pth"
    resume_path = resume_ckpt if resume_ckpt.exists() else None

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=target_device,
        mixed_precision=use_amp,
        gradient_accumulation_steps=grad_accum,
        early_stopping_patience=patience_val,
        model_name="unet",
        seed=seed,
    )
    training_res = trainer.fit(num_epochs=epochs_val, resume_from=resume_path)

    # 4. Generate Training Curves
    print("\n--- 4. Generating Training Dynamics Curves ---")
    plot_training_curves()

    # 5. Evaluate Best Checkpoint on Untouched Test Set
    print("\n--- 5. Evaluating Best Checkpoint on Untouched Test Set ---")
    best_ckpt = CHECKPOINT_DIR / "unet_best.pth"
    test_res = evaluate_model_on_test_set(
        checkpoint_path=best_ckpt,
        metadata_path=METADATA_CSV,
        batch_size=batch_size_val,
        device=target_device,
    )

    # 6. Generate Test Prediction Visualizations
    print("\n--- 6. Generating Visual Test Prediction Overlays ---")
    plot_test_predictions(checkpoint_path=best_ckpt, device=target_device)

    print("\n[SUCCESS] U-Net Baseline Training & Evaluation Completed!")
    return {
        "training": training_res,
        "test": test_res,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DocForensics AI — U-Net Training Runner")
    parser.add_argument("--profile", type=str, default=None, help="Hardware profile (laptop_rtx5050, dgx_spark, cpu)")
    parser.add_argument("--device", type=str, default=None, help="Compute device (cuda, cpu)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader worker threads")
    args = parser.parse_args()

    run_training_pipeline(
        profile=args.profile,
        device_str=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_workers=args.num_workers,
    )
