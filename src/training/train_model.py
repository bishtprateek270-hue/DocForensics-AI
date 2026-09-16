"""
DocForensics AI — Unified Model Training Pipeline (Phase 6)
Trains DeepLabV3+ and SegFormer segmentation models on GPU with AMP and early stopping.
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
from src.models.deeplabv3plus import get_deeplabv3plus_model
from src.models.segformer import get_segformer_model
from src.preprocessing.dataset import DocForensicsDataset
from src.training.losses import BCEDiceLoss
from src.training.trainer import Trainer
from src.training.gpu_utils import set_seed, get_hardware_diagnostics
from src.evaluation.evaluate_models import evaluate_single_model


MODEL_BUILDERS = {
    "unet": lambda dev, **kw: get_unet_model(in_channels=3, num_classes=1, base_channels=32, device=dev),
    "deeplabv3plus": lambda dev, **kw: get_deeplabv3plus_model(in_channels=3, num_classes=1, pretrained=False, device=dev),
    "segformer": lambda dev, **kw: get_segformer_model(in_channels=3, num_classes=1, variant="b0", device=dev),
}

# Architecture-specific learning rates
DEFAULT_LR = {
    "unet": 1e-3,
    "deeplabv3plus": 5e-4,  # ResNet backbone with ASPP converges stably at 5e-4
    "segformer": 2e-4,      # Transformer self-attention benefits from 2e-4
}


def train_and_evaluate_model(
    model_name: str,
    epochs: int = 15,
    batch_size: int = 4,
    learning_rate: Optional[float] = None,
    patience: int = 5,
    seed: int = 42,
    device_str: Optional[str] = "cuda",
) -> dict:
    """
    Executes training and evaluation for a specific model architecture.
    """
    set_seed(seed)
    target_device = get_target_device(device_str)
    lr_val = learning_rate or DEFAULT_LR.get(model_name.lower(), 5e-4)
    diag = get_hardware_diagnostics()

    print("=" * 75)
    print(f"  DocForensics AI — Training {model_name.upper()} Model on {target_device}  ")
    print(f"  GPU Hardware      : {diag.get('gpu_name', 'N/A')} ({diag.get('total_memory_gb', 0)} GB)")
    print(f"  Batch Size        : {batch_size} (Grad Accum: {cfg.training.gradient_accumulation_steps})")
    print(f"  Learning Rate     : {lr_val}")
    print(f"  AMP Enabled       : True")
    print(f"  Max Epochs        : {epochs} (Patience={patience})")
    print("=" * 75)

    # 1. Datasets & Loaders
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

    pin_mem = target_device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=pin_mem,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=pin_mem,
    )

    # 2. Build Model
    builder = MODEL_BUILDERS.get(model_name.lower())
    if builder is None:
        raise ValueError(f"Unknown model name: {model_name}. Available: {list(MODEL_BUILDERS.keys())}")

    model = builder(target_device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total Trainable Parameters: {param_count:,}")

    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_val, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # 3. Fit
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=target_device,
        mixed_precision=True,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        early_stopping_patience=patience,
        model_name=model_name.lower(),
        seed=seed,
    )

    train_res = trainer.fit(num_epochs=epochs)

    # 4. Evaluate on Test Set
    best_ckpt = CHECKPOINT_DIR / f"{model_name.lower()}_best.pth"
    test_res = evaluate_single_model(
        model_name=model_name.lower(),
        checkpoint_path=best_ckpt,
        batch_size=batch_size,
        device=target_device,
    )

    return {
        "train_results": train_res,
        "test_results": test_res,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a specific segmentation model")
    parser.add_argument("--model", type=str, required=True, choices=["unet", "deeplabv3plus", "segformer"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    train_and_evaluate_model(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device_str=args.device,
    )
