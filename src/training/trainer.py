"""
DocForensics AI — Segmentation Model Trainer (Phase 5 & GPU Acceleration)
Supports:
- Automatic Mixed Precision (AMP) via torch.amp on CUDA
- Real-time GPU VRAM tracking and logging
- Gradient accumulation for memory efficiency
- Portable checkpoint saving & loading with device mapping
- Early stopping and validation metric tracking
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CHECKPOINT_DIR, REPORTS_DIR, cfg, get_target_device
from src.training.losses import BCEDiceLoss
from src.training.gpu_utils import get_vram_usage
from src.evaluation.metrics import compute_batch_metrics, MetricTracker


class Trainer:
    """
    Production-grade, hardware-accelerated Trainer for Document Tampering Segmentation.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
        mixed_precision: bool = True,
        gradient_accumulation_steps: int = 1,
        early_stopping_patience: int = 6,
        model_name: str = "unet",
        seed: int = 42,
    ):
        self.device = device or get_target_device(cfg.training.device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = criterion or BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
        self.optimizer = optimizer or torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.scheduler = scheduler
        self.early_stopping_patience = early_stopping_patience
        self.model_name = model_name
        self.seed = seed

        self.gradient_accumulation_steps = max(1, gradient_accumulation_steps)
        self.use_amp = mixed_precision and (self.device.type == "cuda")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.best_val_dice = -1.0
        self.best_epoch = 0
        self.patience_counter = 0

        self.history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "val_dice": [],
            "val_iou": [],
            "val_precision": [],
            "val_recall": [],
            "lr": [],
            "vram_peak_mb": [],
        }

        self.best_checkpoint_path = CHECKPOINT_DIR / f"{self.model_name}_best.pth"
        self.latest_checkpoint_path = CHECKPOINT_DIR / f"{self.model_name}_latest.pth"
        self.history_path = REPORTS_DIR / f"{self.model_name}_training_history.json"

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Runs one full training epoch with AMP and gradient accumulation."""
        self.model.train()
        tracker = MetricTracker()
        self.optimizer.zero_grad()

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch:02d} [Train]", leave=False)
        for step, batch in enumerate(pbar):
            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)

            with torch.amp.autocast(self.device.type, enabled=self.use_amp):
                logits = self.model(images)
                loss = self.criterion(logits, masks)
                loss_scaled = loss / self.gradient_accumulation_steps

            if self.use_amp:
                self.scaler.scale(loss_scaled).backward()
                if (step + 1) % self.gradient_accumulation_steps == 0 or (step + 1) == len(self.train_loader):
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                loss_scaled.backward()
                if (step + 1) % self.gradient_accumulation_steps == 0 or (step + 1) == len(self.train_loader):
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            batch_metrics = compute_batch_metrics(logits, masks)
            tracker.update(loss.item(), batch_metrics, batch_size=images.size(0))
            
            postfix = {"loss": f"{loss.item():.4f}", "dice": f"{batch_metrics['dice']:.4f}"}
            if self.device.type == "cuda":
                vram = get_vram_usage(self.device)
                postfix["vram"] = f"{vram['allocated_mb']}MB"
            pbar.set_postfix(postfix)

        return tracker.compute()

    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Runs validation evaluation over val_loader."""
        self.model.eval()
        tracker = MetricTracker()

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch:02d} [Val]", leave=False)
            for batch in pbar:
                images = batch["image"].to(self.device, non_blocking=True)
                masks = batch["mask"].to(self.device, non_blocking=True)

                with torch.amp.autocast(self.device.type, enabled=self.use_amp):
                    logits = self.model(images)
                    loss = self.criterion(logits, masks)

                batch_metrics = compute_batch_metrics(logits, masks)
                tracker.update(loss.item(), batch_metrics, batch_size=images.size(0))

        return tracker.compute()

    def fit(self, num_epochs: int = 20, resume_from: Optional[Path] = None) -> Dict[str, Any]:
        """Main training loop across all epochs with resume and checkpointing."""
        start_epoch = 1

        if resume_from is not None and Path(resume_from).exists():
            print(f"[*] Resuming training state from checkpoint: {resume_from}")
            ckpt = torch.load(resume_from, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt and self.optimizer is not None:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scaler_state_dict" in ckpt and self.use_amp:
                self.scaler.load_state_dict(ckpt["scaler_state_dict"])
            if "history" in ckpt:
                self.history = ckpt["history"]
            if "val_metrics" in ckpt and "dice" in ckpt["val_metrics"]:
                self.best_val_dice = ckpt["val_metrics"]["dice"]
            elif "val_dice" in ckpt:
                self.best_val_dice = ckpt["val_dice"]
            self.best_epoch = ckpt.get("epoch", 1)
            start_epoch = ckpt.get("epoch", 0) + 1
            print(f"[*] Resumed from epoch {ckpt.get('epoch')}. Best Val Dice so far: {self.best_val_dice:.4f}")

        vram_info = get_vram_usage(self.device) if self.device.type == "cuda" else {}
        print("=" * 75)
        print(f"  DocForensics AI — Training {self.model_name.upper()} Model on {self.device}  ")
        print(f"  AMP Enabled: {self.use_amp} | Grad Accum: {self.gradient_accumulation_steps}")
        if self.device.type == "cuda":
            print(f"  GPU: {torch.cuda.get_device_name(self.device)} (VRAM: {vram_info.get('total_gb')} GB)")
        print(f"  Epochs: {start_epoch} -> {num_epochs} | Train Batches: {len(self.train_loader)} | Val Batches: {len(self.val_loader)}")
        print("=" * 75)

        start_time = time.time()

        for epoch in range(start_epoch, num_epochs + 1):
            t0 = time.time()
            current_lr = self.optimizer.param_groups[0]["lr"]

            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate_epoch(epoch)

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["dice"])
                else:
                    self.scheduler.step()

            epoch_time = time.time() - t0
            vram_stats = get_vram_usage(self.device)
            peak_vram = vram_stats.get("peak_mb", 0.0)

            # Record history
            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(round(train_metrics["loss"], 4))
            self.history["val_loss"].append(round(val_metrics["loss"], 4))
            self.history["val_dice"].append(round(val_metrics["dice"], 4))
            self.history["val_iou"].append(round(val_metrics["iou"], 4))
            self.history["val_precision"].append(round(val_metrics["precision"], 4))
            self.history["val_recall"].append(round(val_metrics["recall"], 4))
            self.history["lr"].append(current_lr)
            self.history["vram_peak_mb"].append(peak_vram)

            vram_log = f" | Peak VRAM: {peak_vram:.0f}MB" if self.device.type == "cuda" else ""

            # Print formatted epoch summary
            print(
                f"Epoch [{epoch:02d}/{num_epochs:02d}] ({epoch_time:.1f}s{vram_log}) — "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val Dice: {val_metrics['dice']:.4f} | "
                f"Val IoU: {val_metrics['iou']:.4f} | "
                f"Val Prec: {val_metrics['precision']:.4f} | "
                f"Val Rec: {val_metrics['recall']:.4f} | "
                f"LR: {current_lr:.6f}"
            )

            # Portable checkpoint payload
            checkpoint_payload = {
                "epoch": epoch,
                "model_name": self.model_name,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                "scaler_state_dict": self.scaler.state_dict() if self.use_amp else None,
                "val_metrics": val_metrics,
                "val_dice": val_metrics["dice"],
                "val_iou": val_metrics["iou"],
                "history": self.history,
                "seed": self.seed,
                "hardware_profile": cfg.hardware_profile,
                "device_type": self.device.type,
            }

            # Save Latest Checkpoint
            torch.save(checkpoint_payload, self.latest_checkpoint_path)

            # Check if Best Checkpoint
            if val_metrics["dice"] > self.best_val_dice:
                self.best_val_dice = val_metrics["dice"]
                self.best_epoch = epoch
                self.patience_counter = 0

                torch.save(checkpoint_payload, self.best_checkpoint_path)
                print(f"  [*] Saved new best model checkpoint to {self.best_checkpoint_path.name} (Val Dice: {val_metrics['dice']:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stopping_patience:
                    print(f"\n[!] Early stopping triggered at epoch {epoch} (No improvement for {self.early_stopping_patience} epochs).")
                    break

        total_training_time = time.time() - start_time
        print("\n" + "=" * 75)
        print(f"  Training Finished in {total_training_time:.2f}s ({total_training_time/60:.2f} min)")
        print(f"  Best Validation Epoch : {self.best_epoch}")
        print(f"  Best Validation Dice  : {self.best_val_dice:.4f}")
        print(f"  Best Checkpoint Saved : {self.best_checkpoint_path}")
        print("=" * 75 + "\n")

        # Save training history to JSON
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.history_path, "w", encoding="utf-8") as f:
            json.dump({
                "model_name": self.model_name,
                "device": str(self.device),
                "gpu_name": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "CPU",
                "total_training_time_seconds": round(total_training_time, 2),
                "best_epoch": self.best_epoch,
                "best_val_dice": round(self.best_val_dice, 4),
                "history": self.history,
            }, f, indent=2)

        return {
            "best_epoch": self.best_epoch,
            "best_val_dice": self.best_val_dice,
            "training_time": total_training_time,
            "best_checkpoint": str(self.best_checkpoint_path),
            "history": self.history,
        }
