"""
DocForensics AI — Phase 10 Controlled Validation Experiments & Ablation Runner
Executes progressive, controlled validation experiments:
- Exp A: Baseline RGB + SRM reproduction on validation set
- Exp B: Baseline + High-Resolution Patch Analysis
- Exp C: Baseline + Hard-Case & Hard-Negative Dataset Expansion
- Exp D: Baseline + Advanced Losses (Focal / Tversky)
- Exp E: Baseline + Frequency / DCT stream
- Exp F: Final Compound Selection & Checkpointing
Logs all metrics into reports/phase10_experiments/
"""

import os
import sys
import time
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_target_device, CHECKPOINT_DIR, REPORTS_DIR, METADATA_CSV
from src.preprocessing.dataset import DocForensicsDataset, create_dataloaders
from src.models.dual_stream_forensics import get_dual_stream_model
from src.models.multi_stream_forensics import get_multi_stream_model
from src.training.losses import get_loss_function
from src.training.gpu_utils import get_vram_usage
from src.evaluation.evaluate_phase10 import evaluate_model_comprehensive, calibrate_threshold_on_validation


EXPERIMENTS_DIR = REPORTS_DIR / "phase10_experiments"
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)


def run_experiment(
    exp_id: str,
    stream_mode: str = "rgb_srm",
    loss_name: str = "bce_dice",
    use_patches: bool = False,
    multiscale_fusion: bool = False,
    alpha: float = 0.50,
    epochs: int = 15,
    lr: float = 1e-4,
    device_str: str = "auto",
) -> Dict[str, Any]:
    """
    Executes a single controlled experiment, evaluates on validation data, and logs metrics.
    """
    out_file = EXPERIMENTS_DIR / f"{exp_id.lower()}_metrics.json"
    if out_file.exists():
        print(f"[+] Found completed results for {exp_id} at {out_file.name}, loading...")
        with open(out_file, "r", encoding="utf-8") as f:
            return json.load(f)

    device = get_target_device(device_str)
    print(f"\n{'='*70}")
    print(f"[*] Starting Experiment: {exp_id}")
    print(f"    - Stream Mode: {stream_mode} | Loss: {loss_name} | Device: {device}")
    print(f"    - High-Res Patches: {use_patches} | Multi-scale Fusion: {multiscale_fusion}")
    print(f"{'='*70}")

    train_loader, val_loader, _ = create_dataloaders(
        batch_size=4,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # 1. Initialize Architecture
    if stream_mode == "rgb_srm":
        model = get_dual_stream_model(fusion_type="baseline", pretrained_backbone=True, device=device)
        # Check if baseline checkpoint exists for Exp A & B
        baseline_ckpt = CHECKPOINT_DIR / "dual_stream_best.pth"
        if baseline_ckpt.exists() and exp_id in ["Exp_A_Baseline_Reproduction", "Exp_B_HighRes_Patches"]:
            print(f"[*] Loading locked Phase 7 baseline weights from {baseline_ckpt.name}...")
            ckpt = torch.load(baseline_ckpt, map_location=device)
            state_dict = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state_dict, strict=False)
    else:
        model = get_multi_stream_model(stream_mode=stream_mode, rgb_backbone="resnet34", pretrained=True, device=device)

    # 2. Training (if required for new architectures / losses)
    criterion = get_loss_function(loss_name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

    best_val_dice = 0.0
    t_train_start = time.perf_counter()

    if exp_id not in ["Exp_A_Baseline_Reproduction", "Exp_B_HighRes_Patches"]:
        print(f"[*] Training {exp_id} for {epochs} epochs on {len(train_loader.dataset)} samples...")
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0.0
            for batch in train_loader:
                imgs = batch["image"].to(device)
                masks = batch["mask"].to(device)

                optimizer.zero_grad()
                with torch.amp.autocast(device.type, enabled=(device.type == "cuda")):
                    logits = model(imgs)
                    loss = criterion(logits, masks)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss.item()

            # Periodic fast validation check
            if epoch % 3 == 0 or epoch == epochs:
                val_res = evaluate_model_comprehensive(model, val_loader, device, threshold=0.50)
                print(f"    Epoch {epoch:02d}/{epochs:02d} - Loss: {epoch_loss/len(train_loader):.4f} - Val Dice: {val_res['mean_dice']:.4f}")
                if val_res["mean_dice"] > best_val_dice:
                    best_val_dice = val_res["mean_dice"]
                    # Save intermediate experiment checkpoint
                    ckpt_path = CHECKPOINT_DIR / f"phase10_{exp_id.lower()}_best.pth"
                    torch.save({"model_state_dict": model.state_dict(), "exp_id": exp_id}, ckpt_path)

    t_train_duration = time.perf_counter() - t_train_start

    # 3. Threshold Calibration on Validation Split
    best_th, calib_info = calibrate_threshold_on_validation(model, val_loader, device)

    # 4. Final Comprehensive Validation Evaluation
    val_results = evaluate_model_comprehensive(
        model=model,
        dataloader=val_loader,
        device=device,
        threshold=best_th,
        use_patch_engine=use_patches,
        multiscale_fusion=multiscale_fusion,
        alpha=alpha,
    )

    peak_vram_mb = round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2) if device.type == "cuda" else 0.0

    exp_record = {
        "experiment_id": exp_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "stream_mode": stream_mode,
        "loss_function": loss_name,
        "use_patches": use_patches,
        "multiscale_fusion": multiscale_fusion,
        "fusion_alpha": alpha,
        "optimal_val_threshold": best_th,
        "val_metrics": {
            "val_dice": val_results["mean_dice"],
            "val_iou": val_results["mean_iou"],
            "val_precision": val_results["mean_precision"],
            "val_recall": val_results["mean_recall"],
            "hard_negative_fpr": val_results["hard_negative_false_positive_rate"],
            "avg_latency_ms": val_results["avg_latency_ms"],
            "peak_vram_mb": peak_vram_mb,
        },
        "per_category_val_metrics": val_results["per_category_metrics"],
        "per_area_bucket_val_metrics": val_results["per_area_bucket_metrics"],
        "training_duration_seconds": round(t_train_duration, 2),
    }

    # Save Experiment JSON
    out_file = EXPERIMENTS_DIR / f"{exp_id.lower()}_metrics.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(exp_record, f, indent=2)

    print(f"[+] Experiment {exp_id} Complete!")
    print(f"    - Optimal Threshold: {best_th:.2f}")
    print(f"    - Val Dice: {val_results['mean_dice']:.4f} | IoU: {val_results['mean_iou']:.4f} | Recall: {val_results['mean_recall']:.4f}")
    print(f"    - Tiny Region Dice (<0.5%): {val_results['per_area_bucket_metrics'].get('tiny', {}).get('mean_dice', 'N/A')}")
    print(f"    - Hard Negative FPR: {val_results['hard_negative_false_positive_rate']:.4f}")
    print(f"    - Logged to: {out_file.name}")

    return exp_record


def run_full_ablation_suite():
    """Runs all progressive ablation experiments A through F."""
    records = []

    # Exp A: Baseline RGB + SRM reproduction
    rec_a = run_experiment("Exp_A_Baseline_Reproduction", stream_mode="rgb_srm", loss_name="bce_dice", use_patches=False)
    records.append(rec_a)

    # Exp B: Baseline + High-Resolution Overlapping Patches (Multi-Scale Fusion)
    rec_b = run_experiment("Exp_B_HighRes_Patches", stream_mode="rgb_srm", loss_name="bce_dice", use_patches=True, multiscale_fusion=True, alpha=0.40)
    records.append(rec_b)

    # Exp C: Baseline + Advanced Tversky Loss
    rec_c = run_experiment("Exp_C_Tversky_Loss", stream_mode="rgb_srm", loss_name="tversky", epochs=8)
    records.append(rec_c)

    # Exp D: Multi-Stream RGB + SRM + DCT Frequency Stream
    rec_d = run_experiment("Exp_D_DCT_Frequency", stream_mode="rgb_srm_dct", loss_name="bce_dice", epochs=8)
    records.append(rec_d)

    # Exp E: Final Compound Multi-Scale Model
    rec_e = run_experiment("Exp_E_Final_Compound", stream_mode="rgb_srm", loss_name="tversky", use_patches=True, multiscale_fusion=True, alpha=0.45, epochs=10)
    records.append(rec_e)

    # Save summary table
    summary_path = REPORTS_DIR / "phase10_ablation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print("\n[+] Full Phase 10 Ablation Suite successfully executed!")
    return records


if __name__ == "__main__":
    run_full_ablation_suite()
