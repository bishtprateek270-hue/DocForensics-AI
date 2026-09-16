"""
DocForensics AI — Multi-Model Benchmark Evaluator (Phase 6)
Evaluates U-Net, DeepLabV3+, and SegFormer on the untouched test set (122 samples).
Computes:
- Overall Test Loss, Dice, IoU, Precision, Recall
- Inference Latency (ms/sample) and Peak VRAM Usage
- Per-category tampering performance breakdown
- Generates master comparison report at reports/model_comparison_benchmark.json.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import BASE_DIR, CHECKPOINT_DIR, REPORTS_DIR, METADATA_CSV, cfg, get_target_device
from src.models.unet import get_unet_model
from src.models.deeplabv3plus import get_deeplabv3plus_model
from src.models.segformer import get_segformer_model
from src.preprocessing.dataset import DocForensicsDataset
from src.training.losses import BCEDiceLoss
from src.training.gpu_utils import get_vram_usage
from src.evaluation.metrics import compute_batch_metrics, MetricTracker


MODEL_BUILDERS = {
    "unet": lambda dev: get_unet_model(in_channels=3, num_classes=1, base_channels=32, device=dev),
    "deeplabv3plus": lambda dev: get_deeplabv3plus_model(in_channels=3, num_classes=1, device=dev),
    "segformer": lambda dev: get_segformer_model(in_channels=3, num_classes=1, variant="b0", device=dev),
}


def evaluate_single_model(
    model_name: str,
    checkpoint_path: Path,
    metadata_path: Path = METADATA_CSV,
    batch_size: int = 4,
    threshold: float = 0.5,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Evaluates a single model architecture on the test set."""
    dev = device or get_target_device(cfg.training.device)
    print(f"\n[*] Evaluating {model_name.upper()} on {dev}...")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    # 1. Dataset & DataLoader
    test_dataset = DocForensicsDataset(
        metadata_path=metadata_path,
        split="test",
        is_training=False,
        image_size=cfg.preprocessing.image_size,
        normalize=True,
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 2. Build Model & Load Checkpoint
    builder = MODEL_BUILDERS.get(model_name.lower())
    if builder is None:
        raise ValueError(f"Unknown model name: {model_name}")

    model = builder(dev)
    checkpoint = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    tracker = MetricTracker()
    category_metrics_map: Dict[str, List[Dict[str, float]]] = {}

    # Inference Latency Benchmark
    inference_times = []
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Eval {model_name.upper()}", leave=False):
            images = batch["image"].to(dev)
            masks = batch["mask"].to(dev)
            is_tampered = batch["is_tampered"]
            manip_types = batch["manipulation_type"]
            source_types = batch["source_type"]
            sample_ids = batch["sample_id"]

            t_start = time.perf_counter()
            logits = model(images)
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            t_end = time.perf_counter()

            batch_size_cur = images.size(0)
            inference_times.append((t_end - t_start) / batch_size_cur * 1000.0)  # ms per sample

            loss = criterion(logits, masks)
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()

            for i in range(batch_size_cur):
                p_mask = preds[i, 0].cpu()
                t_mask = masks[i, 0].cpu()
                tp = (p_mask * t_mask).sum().item()
                fp = (p_mask * (1.0 - t_mask)).sum().item()
                fn = ((1.0 - p_mask) * t_mask).sum().item()
                is_t = is_tampered[i].item()

                if is_t == 0:
                    dice = 1.0 if fp == 0 else 0.0
                    iou = 1.0 if fp == 0 else 0.0
                    prec = 1.0 if fp == 0 else 0.0
                    rec = 1.0
                else:
                    dice = (2.0 * tp + 1e-6) / (2.0 * tp + fp + fn + 1e-6)
                    iou = (tp + 1e-6) / (tp + fp + fn + 1e-6)
                    prec = (tp + 1e-6) / (tp + fp + 1e-6)
                    rec = (tp + 1e-6) / (tp + fn + 1e-6)

                m_type = manip_types[i]
                sample_res = {
                    "sample_id": sample_ids[i],
                    "manipulation_type": m_type,
                    "is_tampered": is_t,
                    "dice": round(dice, 4),
                    "iou": round(iou, 4),
                    "precision": round(prec, 4),
                    "recall": round(rec, 4),
                }

                if m_type not in category_metrics_map:
                    category_metrics_map[m_type] = []
                category_metrics_map[m_type].append(sample_res)

            batch_m = compute_batch_metrics(logits, masks, threshold=threshold)
            tracker.update(loss.item(), batch_m, batch_size=batch_size_cur)

    summary_metrics = tracker.compute()
    avg_inference_ms = float(np.mean(inference_times))
    peak_vram_mb = round(torch.cuda.max_memory_allocated(dev) / (1024 ** 2), 2) if dev.type == "cuda" else 0.0

    # Per-category summary
    category_summary = {}
    for cat, items in category_metrics_map.items():
        category_summary[cat] = {
            "sample_count": len(items),
            "mean_dice": round(float(np.mean([x["dice"] for x in items])), 4),
            "mean_iou": round(float(np.mean([x["iou"] for x in items])), 4),
            "mean_precision": round(float(np.mean([x["precision"] for x in items])), 4),
            "mean_recall": round(float(np.mean([x["recall"] for x in items])), 4),
        }

    report = {
        "model_name": model_name,
        "checkpoint_path": str(checkpoint_path),
        "best_epoch": checkpoint.get("epoch", -1),
        "device_used": str(dev),
        "test_dataset_size": len(test_dataset),
        "avg_inference_latency_ms": round(avg_inference_ms, 2),
        "peak_vram_mb": peak_vram_mb,
        "test_metrics": {
            "test_loss": round(summary_metrics["loss"], 4),
            "test_dice": round(summary_metrics["dice"], 4),
            "test_iou": round(summary_metrics["iou"], 4),
            "test_precision": round(summary_metrics["precision"], 4),
            "test_recall": round(summary_metrics["recall"], 4),
        },
        "per_category_metrics": category_summary,
    }

    out_json = REPORTS_DIR / f"{model_name}_test_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def run_multi_model_benchmark() -> Dict[str, Any]:
    """Evaluates all 3 models and compiles comparison benchmark."""
    print("=" * 75)
    print("    DocForensics AI — Phase 6 Multi-Model Benchmark Suite             ")
    print("=" * 75)

    models_to_eval = [
        ("unet", CHECKPOINT_DIR / "unet_best.pth"),
        ("deeplabv3plus", CHECKPOINT_DIR / "deeplabv3plus_best.pth"),
        ("segformer", CHECKPOINT_DIR / "segformer_best.pth"),
    ]

    all_reports = {}
    for m_name, ckpt_p in models_to_eval:
        if ckpt_p.exists():
            rep = evaluate_single_model(m_name, ckpt_p)
            all_reports[m_name] = rep
        else:
            print(f"[!] Warning: Checkpoint for {m_name} not found at {ckpt_p}. Skipping.")

    comparison_json = REPORTS_DIR / "model_comparison_benchmark.json"
    with open(comparison_json, "w", encoding="utf-8") as f:
        json.dump(all_reports, f, indent=2)

    print(f"\n[+] Master model comparison benchmark saved to: {comparison_json}")
    return all_reports


if __name__ == "__main__":
    run_multi_model_benchmark()
