"""
DocForensics AI — Phase 10 Final Master Benchmark Evaluation
Executes the single frozen test set evaluation comparing:
- U-Net Baseline
- DeepLabV3+
- SegFormer-B0
- Phase 7 Dual-Stream RGB + SRM (Locked Baseline)
- Phase 10 High-Resolution Multi-Scale Model
Saves: reports/phase10_master_benchmark.json
"""

import os
import sys
import time
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List
import torch

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import get_target_device, CHECKPOINT_DIR, REPORTS_DIR
from src.preprocessing.dataset import create_dataloaders
from src.models.unet import get_unet_model
from src.models.deeplabv3plus import get_deeplabv3plus_model
from src.models.segformer import get_segformer_model
from src.models.dual_stream_forensics import get_dual_stream_model
from src.evaluation.evaluate_phase10 import evaluate_model_comprehensive


def run_master_benchmark() -> Dict[str, Any]:
    """Runs final evaluation on the untouched frozen test set for all models."""
    device = get_target_device("auto")
    print(f"[*] Running Phase 10 Master Benchmark on {device}...")

    _, _, test_loader = create_dataloaders(
        batch_size=4,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    models_to_evaluate = [
        {
            "name": "U-Net",
            "model_loader": lambda: get_unet_model(in_channels=3, num_classes=1, base_channels=32, device=device),
            "checkpoint": CHECKPOINT_DIR / "unet_best.pth",
            "threshold": 0.50,
            "multiscale": False,
        },
        {
            "name": "DeepLabV3+",
            "model_loader": lambda: get_deeplabv3plus_model(in_channels=3, num_classes=1, pretrained=False, device=device),
            "checkpoint": CHECKPOINT_DIR / "deeplabv3plus_best.pth",
            "threshold": 0.50,
            "multiscale": False,
        },
        {
            "name": "SegFormer-B0",
            "model_loader": lambda: get_segformer_model(in_channels=3, num_classes=1, variant="b0", device=device),
            "checkpoint": CHECKPOINT_DIR / "segformer_best.pth",
            "threshold": 0.50,
            "multiscale": False,
        },
        {
            "name": "Phase 7 Dual-Stream (Locked Baseline)",
            "model_loader": lambda: get_dual_stream_model(in_channels=3, num_classes=1, fusion_type="baseline", pretrained_backbone=False, device=device),
            "checkpoint": CHECKPOINT_DIR / "dual_stream_best.pth",
            "threshold": 0.50,
            "multiscale": False,
        },
        {
            "name": "Phase 10 High-Res Multi-Scale Model",
            "model_loader": lambda: get_dual_stream_model(in_channels=3, num_classes=1, fusion_type="baseline", pretrained_backbone=False, device=device),
            "checkpoint": CHECKPOINT_DIR / "dual_stream_best.pth",
            "threshold": 0.44,
            "multiscale": True,
            "alpha": 0.45,
        },
    ]

    benchmark_results = {}

    for cfg in models_to_evaluate:
        m_name = cfg["name"]
        print(f"\n[*] Evaluating: {m_name}")
        model = cfg["model_loader"]()

        ckpt_path = cfg["checkpoint"]
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            state_dict = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state_dict, strict=False)
            print(f"    Loaded weights from {ckpt_path.name}")
        else:
            print(f"    [!] Warning: Checkpoint not found at {ckpt_path}, evaluating uninitialized.")

        res = evaluate_model_comprehensive(
            model=model,
            dataloader=test_loader,
            device=device,
            threshold=cfg["threshold"],
            multiscale_fusion=cfg.get("multiscale", False),
            alpha=cfg.get("alpha", 0.50),
        )

        benchmark_results[m_name] = {
            "checkpoint_path": str(ckpt_path),
            "decision_threshold": cfg["threshold"],
            "multiscale_fusion": cfg.get("multiscale", False),
            "test_metrics": {
                "test_dice": res["mean_dice"],
                "test_iou": res["mean_iou"],
                "test_precision": res["mean_precision"],
                "test_recall": res["mean_recall"],
                "hard_negative_fpr": res["hard_negative_false_positive_rate"],
                "avg_inference_latency_ms": res["avg_latency_ms"],
            },
            "per_area_bucket_metrics": res["per_area_bucket_metrics"],
            "per_category_metrics": res["per_category_metrics"],
        }

        print(f"    - Test Dice: {res['mean_dice']:.4f} | IoU: {res['mean_iou']:.4f} | Recall: {res['mean_recall']:.4f}")
        print(f"    - Tiny Region Dice (<0.5%): {res['per_area_bucket_metrics'].get('tiny', {}).get('mean_dice', 'N/A')}")
        print(f"    - Hard Negative FPR: {res['hard_negative_false_positive_rate']:.4f}")

    master_report = {
        "benchmark_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "test_samples_count": len(test_loader.dataset),
        "models": benchmark_results,
    }

    out_file = REPORTS_DIR / "phase10_master_benchmark.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(master_report, f, indent=2)

    print(f"\n[+] Master Benchmark successfully written to: {out_file}")
    return master_report


if __name__ == "__main__":
    run_master_benchmark()
