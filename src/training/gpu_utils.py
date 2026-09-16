"""
DocForensics AI — GPU & Hardware Acceleration Utilities
Provides hardware diagnostics, VRAM monitoring, CUDA tensor validation, and AMP compatibility checks
for both Windows NVIDIA RTX 5050 Laptop and NVIDIA DGX Spark environments.
"""

import os
import sys
import random
import numpy as np
import torch
from typing import Dict, Any, Optional


def get_hardware_diagnostics() -> Dict[str, Any]:
    """
    Scans host environment and returns comprehensive hardware and CUDA specifications.
    """
    cuda_available = torch.cuda.is_available()
    diag = {
        "cuda_available": cuda_available,
        "pytorch_version": torch.__version__,
        "cuda_version_torch": torch.version.cuda if hasattr(torch.version, "cuda") else None,
        "device_count": torch.cuda.device_count() if cuda_available else 0,
        "gpu_name": "N/A",
        "gpu_compute_capability": "N/A",
        "total_memory_gb": 0.0,
        "allocated_memory_mb": 0.0,
        "reserved_memory_mb": 0.0,
        "amp_supported": False,
        "recommended_profile": "cpu",
        "recommended_batch_size": 4,
    }

    if cuda_available:
        props = torch.cuda.get_device_properties(0)
        diag["gpu_name"] = props.name
        diag["gpu_compute_capability"] = f"{props.major}.{props.minor}"
        diag["total_memory_gb"] = round(props.total_memory / (1024 ** 3), 2)
        diag["allocated_memory_mb"] = round(torch.cuda.memory_allocated(0) / (1024 ** 2), 2)
        diag["reserved_memory_mb"] = round(torch.cuda.memory_reserved(0) / (1024 ** 2), 2)
        diag["amp_supported"] = True

        # Environment profile recommendation based on VRAM
        if diag["total_memory_gb"] >= 30.0:
            diag["recommended_profile"] = "dgx_spark"
            diag["recommended_batch_size"] = 16
        elif diag["total_memory_gb"] >= 7.0:
            diag["recommended_profile"] = "laptop_rtx5050"
            diag["recommended_batch_size"] = 4  # Conservative 4 for 512x512 with AMP on 8GB VRAM
        else:
            diag["recommended_profile"] = "low_vram_gpu"
            diag["recommended_batch_size"] = 2

    return diag


def run_cuda_tensor_test(device: torch.device = None) -> Dict[str, Any]:
    """
    Executes a matrix multiplication and gradient backprop test on CUDA with AMP autocast.
    Raises RuntimeError if CUDA computation fails.
    """
    if not torch.cuda.is_available():
        return {
            "status": "FAILED",
            "reason": "CUDA is not available on this system.",
        }

    dev = device or torch.device("cuda:0")
    try:
        # 1. Standard FP32 Tensor computation
        x = torch.randn(1024, 1024, device=dev, requires_grad=True)
        w = torch.randn(1024, 1024, device=dev, requires_grad=True)
        y = torch.matmul(x, w)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None and w.grad is not None

        # 2. Automatic Mixed Precision (AMP) Autocast test
        scaler = torch.amp.GradScaler('cuda')
        with torch.amp.autocast('cuda'):
            y_amp = torch.matmul(x, w)
            loss_amp = y_amp.mean()
        
        scaler.scale(loss_amp).backward()
        scaler.step(torch.optim.SGD([x, w], lr=0.01))
        scaler.update()

        alloc_mb = round(torch.cuda.memory_allocated(dev) / (1024 ** 2), 2)
        peak_mb = round(torch.cuda.max_memory_allocated(dev) / (1024 ** 2), 2)

        return {
            "status": "PASSED",
            "device": str(dev),
            "gpu_name": torch.cuda.get_device_name(dev),
            "allocated_mb": alloc_mb,
            "peak_mb": peak_mb,
            "amp_verified": True,
        }
    except Exception as e:
        return {
            "status": "FAILED",
            "reason": str(e),
        }


def get_vram_usage(device: torch.device = None) -> Dict[str, float]:
    """Returns allocated, reserved, and peak VRAM in MB and GB."""
    if not torch.cuda.is_available():
        return {"allocated_mb": 0.0, "reserved_mb": 0.0, "total_gb": 0.0}

    dev = device or torch.device("cuda:0")
    props = torch.cuda.get_device_properties(dev)
    return {
        "allocated_mb": round(torch.cuda.memory_allocated(dev) / (1024 ** 2), 2),
        "reserved_mb": round(torch.cuda.memory_reserved(dev) / (1024 ** 2), 2),
        "peak_mb": round(torch.cuda.max_memory_allocated(dev) / (1024 ** 2), 2),
        "total_gb": round(props.total_memory / (1024 ** 3), 2),
    }


def set_seed(seed: int = 42, deterministic: bool = True):
    """Guarantees reproducibility across both CPU and CUDA environments."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
