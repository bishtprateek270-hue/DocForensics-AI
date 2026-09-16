"""
DocForensics AI — Hardware & GPU Acceleration Diagnostic Suite
Validates:
1. PyTorch version and CUDA integration
2. GPU hardware detection (Name, VRAM, Compute Capability)
3. CUDA Tensor matrix multiplication & gradient computation test
4. Automatic Mixed Precision (AMP) autocast and GradScaler execution
5. Hardware profile configuration & safe batch size recommendations
"""

import sys
import platform
import torch

from config import cfg, HARDWARE_PROFILES, get_target_device
from src.training.gpu_utils import get_hardware_diagnostics, run_cuda_tensor_test, get_vram_usage


def run_gpu_verification():
    print("=" * 75)
    print("        DocForensics AI — GPU Hardware & Acceleration Diagnostic       ")
    print("=" * 75)

    diag = get_hardware_diagnostics()
    
    print("\n--- 1. System & PyTorch Environment ---")
    print(f"  Operating System      : {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"  Python Version        : {platform.python_version()}")
    print(f"  PyTorch Version       : {diag['pytorch_version']}")
    print(f"  CUDA Version (Torch)  : {diag['cuda_version_torch']}")
    print(f"  CUDA Available        : {diag['cuda_available']}")

    if not diag["cuda_available"]:
        print("\n[-] CUDA is NOT available. PyTorch cannot access GPU acceleration.")
        return False

    print("\n--- 2. NVIDIA GPU Hardware Specifications ---")
    print(f"  GPU Device Count      : {diag['device_count']}")
    print(f"  GPU Model Name        : {diag['gpu_name']}")
    print(f"  Compute Capability    : {diag['gpu_compute_capability']}")
    print(f"  Total VRAM Available  : {diag['total_memory_gb']} GB ({int(diag['total_memory_gb'] * 1024)} MB)")
    print(f"  Current Allocated VRAM: {diag['allocated_memory_mb']} MB")
    print(f"  Current Reserved VRAM : {diag['reserved_memory_mb']} MB")

    print("\n--- 3. CUDA Real Tensor & AMP Execution Test ---")
    test_res = run_cuda_tensor_test()
    if test_res["status"] == "PASSED":
        print(f"  [PASS] Real CUDA Matrix Multiplication & Backward Pass: PASSED")
        print(f"  [PASS] Automatic Mixed Precision (AMP FP16/BF16) Test   : PASSED")
        print(f"  [PASS] Peak VRAM Allocated in Test                     : {test_res['peak_mb']} MB")
    else:
        print(f"  [FAIL] CUDA Computation Test Failed: {test_res.get('reason')}")
        return False

    print("\n--- 4. Hardware Profiles & Batch Size Configurations ---")
    print(f"  Active Profile               : {cfg.hardware_profile}")
    print(f"  Recommended Profile          : {diag['recommended_profile']}")
    print(f"  Recommended Start Batch Size : {diag['recommended_batch_size']} (Resolution: 512x512 with AMP)")
    print("-" * 75)
    print("  Available Environment Presets:")
    for pname, pinfo in HARDWARE_PROFILES.items():
        print(f"    • [{pname:<14}] -> Device: {pinfo['device']}, Batch: {pinfo['batch_size']}, Workers: {pinfo['num_workers']}, AMP: {pinfo['mixed_precision']}")

    print("\n" + "=" * 75)
    print("  FINAL STATUS: [GPU READY FOR ACCELERATED TRAINING]")
    print("=" * 75 + "\n")
    return True


if __name__ == "__main__":
    success = run_gpu_verification()
    if not success:
        sys.exit(1)
