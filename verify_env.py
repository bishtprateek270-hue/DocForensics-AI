#!/usr/bin/env python3
"""
DocForensics AI — Environment & Hardware Verification Script
Checks Python, PyTorch, CUDA/GPU acceleration, core packages, and project directories.
"""

import sys
import os
import platform
from pathlib import Path


def print_banner(title: str):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


def check_python():
    print_banner("1. Python Environment")
    print(f" Python Executable : {sys.executable}")
    print(f" Python Version    : {platform.python_version()}")
    print(f" Platform / OS     : {platform.system()} {platform.release()} ({platform.architecture()[0]})")
    
    if sys.version_info >= (3, 9):
        print(" [OK] Python version meets minimum requirement (>= 3.9).")
    else:
        print(" [WARN] Recommended Python version is >= 3.9.")


def check_core_libraries():
    print_banner("2. Core Dependencies & Versions")
    packages = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("cv2", "OpenCV"),
        ("albumentations", "Albumentations"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("matplotlib", "Matplotlib"),
        ("sklearn", "Scikit-Learn"),
        ("PIL", "Pillow"),
        ("tqdm", "tqdm"),
    ]
    
    all_ok = True
    for module_name, display_name in packages:
        try:
            mod = __import__(module_name)
            ver = getattr(mod, "__version__", "Installed")
            print(f"  + {display_name:<16} : {ver:<15} [AVAILABLE]")
        except ImportError as e:
            print(f"  x {display_name:<16} : NOT FOUND       [MISSING: {e}]")
            all_ok = False
            
    if all_ok:
        print("\n [OK] All core dependencies are successfully installed.")
    else:
        print("\n [WARN] Some dependencies are missing. Run `pip install -r requirements.txt`.")
    return all_ok


def check_pytorch_and_gpu():
    print_banner("3. PyTorch & GPU / Acceleration Verification")
    try:
        import torch
        print(f" PyTorch Version       : {torch.__version__}")
        print(f" CUDA Compiled Version : {torch.version.cuda if torch.version.cuda else 'None (CPU build)'}")
        print(f" CUDA Available        : {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            vram_gb = torch.cuda.get_device_properties(current_device).total_memory / (1024 ** 3)
            
            print(f" CUDA Device Count     : {device_count}")
            print(f" Primary GPU Device    : {device_name}")
            print(f" Dedicated VRAM        : {vram_gb:.2f} GB")
            print(f" cuDNN Enabled         : {torch.backends.cudnn.enabled}")
            if torch.backends.cudnn.is_available():
                print(f" cuDNN Version         : {torch.backends.cudnn.version()}")
                
            # Perform dummy tensor test on GPU
            x = torch.randn(1000, 1000, device="cuda")
            y = torch.matmul(x, x)
            print(" [OK] GPU Tensor Matrix Multiplication test PASSED on CUDA device.")
        else:
            print(" [INFO] GPU (CUDA) is not available. PyTorch will run on CPU.")
            x = torch.randn(100, 100)
            y = torch.matmul(x, x)
            print(" [OK] CPU Tensor Matrix Multiplication test PASSED.")
            
    except Exception as e:
        print(f" [ERROR] PyTorch verification failed: {e}")


def check_project_structure():
    print_banner("4. Project Directory Structure Check")
    base_dir = Path(__file__).resolve().parent
    expected_folders = [
        "data/raw",
        "data/processed",
        "data/synthetic",
        "src/preprocessing",
        "src/models",
        "src/training",
        "src/evaluation",
        "src/inference",
        "src/ocr",
        "notebooks",
        "tests",
        "backend",
        "frontend",
    ]
    
    for folder in expected_folders:
        path = base_dir / folder
        status = "[EXISTS]" if path.exists() and path.is_dir() else "[CREATED]"
        path.mkdir(parents=True, exist_ok=True)
        print(f"  - {folder:<24} : {status}")
        
    print("\n [OK] Project directory hierarchy validated.")


def main():
    print("=================================================================")
    print("      DocForensics AI - Environment & System Diagnostic          ")
    print("=================================================================")
    check_python()
    check_core_libraries()
    check_pytorch_and_gpu()
    check_project_structure()
    print_banner("Verification Completed Successfully")


if __name__ == "__main__":
    main()
