"""
DocForensics AI — Central Hardware & Training Configuration Module
Supports seamless execution across:
1. Windows Laptop with NVIDIA GeForce RTX 5050 (8GB VRAM)
2. NVIDIA DGX Spark (Multi-GPU / High-VRAM)
3. Standard CPU environments
"""

import os
import platform
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, Dict, Any
import torch


# ==========================================
# Base Directory Paths
# ==========================================
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_PUBLIC_DIR = RAW_DATA_DIR / "public"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

SYNTHETIC_DIR = DATA_DIR / "synthetic"
SYNTHETIC_ORIGINALS_DIR = SYNTHETIC_DIR / "originals"
SYNTHETIC_TAMPERED_DIR = SYNTHETIC_DIR / "tampered"
SYNTHETIC_MASKS_DIR = SYNTHETIC_DIR / "masks"
SYNTHETIC_METADATA_CSV = SYNTHETIC_DIR / "metadata.csv"

TRAIN_DATA_DIR = DATA_DIR / "train"
VAL_DATA_DIR = DATA_DIR / "val"
TEST_DATA_DIR = DATA_DIR / "test"
METADATA_CSV = DATA_DIR / "metadata.csv"
DATASET_SUMMARY_JSON = DATA_DIR / "dataset_summary.json"

CHECKPOINT_DIR = BASE_DIR / "checkpoints"
LOG_DIR = BASE_DIR / "logs"
OUTPUT_DIR = BASE_DIR / "outputs"
REPORTS_DIR = BASE_DIR / "reports"

# Ensure runtime directories exist
for path in [DATA_DIR, RAW_DATA_DIR, RAW_PUBLIC_DIR, PROCESSED_DATA_DIR,
             SYNTHETIC_DIR, SYNTHETIC_ORIGINALS_DIR, SYNTHETIC_TAMPERED_DIR, SYNTHETIC_MASKS_DIR,
             TRAIN_DATA_DIR, VAL_DATA_DIR, TEST_DATA_DIR,
             CHECKPOINT_DIR, LOG_DIR, OUTPUT_DIR, REPORTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)


# ==========================================
# Hardware & Device Management
# ==========================================
def get_target_device(device_str: str = "auto") -> torch.device:
    """
    Selects compute device with strict validation.
    If 'cuda' is explicitly requested and unavailable, raises RuntimeError instead of silent CPU fallback.
    """
    device_str = device_str.lower().strip()
    if device_str == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "[-] CUDA device explicitly requested, but torch.cuda.is_available() is False. "
                "Ensure NVIDIA GPU drivers and a CUDA-compatible PyTorch build are installed. "
                "Will NOT silently fallback to CPU."
            )
        return torch.device("cuda")
    elif device_str == "cpu":
        return torch.device("cpu")
    elif device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device(device_str)


# ==========================================
# Hardware Profile Definitions
# ==========================================
@dataclass
class HardwareProfile:
    name: str = "auto"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 4                            # Conservative default for RTX 5050 8GB
    num_workers: int = 0 if platform.system() == "Windows" else 4
    pin_memory: bool = torch.cuda.is_available()
    mixed_precision: bool = torch.cuda.is_available()  # Automatic Mixed Precision (AMP)
    gradient_accumulation_steps: int = 1


HARDWARE_PROFILES: Dict[str, Dict[str, Any]] = {
    "laptop_rtx5050": {
        "device": "cuda",
        "batch_size": 4,                           # Safe 512x512 batch size on 8GB VRAM
        "num_workers": 0 if platform.system() == "Windows" else 2,
        "pin_memory": True,
        "mixed_precision": True,
        "gradient_accumulation_steps": 2,          # Effective batch size = 8
    },
    "dgx_spark": {
        "device": "cuda",
        "batch_size": 16,                          # High throughput on DGX VRAM
        "num_workers": 8,
        "pin_memory": True,
        "mixed_precision": True,
        "gradient_accumulation_steps": 1,
    },
    "cpu": {
        "device": "cpu",
        "batch_size": 4,
        "num_workers": 0,
        "pin_memory": False,
        "mixed_precision": False,
        "gradient_accumulation_steps": 1,
    },
}


# ==========================================
# Dataset Configuration
# ==========================================
@dataclass
class DatasetConfig:
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    standardize_ext: str = ".png"
    min_tampered_area_ratio: float = 0.001
    max_tampered_area_ratio: float = 0.50


# ==========================================
# Preprocessing Configuration
# ==========================================
@dataclass
class PreprocessingConfig:
    image_size: Tuple[int, int] = (512, 512)       # (Height, Width)
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)  # ImageNet normalization
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    ela_quality: int = 90
    ela_scale: float = 15.0
    srm_filter_count: int = 3


# ==========================================
# Model Architecture Settings
# ==========================================
@dataclass
class ModelConfig:
    architecture: str = "unet"                     # Options: unet, segformer, two_stream_unet
    backbone: str = "custom"                       # Visual stream backbone
    in_channels: int = 3
    num_classes: int = 1
    base_channels: int = 32
    dropout: float = 0.1


# ==========================================
# Training Hyperparameters
# ==========================================
@dataclass
class TrainingConfig:
    batch_size: int = 4                            # Default starting batch size on RTX 5050
    num_epochs: int = 25
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    scheduler: str = "cosine"                      # Options: cosine, plateau, step
    early_stopping_patience: int = 6
    mixed_precision: bool = torch.cuda.is_available()  # AMP (FP16)
    gradient_accumulation_steps: int = 1
    num_workers: int = 0 if platform.system() == "Windows" else 4
    pin_memory: bool = torch.cuda.is_available()
    seed: int = 42
    save_top_k: int = 3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================================
# Global Project Configuration Singleton
# ==========================================
@dataclass
class Config:
    project_name: str = "DocForensics AI"
    version: str = "1.0.0"
    base_dir: Path = BASE_DIR
    hardware_profile: str = "laptop_rtx5050" if torch.cuda.is_available() else "cpu"

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def apply_hardware_profile(self, profile_name: str):
        """Applies hardware settings preset (laptop_rtx5050, dgx_spark, cpu)."""
        if profile_name not in HARDWARE_PROFILES:
            raise ValueError(f"Unknown hardware profile '{profile_name}'. Available: {list(HARDWARE_PROFILES.keys())}")
        
        p = HARDWARE_PROFILES[profile_name]
        self.hardware_profile = profile_name
        self.training.device = p["device"]
        self.training.batch_size = p["batch_size"]
        self.training.num_workers = p["num_workers"]
        self.training.pin_memory = p["pin_memory"]
        self.training.mixed_precision = p["mixed_precision"]
        self.training.gradient_accumulation_steps = p.get("gradient_accumulation_steps", 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_name": self.project_name,
            "version": self.version,
            "hardware_profile": self.hardware_profile,
            "dataset": asdict(self.dataset),
            "preprocessing": asdict(self.preprocessing),
            "model": asdict(self.model),
            "training": asdict(self.training),
        }


# Instantiate default config instance
cfg = Config()

if __name__ == "__main__":
    print(f"=== {cfg.project_name} v{cfg.version} Configuration ===")
    print(f"Base Directory   : {cfg.base_dir}")
    print(f"Hardware Profile : {cfg.hardware_profile}")
    print(f"Active Device    : {cfg.training.device}")
    print(f"Batch Size       : {cfg.training.batch_size} (Grad Accum: {cfg.training.gradient_accumulation_steps})")
    print(f"Workers          : {cfg.training.num_workers} | Pin Memory: {cfg.training.pin_memory}")
    print(f"AMP Enabled      : {cfg.training.mixed_precision}")
    print(f"Input Resolution : {cfg.preprocessing.image_size}")
