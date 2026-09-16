"""
DocForensics AI — Document Tampering Detection & Localization
Configuration Module (config.py)

Central configuration for paths, device selection, model parameters,
preprocessing settings, training hyperparameters, and logging.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
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
# Device Configuration
# ==========================================
def get_device() -> torch.device:
    """Auto-detect best available computing device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()
NUM_WORKERS = min(4, os.cpu_count() or 1)


# ==========================================
# Dataset Split & Preparation Configuration
# ==========================================
@dataclass
class DatasetConfig:
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    standardize_ext: str = ".png"                  # Target standardized image format
    min_tampered_area_ratio: float = 0.001         # Minimum fraction of tampered pixels for valid tamper
    max_tampered_area_ratio: float = 0.50          # Maximum fraction of tampered pixels


# ==========================================
# Image & Preprocessing Settings
# ==========================================
@dataclass
class PreprocessingConfig:
    image_size: tuple[int, int] = (512, 512)       # (Height, Width) for model input
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)  # ImageNet normalization
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    ela_quality: int = 90                          # Quality factor for Error Level Analysis
    ela_scale: float = 15.0                        # Visual multiplier for ELA difference
    srm_filter_count: int = 3                      # SRM forensic residual filters


# ==========================================
# Model Architecture Settings
# ==========================================
@dataclass
class ModelConfig:
    architecture: str = "two_stream_unet"          # Options: unet, segformer, two_stream_unet, trufour
    backbone: str = "resnet34"                     # Visual stream backbone
    in_channels: int = 3                           # RGB (3) or Dual-Stream (6/7)
    num_classes: int = 1                           # Binary segmentation mask (0: authentic, 1: tampered)
    pretrained: bool = True
    dropout: float = 0.2


# ==========================================
# Training Hyperparameters
# ==========================================
@dataclass
class TrainingConfig:
    batch_size: int = 8
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    scheduler: str = "cosine"                      # Options: cosine, plateau, step
    early_stopping_patience: int = 10
    mixed_precision: bool = torch.cuda.is_available()  # Automatic Mixed Precision (AMP)
    seed: int = 42
    save_top_k: int = 3


# ==========================================
# Global Project Configuration Singleton
# ==========================================
@dataclass
class Config:
    project_name: str = "DocForensics AI"
    version: str = "1.0.0"
    base_dir: Path = BASE_DIR
    device: torch.device = field(default_factory=get_device)
    num_workers: int = NUM_WORKERS
    
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


# Instantiate default config instance
cfg = Config()

if __name__ == "__main__":
    print(f"=== {cfg.project_name} v{cfg.version} Configuration ===")
    print(f"Base Directory : {cfg.base_dir}")
    print(f"Active Device  : {cfg.device}")
    print(f"Num Workers    : {cfg.num_workers}")
    print(f"Input Size     : {cfg.preprocessing.image_size}")
    print(f"Architecture   : {cfg.model.architecture} (Backbone: {cfg.model.backbone})")
    print(f"Batch Size     : {cfg.training.batch_size} | Epochs: {cfg.training.num_epochs}")
    print(f"AMP Enabled    : {cfg.training.mixed_precision}")
