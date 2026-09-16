"""
Unit tests for project configuration integrity, hardware profiles, and GPU utilities.
"""
import pytest
import torch
from config import cfg, BASE_DIR, DATA_DIR, CHECKPOINT_DIR, LOG_DIR, HARDWARE_PROFILES, get_target_device
from src.training.gpu_utils import get_hardware_diagnostics, run_cuda_tensor_test, get_vram_usage


def test_base_directories_exist():
    assert BASE_DIR.exists(), "Base directory does not exist"
    assert DATA_DIR.exists(), "Data directory does not exist"
    assert CHECKPOINT_DIR.exists(), "Checkpoint directory does not exist"
    assert LOG_DIR.exists(), "Log directory does not exist"


def test_config_initialization():
    assert cfg.project_name == "DocForensics AI"
    assert cfg.preprocessing.image_size == (512, 512)
    assert cfg.model.num_classes == 1
    assert cfg.training.batch_size > 0


def test_hardware_profile_switching():
    # Test switching to dgx_spark profile
    cfg.apply_hardware_profile("dgx_spark")
    assert cfg.hardware_profile == "dgx_spark"
    assert cfg.training.batch_size == 16
    assert cfg.training.num_workers == 8

    # Test switching to laptop_rtx5050 profile
    cfg.apply_hardware_profile("laptop_rtx5050")
    assert cfg.hardware_profile == "laptop_rtx5050"
    assert cfg.training.batch_size == 4


def test_device_validation():
    # CPU should always succeed
    dev_cpu = get_target_device("cpu")
    assert dev_cpu.type == "cpu"

    # Auto should succeed
    dev_auto = get_target_device("auto")
    assert dev_auto.type in ["cuda", "cpu"]


def test_gpu_diagnostics():
    diag = get_hardware_diagnostics()
    assert isinstance(diag, dict)
    assert "cuda_available" in diag
    assert "pytorch_version" in diag
    if torch.cuda.is_available():
        assert diag["cuda_available"] is True
        assert diag["total_memory_gb"] > 0
        assert diag["gpu_name"] != "N/A"

        # Run CUDA tensor test
        test_res = run_cuda_tensor_test()
        assert test_res["status"] == "PASSED"
        assert test_res["amp_verified"] is True
