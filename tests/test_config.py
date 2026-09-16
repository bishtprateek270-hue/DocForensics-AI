"""
Unit tests for project configuration integrity.
"""
from config import cfg, BASE_DIR, DATA_DIR, CHECKPOINT_DIR, LOG_DIR


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
    assert cfg.device is not None
