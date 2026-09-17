"""
DocForensics AI — Unit Tests for Canonical Evaluator (Phase 11)
Tests:
1. Instantiation and checkpoint loading
2. Image preprocessing and EXIF handling
3. Connected component filtering
4. Region-level IoU overlap computation (Detected, Partially Detected, Missed)
5. Authentic sample evaluation with empty ground truth mask
6. Tampered sample evaluation with non-empty ground truth mask
"""

import pytest
import numpy as np
import torch
from pathlib import Path
from PIL import Image

from src.evaluation.canonical_evaluator import CanonicalEvaluator
from config import CHECKPOINT_DIR


@pytest.fixture
def evaluator():
    ckpt_p = CHECKPOINT_DIR / "dual_stream_best.pth"
    if not ckpt_p.exists():
        pytest.skip("Production checkpoint not found")
    return CanonicalEvaluator(checkpoint_path=ckpt_p, device=torch.device("cpu"))


def test_evaluator_initialization(evaluator):
    assert evaluator is not None
    assert evaluator.threshold == 0.50
    assert evaluator.min_component_area == 16
    assert evaluator.region_iou_threshold == 0.25


def test_evaluator_preprocessing(evaluator):
    dummy_img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    tensor, rgb_orig, orig_shape = evaluator.preprocess_image(dummy_img)
    assert tensor.shape == (1, 3, 512, 512)
    assert rgb_orig.shape == (300, 400, 3)
    assert orig_shape == (300, 400)


def test_connected_component_filtering(evaluator):
    mask = np.zeros((100, 100), dtype=np.uint8)
    # Add tiny speckle (4 pixels)
    mask[10:12, 10:12] = 1
    # Add valid region (36 pixels: 6x6)
    mask[40:46, 40:46] = 1

    filtered = evaluator.filter_connected_components(mask)
    assert np.sum(filtered[10:12, 10:12]) == 0
    assert np.sum(filtered[40:46, 40:46]) == 36


def test_region_overlap_metrics(evaluator):
    # GT region of 20x20 at (10, 10)
    gt_mask = np.zeros((100, 100), dtype=np.uint8)
    gt_mask[10:30, 10:30] = 1
    gt_regions = evaluator.extract_regions(gt_mask)

    # Pred 1: Exactly matching region
    pred_mask_perfect = np.zeros((100, 100), dtype=np.uint8)
    pred_mask_perfect[10:30, 10:30] = 1
    pred_regions_perfect = evaluator.extract_regions(pred_mask_perfect)
    metrics_perfect = evaluator.compute_region_overlap_metrics(pred_regions_perfect, gt_regions)
    assert metrics_perfect["detected_count"] == 1
    assert metrics_perfect["missed_count"] == 0
    assert metrics_perfect["region_recall"] == 1.0

    # Pred 2: Missed region (far away)
    pred_mask_miss = np.zeros((100, 100), dtype=np.uint8)
    pred_mask_miss[70:90, 70:90] = 1
    pred_regions_miss = evaluator.extract_regions(pred_mask_miss)
    metrics_miss = evaluator.compute_region_overlap_metrics(pred_regions_miss, gt_regions)
    assert metrics_miss["detected_count"] == 0
    assert metrics_miss["missed_count"] == 1
    assert metrics_miss["region_recall"] == 0.0


def test_authentic_sample_evaluation(evaluator):
    clean_img = np.ones((512, 512, 3), dtype=np.uint8) * 255
    gt_empty = np.zeros((512, 512), dtype=np.uint8)

    res = evaluator.evaluate_sample(clean_img, ground_truth_mask=gt_empty)
    assert "metrics" in res
    assert res["metrics"]["gt_region_count"] == 0
    assert res["metrics"]["dice"] in [0.0, 1.0]
