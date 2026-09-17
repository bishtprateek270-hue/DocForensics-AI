"""
DocForensics AI — Reference Verification Module (Phase 11)
Performs deterministic comparison between a document under analysis and a trusted reference original.
Clearly labeled as: "Reference comparison" (NOT "AI tampering detection").
"""

import cv2
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
from PIL import Image


def align_images(im1_gray: np.ndarray, im2_gray: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Aligns im2 (test document) to im1 (trusted reference) using ORB feature matching and Homography.
    """
    MAX_FEATURES = 2000
    GOOD_MATCH_PERCENT = 0.15

    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2_gray, None)

    if descriptors1 is None or descriptors2 is None or len(keypoints1) < 10 or len(keypoints2) < 10:
        # Fallback to simple resize
        return cv2.resize(im2_gray, (im1_gray.shape[1], im1_gray.shape[0])), False

    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)
    matches = sorted(matches, key=lambda x: x.distance)

    num_good_matches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:max(10, num_good_matches)]

    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    h_matrix, mask = cv2.findHomography(points2, points1, cv2.RANSAC)
    if h_matrix is None:
        return cv2.resize(im2_gray, (im1_gray.shape[1], im1_gray.shape[0])), False

    height, width = im1_gray.shape
    aligned = cv2.warpPerspective(im2_gray, h_matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)
    return aligned, True


def compare_with_reference(
    test_image_input: Any,
    reference_image_input: Any,
    diff_threshold: int = 35,
    min_diff_area: int = 24
) -> Dict[str, Any]:
    """
    Compares the document under test against a trusted reference original.
    
    Returns structured results:
    - analysis_type: "Reference comparison"
    - is_identical: True/False
    - discrepancy_region_count: int
    - discrepancy_regions: List of bounding boxes
    - difference_percentage: float
    - alignment_success: bool
    """
    # Load test image
    if isinstance(test_image_input, (str, Path)):
        test_rgb = np.array(Image.open(test_image_input).convert("RGB"))
    elif isinstance(test_image_input, Image.Image):
        test_rgb = np.array(test_image_input.convert("RGB"))
    else:
        test_rgb = test_image_input.copy()

    # Load reference image
    if isinstance(reference_image_input, (str, Path)):
        ref_rgb = np.array(Image.open(reference_image_input).convert("RGB"))
    elif isinstance(reference_image_input, Image.Image):
        ref_rgb = np.array(reference_image_input.convert("RGB"))
    else:
        ref_rgb = reference_image_input.copy()

    ref_gray = cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2GRAY)
    test_gray = cv2.cvtColor(test_rgb, cv2.COLOR_RGB2GRAY)

    # 1. Align test image to reference geometry
    aligned_test_gray, aligned_ok = align_images(ref_gray, test_gray)

    # 2. Compute absolute difference map
    diff = cv2.absdiff(ref_gray, aligned_test_gray)
    _, thresh_diff = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned_diff = cv2.morphologyEx(thresh_diff, cv2.MORPH_OPEN, kernel)

    # 3. Extract discrepancy regions
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned_diff, connectivity=8)
    h, w = ref_gray.shape
    total_diff_pixels = 0
    discrepancy_regions = []

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_diff_area:
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            rw = int(stats[i, cv2.CC_STAT_WIDTH])
            rh = int(stats[i, cv2.CC_STAT_HEIGHT])
            total_diff_pixels += area
            discrepancy_regions.append({
                "region_id": len(discrepancy_regions) + 1,
                "bbox": [x, y, x + rw, y + rh],
                "pixel_area": int(area),
                "percentage_of_document": round(float(area) / (h * w) * 100.0, 4)
            })

    diff_pct = round(float(total_diff_pixels) / (h * w) * 100.0, 4)
    is_identical = (len(discrepancy_regions) == 0)

    return {
        "analysis_type": "Reference comparison",
        "reference_alignment_applied": aligned_ok,
        "is_structurally_identical": is_identical,
        "discrepancy_region_count": len(discrepancy_regions),
        "discrepancy_percentage": diff_pct,
        "discrepancy_regions": discrepancy_regions,
        "summary": (
            "Document matches trusted reference with zero significant discrepancies."
            if is_identical else
            f"Reference comparison detected {len(discrepancy_regions)} modified region(s) covering {diff_pct}% of document area."
        )
    }
