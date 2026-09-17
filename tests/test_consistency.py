"""
DocForensics AI — Content Consistency & Inspect-Element Analysis Test Suite (Phase 10)
Validates:
- Academic marksheet SGPA calculation and inspect-element alteration detection
- Component marks addition verification
- Out-of-bounds / impossible marks detection
- Unknown grading scheme conservative fallback
- Invoice line-item and grand total arithmetic checks
- Classification heuristics and modular verification provider
"""

import pytest
from src.consistency.classifier import detect_document_type
from src.consistency.extractor import extract_structured_fields
from src.consistency.academic import verify_academic_consistency, calculate_sgpa
from src.consistency.invoice import verify_invoice_consistency
from src.consistency.pipeline import ContentConsistencyPipeline
from src.consistency.verification_provider import MockVerificationProvider


def test_academic_classification():
    """Verify classifier detects academic marksheet keywords."""
    ocr_results = [
        {"text": "State University of Technology", "bbox": [100, 50, 400, 80], "confidence": 0.95},
        {"text": "Statement of Marks / Grade Sheet", "bbox": [120, 90, 380, 110], "confidence": 0.92},
        {"text": "Roll No: 202104592", "bbox": [50, 130, 200, 150], "confidence": 0.90},
        {"text": "Semester: IV", "bbox": [250, 130, 350, 150], "confidence": 0.88},
        {"text": "Course Code Credits Grade Grade Point", "bbox": [50, 180, 500, 200], "confidence": 0.94},
        {"text": "CS401 Computer Networks 4 A 8", "bbox": [50, 210, 450, 230], "confidence": 0.91},
        {"text": "CS402 Operating Systems 4 A+ 9", "bbox": [50, 240, 450, 260], "confidence": 0.90},
        {"text": "SGPA: 8.50", "bbox": [300, 300, 400, 320], "confidence": 0.96},
    ]

    doc_type, confidence = detect_document_type(ocr_results)
    assert doc_type == "academic_result"
    assert confidence > 0.6


def test_academic_sgpa_consistent():
    """Verify that a genuine marksheet with matching SGPA passes without discrepancy."""
    # CS401: 4 credits * 8 (A) = 32
    # CS402: 4 credits * 9 (A+) = 36
    # Total points = 68, Total credits = 8 -> SGPA = 68 / 8 = 8.50
    fields = {
        "displayed_sgpa": {"value": 8.50, "raw_text": "SGPA: 8.50", "confidence": 0.95},
        "subjects": [
            {"code": "CS401", "name": "Computer Networks", "credits": 4.0, "grade": "A", "grade_point": 8.0},
            {"code": "CS402", "name": "Operating Systems", "credits": 4.0, "grade": "A+", "grade_point": 9.0},
        ]
    }

    res = verify_academic_consistency(fields, scheme_id="standard_ugc_10point")
    assert res["status"] == "content_consistent"
    sgpa_check = next((c for c in res["checks"] if c["check_id"] == "sgpa_consistency"), None)
    assert sgpa_check is not None
    assert sgpa_check["status"] == "match"
    assert float(sgpa_check["calculated_value"]) == 8.50


def test_inspect_element_sgpa_alteration():
    """
    CRITICAL TEST: Simulates an Inspect-Element forgery where the user edited
    the displayed SGPA on the webpage from 6.80 to 9.20 before taking a clean screenshot.
    """
    # CS101: 3 credits * 7 (B+) = 21
    # CS102: 3 credits * 6 (B)  = 18
    # CS103: 4 credits * 7 (B+) = 28
    # Total points = 67, Total credits = 10 -> Actual SGPA = 6.70
    # Altered displayed SGPA: 9.20
    fields = {
        "displayed_sgpa": {"value": 9.20, "raw_text": "SGPA: 9.20", "confidence": 0.98},
        "subjects": [
            {"code": "CS101", "name": "Programming in C", "credits": 3.0, "grade": "B+", "grade_point": 7.0},
            {"code": "CS102", "name": "Digital Logic", "credits": 3.0, "grade": "B", "grade_point": 6.0},
            {"code": "CS103", "name": "Discrete Mathematics", "credits": 4.0, "grade": "B+", "grade_point": 7.0},
        ]
    }

    res = verify_academic_consistency(fields, scheme_id="standard_ugc_10point")
    assert res["status"] == "content_inconsistency_detected"
    sgpa_check = next((c for c in res["checks"] if c["check_id"] == "sgpa_consistency"), None)
    assert sgpa_check is not None
    assert sgpa_check["status"] == "mismatch"
    assert "9.20" in sgpa_check["displayed_value"]
    assert "6.70" in sgpa_check["calculated_value"]


def test_component_marks_addition_mismatch():
    """Verify internal + external = total marks check catches forged total."""
    # Internal: 20, External: 40, Total entered: 85 (tampered / forged)
    fields = {
        "subjects": [
            {
                "code": "PH101",
                "internal_marks": 20.0,
                "external_marks": 40.0,
                "total_marks": 85.0,  # 20 + 40 != 85
                "max_marks": 100.0,
            }
        ]
    }

    res = verify_academic_consistency(fields, scheme_id="standard_ugc_10point")
    assert res["status"] == "content_inconsistency_detected"
    add_check = next((c for c in res["checks"] if "component_sum" in c["check_id"]), None)
    assert add_check is not None
    assert add_check["status"] == "mismatch"


def test_impossible_out_of_bounds_marks():
    """Verify check flags impossible marks (e.g. 105 out of 100)."""
    fields = {
        "subjects": [
            {
                "code": "MA201",
                "total_marks": 105.0,
                "max_marks": 100.0,
            }
        ]
    }

    res = verify_academic_consistency(fields, scheme_id="standard_ugc_10point")
    assert res["status"] == "content_inconsistency_detected"
    bounds_check = next((c for c in res["checks"] if "marks_bounds" in c["check_id"]), None)
    assert bounds_check is not None
    assert bounds_check["status"] == "impossible_value"


def test_unknown_grading_scheme_conservative_fallback():
    """Verify non-configured schemes do not throw false positives."""
    fields = {
        "displayed_sgpa": {"value": 3.8, "raw_text": "GPA: 3.8", "confidence": 0.90},
        "subjects": [
            {"code": "CS101", "credits": 3.0, "grade": "ALPHA"}  # Non-standard grade
        ]
    }

    res = verify_academic_consistency(fields, scheme_id="non_existent_custom_scheme")
    # Falls back gracefully without raising an exception
    assert res["status"] in ["insufficient_information", "content_consistent"]


def test_invoice_arithmetic():
    """Verify invoice line item calculation and grand total verification."""
    # Item 1: 2 * 50 = 100
    # Item 2: 3 * 20 = 60
    # Subtotal = 160, Tax = 16, Total = 176
    valid_invoice_fields = {
        "items": [
            {"quantity": 2.0, "unit_price": 50.0, "line_total": 100.0},
            {"quantity": 3.0, "unit_price": 20.0, "line_total": 60.0},
        ],
        "subtotal": {"value": 160.0},
        "tax": {"value": 16.0},
        "total": {"value": 176.0},
    }

    res = verify_invoice_consistency(valid_invoice_fields)
    assert res["status"] == "content_consistent"

    # Forged Total Invoice (Subtotal 160 + Tax 16 != Total 120)
    forged_invoice_fields = {
        "items": [
            {"quantity": 2.0, "unit_price": 50.0, "line_total": 100.0},
            {"quantity": 3.0, "unit_price": 20.0, "line_total": 60.0},
        ],
        "subtotal": {"value": 160.0},
        "tax": {"value": 16.0},
        "total": {"value": 120.0},  # Altered!
    }

    res_forged = verify_invoice_consistency(forged_invoice_fields)
    assert res_forged["status"] == "content_inconsistency_detected"


def test_content_consistency_pipeline_end_to_end():
    """Tests complete ContentConsistencyPipeline execution on raw OCR detections."""
    pipeline = ContentConsistencyPipeline(verification_provider=MockVerificationProvider())

    ocr_entries = [
        {"text": "State Technical University Marksheet", "bbox": [50, 20, 400, 50], "confidence": 0.95},
        {"text": "Roll Number: 2021BCS012", "bbox": [50, 60, 250, 80], "confidence": 0.92},
        {"text": "Subject Credits Grade GradePoint", "bbox": [50, 100, 450, 120], "confidence": 0.90},
        {"text": "CS301 Data Structures 4 A+ 9", "bbox": [50, 130, 450, 150], "confidence": 0.93},
        {"text": "CS302 Algorithms 4 A 8", "bbox": [50, 160, 450, 180], "confidence": 0.91},
        {"text": "SGPA: 8.50", "bbox": [300, 220, 420, 240], "confidence": 0.96},
    ]

    report = pipeline.analyze_content(ocr_entries)
    assert report["document_type"] == "academic_result"
    assert report["status"] == "content_consistent"
    assert "record_verification" in report
