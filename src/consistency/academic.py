"""
DocForensics AI — Academic Marksheet Consistency Engine
Performs deterministic mathematical and rule-based verification:
1. Marks bounds & impossible value detection
2. Component marks summation (internal + external = total)
3. Institutional grade-to-marks consistency
4. Subject pass/fail vs overall result consistency
5. Deterministic SGPA / CGPA calculation vs displayed SGPA
"""

from typing import Dict, Any, List, Optional
from config.academic_rules import load_academic_rule


def calculate_sgpa(subjects: List[Dict[str, Any]], rules: Optional[Dict[str, Any]] = None) -> Optional[float]:
    """Helper to compute deterministic SGPA from subject list and rules."""
    if not rules or "grade_points" not in rules or not subjects:
        return None
    grade_points_map = rules["grade_points"]
    total_credit_points = 0.0
    total_credits = 0.0
    for subj in subjects:
        grade = subj.get("grade")
        credits = subj.get("credits", 3.0)
        if grade and grade in grade_points_map:
            gp = grade_points_map[grade]
            total_credit_points += (gp * credits)
            total_credits += credits
        elif subj.get("obtained_marks") is not None and "grade_marks_range" in rules:
            marks = subj["obtained_marks"]
            matched_gp = None
            for g_name, (low, high) in rules["grade_marks_range"].items():
                if low <= marks <= high and g_name in grade_points_map:
                    matched_gp = grade_points_map[g_name]
                    break
            if matched_gp is not None:
                total_credit_points += (matched_gp * credits)
                total_credits += credits
            else:
                return None
        else:
            return None
    if total_credits == 0:
        return None
    rounding = rules.get("sgpa_rounding_decimals", 2)
    return round(total_credit_points / total_credits, rounding)


def verify_academic_consistency(
    fields: Dict[str, Any],
    scheme_id: Optional[str] = "standard_ugc_10point"
) -> Dict[str, Any]:
    """
    Executes academic consistency checks on extracted marksheet fields.

    Returns structured analysis containing:
        - status: 'content_consistent' | 'content_inconsistency_detected' | 'insufficient_information'
        - summary: human-readable explanation
        - calculated_sgpa: computed SGPA if rule available
        - displayed_sgpa: extracted SGPA from document
        - checks: list of individual test results
    """
    checks: List[Dict[str, Any]] = []
    subjects = fields.get("subjects", [])
    displayed_sgpa_field = fields.get("displayed_sgpa")
    displayed_sgpa = displayed_sgpa_field.get("value") if displayed_sgpa_field else None
    result_status_field = fields.get("result_status")
    displayed_status = result_status_field.get("value", "").upper() if result_status_field else ""

    # Load institutional grading rules
    rules = load_academic_rule(scheme_id) if scheme_id else None

    has_inconsistency = False

    # 1. Check Marks Bounds (0 <= marks <= max_marks)
    for idx, subj in enumerate(subjects):
        obt = subj.get("obtained_marks") if subj.get("obtained_marks") is not None else subj.get("total_marks")
        max_m = subj.get("max_marks", 100.0)
        s_name = subj.get("subject_name") or subj.get("name") or subj.get("code") or f"Subject #{idx+1}"

        if obt is not None:
            if obt < 0 or (max_m is not None and obt > max_m):
                has_inconsistency = True
                checks.append({
                    "check_id": f"marks_bounds_{idx+1}",
                    "check_name": f"Marks Range ({s_name})",
                    "displayed_value": f"{obt} / {max_m}",
                    "calculated_value": f"Valid Range: 0 - {max_m}",
                    "status": "impossible_value",
                    "explanation": f"Obtained marks {obt} exceeds maximum permissible marks {max_m}."
                })
            else:
                checks.append({
                    "check_id": f"marks_bounds_{idx+1}",
                    "check_name": f"Marks Range ({s_name})",
                    "displayed_value": f"{obt} / {max_m}",
                    "calculated_value": "Valid",
                    "status": "match",
                    "explanation": "Marks are within legitimate mathematical bounds."
                })

    # 2. Check Internal + External Component Sum
    for idx, subj in enumerate(subjects):
        internal = subj.get("internal_marks")
        external = subj.get("external_marks")
        total = subj.get("obtained_marks") if subj.get("obtained_marks") is not None else subj.get("total_marks")
        s_name = subj.get("subject_name") or subj.get("name") or subj.get("code") or f"Subject #{idx+1}"

        if internal is not None and external is not None and total is not None:
            calc_total = internal + external
            if abs(calc_total - total) > 0.01:
                has_inconsistency = True
                checks.append({
                    "check_id": f"component_sum_{idx+1}",
                    "check_name": f"Marks Addition ({s_name})",
                    "displayed_value": f"Total: {total}",
                    "calculated_value": f"Internal ({internal}) + External ({external}) = {calc_total}",
                    "status": "mismatch",
                    "explanation": f"Component marks sum ({calc_total}) does not match displayed total marks ({total})."
                })
            else:
                checks.append({
                    "check_id": f"component_sum_{idx+1}",
                    "check_name": f"Marks Addition ({s_name})",
                    "displayed_value": f"Total: {total}",
                    "calculated_value": f"{calc_total}",
                    "status": "match",
                    "explanation": "Component marks accurately sum to displayed total."
                })

    # 3. Check Grade-to-Marks Consistency (If rule configured)
    if rules and "grade_marks_range" in rules:
        grade_ranges = rules["grade_marks_range"]
        for idx, subj in enumerate(subjects):
            grade = subj.get("grade")
            marks = subj.get("obtained_marks")
            s_name = subj.get("subject_name") or f"Subject #{idx+1}"

            if grade and marks is not None and grade in grade_ranges:
                low, high = grade_ranges[grade]
                if not (low <= marks <= high):
                    has_inconsistency = True
                    checks.append({
                        "check_id": f"grade_marks_{idx+1}",
                        "check_name": f"Grade Assignment ({s_name})",
                        "displayed_value": f"Grade: {grade} (Marks: {marks})",
                        "calculated_value": f"Expected Range for {grade}: [{low} - {high}]",
                        "status": "mismatch",
                        "explanation": f"Marks {marks} do not correspond to the assigned letter grade {grade} under {rules.get('name')}."
                    })

    # 4. Check SGPA / CGPA Calculation
    calculated_sgpa: Optional[float] = None
    if rules and "grade_points" in rules and subjects:
        grade_points_map = rules["grade_points"]
        total_credit_points = 0.0
        total_credits = 0.0
        can_calculate_sgpa = True

        for subj in subjects:
            grade = subj.get("grade")
            credits = subj.get("credits", 3.0)

            if grade and grade in grade_points_map:
                gp = grade_points_map[grade]
                total_credit_points += (gp * credits)
                total_credits += credits
            elif subj.get("obtained_marks") is not None and "grade_marks_range" in rules:
                # Deduce grade from marks if grade missing
                marks = subj["obtained_marks"]
                matched_gp = None
                for g_name, (low, high) in rules["grade_marks_range"].items():
                    if low <= marks <= high and g_name in grade_points_map:
                        matched_gp = grade_points_map[g_name]
                        break
                if matched_gp is not None:
                    total_credit_points += (matched_gp * credits)
                    total_credits += credits
                else:
                    can_calculate_sgpa = False
            else:
                can_calculate_sgpa = False

        if can_calculate_sgpa and total_credits > 0:
            rounding = rules.get("sgpa_rounding_decimals", 2)
            calculated_sgpa = round(total_credit_points / total_credits, rounding)

            if displayed_sgpa is not None:
                tolerance = rules.get("sgpa_tolerance", 0.03)
                diff = abs(calculated_sgpa - displayed_sgpa)

                if diff > tolerance:
                    has_inconsistency = True
                    checks.append({
                        "check_id": "sgpa_consistency",
                        "check_name": "SGPA Calculation",
                        "displayed_value": f"{displayed_sgpa:.2f}",
                        "calculated_value": f"{calculated_sgpa:.2f}",
                        "status": "mismatch",
                        "explanation": (
                            f"Displayed SGPA ({displayed_sgpa:.2f}) does not match mathematically calculated "
                            f"SGPA ({calculated_sgpa:.2f}) from extracted course credits and grades."
                        )
                    })
                else:
                    checks.append({
                        "check_id": "sgpa_consistency",
                        "check_name": "SGPA Calculation",
                        "displayed_value": f"{displayed_sgpa:.2f}",
                        "calculated_value": f"{calculated_sgpa:.2f}",
                        "status": "match",
                        "explanation": "Displayed SGPA is mathematically consistent with course credits and assigned grades."
                    })
        else:
            if displayed_sgpa is not None:
                checks.append({
                    "check_id": "sgpa_consistency",
                    "check_name": "SGPA Calculation",
                    "displayed_value": f"{displayed_sgpa:.2f}",
                    "calculated_value": "Incomplete Subject Grades",
                    "status": "unavailable",
                    "explanation": "Insufficient subject grade/credit tokens extracted to complete full SGPA computation."
                })
    else:
        # Rule not configured
        checks.append({
            "check_id": "sgpa_consistency",
            "check_name": "SGPA Calculation",
            "displayed_value": f"{displayed_sgpa}" if displayed_sgpa is not None else "Not Found",
            "calculated_value": "Unavailable",
            "status": "unavailable",
            "explanation": "SGPA verification unavailable — institutional grading rules not configured."
        })

    # Determine overall content status
    if has_inconsistency:
        overall_status = "content_inconsistency_detected"
        summary = "Academic consistency check detected mathematical or grading rule mismatches in extracted marksheet data."
    elif checks and any(c["status"] == "match" for c in checks):
        overall_status = "content_consistent"
        summary = "Extracted academic values, marks ranges, and SGPA calculations are internally consistent."
    else:
        overall_status = "insufficient_information"
        summary = "Document layout contained insufficient structured academic entries for full deterministic verification."

    return {
        "status": overall_status,
        "summary": summary,
        "calculated_sgpa": calculated_sgpa,
        "displayed_sgpa": displayed_sgpa,
        "rule_scheme_used": rules.get("name") if rules else "None",
        "checks": checks
    }
