"""
DocForensics AI — Structured Field Extraction Engine
Extracts structured document entities, table rows, academic subjects, marks, grades,
and financial totals from OCR bounding boxes and spatial layout alignment.
"""

import re
from typing import List, Dict, Any, Optional, Tuple


def _find_numeric_near_label(
    label_patterns: List[str],
    ocr_results: List[Dict[str, Any]],
    val_regex: str = r"[-+]?\d*\.?\d+"
) -> Optional[Dict[str, Any]]:
    """Finds a numeric value adjacent to or following a specific label pattern."""
    for i, item in enumerate(ocr_results):
        text = item.get("text", "").strip()
        for pat in label_patterns:
            # Check if label and value are in the same OCR box (e.g. "SGPA: 8.21" or "SGPA = 8.21")
            match = re.search(rf"{pat}[:=\s]*({val_regex})", text, re.IGNORECASE)
            if match:
                try:
                    val = float(match.group(1))
                    return {
                        "value": val,
                        "raw_text": text,
                        "confidence": item.get("confidence", 0.0),
                        "bbox": item.get("bbox", [])
                    }
                except ValueError:
                    pass

            # Check if next OCR token is the value
            if re.fullmatch(pat, text, re.IGNORECASE) or re.search(rf"\b{pat}\b[:=\s]*$", text, re.IGNORECASE):
                # Look at immediate next items
                for next_item in ocr_results[i+1:i+4]:
                    next_text = next_item.get("text", "").strip()
                    num_match = re.search(rf"^({val_regex})$", next_text)
                    if num_match:
                        try:
                            val = float(num_match.group(1))
                            return {
                                "value": val,
                                "raw_text": next_text,
                                "confidence": next_item.get("confidence", 0.0),
                                "bbox": next_item.get("bbox", [])
                            }
                        except ValueError:
                            pass
    return None


def _find_text_near_label(
    label_patterns: List[str],
    ocr_results: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Finds string text adjacent to or following a label pattern."""
    for i, item in enumerate(ocr_results):
        text = item.get("text", "").strip()
        for pat in label_patterns:
            match = re.search(rf"{pat}[:=\s]+(.+)", text, re.IGNORECASE)
            if match:
                val = match.group(1).strip()
                if val:
                    return {
                        "value": val,
                        "raw_text": text,
                        "confidence": item.get("confidence", 0.0),
                        "bbox": item.get("bbox", [])
                    }

            if re.fullmatch(pat, text, re.IGNORECASE) or re.search(rf"\b{pat}\b[:=\s]*$", text, re.IGNORECASE):
                if i + 1 < len(ocr_results):
                    next_item = ocr_results[i+1]
                    val = next_item.get("text", "").strip()
                    if val and not any(re.search(rf"\b{p}\b", val, re.IGNORECASE) for p in label_patterns):
                        return {
                            "value": val,
                            "raw_text": val,
                            "confidence": next_item.get("confidence", 0.0),
                            "bbox": next_item.get("bbox", [])
                        }
    return None


def extract_academic_fields(ocr_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extracts structured academic marksheet fields (Student info, subjects, grades, credits, SGPA).
    """
    fields: Dict[str, Any] = {
        "student_name": _find_text_near_label(["name", "student name", "candidate's name"], ocr_results),
        "roll_number": _find_text_near_label(["roll no", "roll number", "enrollment no", "reg no", "registration no", "student id"], ocr_results),
        "course_semester": _find_text_near_label(["semester", "sem", "programme", "course", "branch"], ocr_results),
        "result_status": _find_text_near_label(["result", "status", "final result", "result status"], ocr_results),
        "result_date": _find_text_near_label(["date", "date of issue", "result date", "declared on"], ocr_results),
        "displayed_sgpa": _find_numeric_near_label(["sgpa", "s.g.p.a", "semester gpa", "gpa"], ocr_results),
        "displayed_cgpa": _find_numeric_near_label(["cgpa", "c.g.p.a", "cumulative gpa"], ocr_results),
        "subjects": []
    }

    # Extract subject rows using line clustering and keyword patterns
    # Look for table entries with course codes, credits, grades, marks
    subjects = []
    
    # Sort OCR tokens vertically then horizontally
    sorted_items = sorted(
        ocr_results,
        key=lambda x: (x.get("bbox", [0, 0, 0, 0])[1] // 15, x.get("bbox", [0, 0, 0, 0])[0])
    )

    # Group tokens into visual text lines (within ~18px y-distance)
    lines: List[List[Dict[str, Any]]] = []
    current_line: List[Dict[str, Any]] = []
    current_y = -100

    for item in sorted_items:
        bbox = item.get("bbox", [0, 0, 0, 0])
        y = bbox[1] if len(bbox) >= 2 else 0
        if abs(y - current_y) > 18:
            if current_line:
                lines.append(sorted(current_line, key=lambda x: x.get("bbox", [0, 0, 0, 0])[0]))
            current_line = [item]
            current_y = y
        else:
            current_line.append(item)
            
    if current_line:
        lines.append(sorted(current_line, key=lambda x: x.get("bbox", [0, 0, 0, 0])[0]))

    # Scan lines for subject rows
    # A subject row typically has: Course Code/Name + Credits (1-6) + Grade (O, A+, A, B, etc.) OR Marks
    grade_tokens = {"O", "A+", "A", "B+", "B", "C", "P", "F", "AB", "PASS", "FAIL"}

    for line in lines:
        line_texts = [item.get("text", "").strip() for item in line]
        joined_line = " ".join(line_texts)

        # Skip headers / metadata
        if any(h in joined_line.lower() for h in ["subject code", "course code", "grade point", "credits", "roll no", "university"]):
            continue

        # Look for tokens in line (handling single or multi-word OCR boxes)
        found_grade = None
        found_grade_point = None
        found_credits = None
        found_marks: List[float] = []
        subject_name_tokens = []
        course_code = None

        # Flatten all words from the OCR boxes in this line
        line_words = []
        for item in line:
            words = item.get("text", "").strip().split()
            line_words.extend(words)

        for w in line_words:
            # Course code pattern (e.g. CS101, CSE-302, KCS-501, 18CS42)
            if re.match(r"^[A-Z]{2,4}[-_\s]?\d{3,4}[A-Z]?$", w, re.IGNORECASE) and course_code is None:
                course_code = w
            elif w.upper() in grade_tokens and found_grade is None:
                found_grade = w.upper()
            elif re.match(r"^\d+\.?\d*$", w):
                val = float(w)
                if found_credits is None and val.is_integer() and 1 <= val <= 8 and found_grade is None:
                    # Credit value preceding grade
                    found_credits = val
                elif found_grade is not None and found_grade_point is None and 0 <= val <= 10:
                    # Grade point following letter grade
                    found_grade_point = val
                elif found_credits is None and val.is_integer() and 1 <= val <= 8:
                    found_credits = val
                else:
                    found_marks.append(val)
            else:
                if len(w) > 1 and not w.isdigit():
                    subject_name_tokens.append(w)

        # If line contains valid subject information (e.g. credits + grade OR code + marks)
        if (found_credits is not None and (found_grade is not None or found_marks)) or (course_code and (found_grade or found_marks)):
            subj_name = " ".join(subject_name_tokens) if subject_name_tokens else (course_code or "Subject")
            
            # Estimate internal / external if multiple marks
            internal = found_marks[0] if len(found_marks) >= 2 else None
            external = found_marks[1] if len(found_marks) >= 2 else None
            total_marks = found_marks[-1] if found_marks else None

            subjects.append({
                "subject_code": course_code,
                "subject_name": subj_name,
                "credits": float(found_credits) if found_credits is not None else 3.0,
                "grade": found_grade,
                "grade_point": found_grade_point,
                "internal_marks": internal,
                "external_marks": external,
                "obtained_marks": total_marks,
                "max_marks": 100.0,
                "confidence": min([item.get("confidence", 0.8) for item in line]) if line else 0.8,
                "bbox": [line[0].get("bbox", [0, 0, 0, 0])[0], line[0].get("bbox", [0, 0, 0, 0])[1],
                         line[-1].get("bbox", [0, 0, 0, 0])[2], line[-1].get("bbox", [0, 0, 0, 0])[3]]
            })

    fields["subjects"] = subjects
    return fields


def extract_invoice_fields(ocr_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extracts structured invoice fields (Amounts, subtotals, tax, line items).
    """
    fields: Dict[str, Any] = {
        "invoice_number": _find_text_near_label(["invoice #", "invoice no", "inv no", "bill no"], ocr_results),
        "invoice_date": _find_text_near_label(["invoice date", "date", "bill date"], ocr_results),
        "displayed_subtotal": _find_numeric_near_label(["subtotal", "sub total", "sub-total", "net amount"], ocr_results),
        "displayed_tax": _find_numeric_near_label(["tax", "vat", "gst", "sales tax"], ocr_results),
        "displayed_grand_total": _find_numeric_near_label(["total", "grand total", "total amount", "amount due", "balance due", "total billed"], ocr_results),
        "line_items": []
    }
    return fields


def extract_structured_fields(
    doc_type: str,
    ocr_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Dispatches extraction based on detected document type.
    """
    if doc_type == "academic_result":
        return extract_academic_fields(ocr_results)
    elif doc_type == "invoice":
        return extract_invoice_fields(ocr_results)
    else:
        return {
            "generic_text_count": len(ocr_results),
            "summary_date": _find_text_near_label(["date"], ocr_results),
            "summary_total": _find_numeric_near_label(["total", "amount"], ocr_results)
        }
