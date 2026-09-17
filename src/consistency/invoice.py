"""
DocForensics AI — Invoice Consistency Engine
Validates deterministic financial arithmetic across invoice totals, tax, and line items.
"""

from typing import Dict, Any, List


def _extract_val(field_obj: Any) -> Any:
    """Helper to extract float value from dict or float/int."""
    if isinstance(field_obj, dict):
        return field_obj.get("value")
    elif isinstance(field_obj, (int, float)):
        return float(field_obj)
    return None


def verify_invoice_consistency(fields: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes financial arithmetic consistency checks on extracted invoice fields.
    """
    checks: List[Dict[str, Any]] = []
    has_inconsistency = False

    subtotal = _extract_val(fields.get("displayed_subtotal") or fields.get("subtotal"))
    tax = _extract_val(fields.get("displayed_tax") or fields.get("tax"))
    grand_total = _extract_val(fields.get("displayed_grand_total") or fields.get("grand_total") or fields.get("total"))

    # Check 1: Line items quantity * unit_price = line_total
    items = fields.get("items", [])
    for idx, item in enumerate(items):
        qty = item.get("quantity")
        unit_p = item.get("unit_price")
        line_t = item.get("line_total")

        if qty is not None and unit_p is not None and line_t is not None:
            calc_line = round(qty * unit_p, 2)
            if abs(calc_line - line_t) > 0.05:
                has_inconsistency = True
                checks.append({
                    "check_id": f"item_arithmetic_{idx+1}",
                    "check_name": f"Item #{idx+1} Math",
                    "displayed_value": f"${line_t:.2f}",
                    "calculated_value": f"{qty} x ${unit_p:.2f} = ${calc_line:.2f}",
                    "status": "mismatch",
                    "explanation": f"Line total ${line_t:.2f} does not equal quantity times unit price (${calc_line:.2f})."
                })
            else:
                checks.append({
                    "check_id": f"item_arithmetic_{idx+1}",
                    "check_name": f"Item #{idx+1} Math",
                    "displayed_value": f"${line_t:.2f}",
                    "calculated_value": f"${calc_line:.2f}",
                    "status": "match",
                    "explanation": "Item quantity multiplied by unit price matches line total."
                })

    # Check 2: Subtotal + Tax = Grand Total
    if subtotal is not None and tax is not None and grand_total is not None:
        expected_total = round(subtotal + tax, 2)
        diff = abs(expected_total - grand_total)

        if diff > 0.05:  # Tolerance for minor cents rounding
            has_inconsistency = True
            checks.append({
                "check_id": "invoice_total_arithmetic",
                "check_name": "Invoice Total Sum",
                "displayed_value": f"${grand_total:.2f}",
                "calculated_value": f"Subtotal (${subtotal:.2f}) + Tax (${tax:.2f}) = ${expected_total:.2f}",
                "status": "mismatch",
                "explanation": f"Displayed grand total (${grand_total:.2f}) does not equal sum of subtotal and tax (${expected_total:.2f})."
            })
        else:
            checks.append({
                "check_id": "invoice_total_arithmetic",
                "check_name": "Invoice Total Sum",
                "displayed_value": f"${grand_total:.2f}",
                "calculated_value": f"${expected_total:.2f}",
                "status": "match",
                "explanation": "Subtotal, tax, and grand total are mathematically consistent."
            })
    elif grand_total is not None and subtotal is not None:
        # Check subtotal <= grand_total
        if subtotal > grand_total:
            has_inconsistency = True
            checks.append({
                "check_id": "invoice_subtotal_bounds",
                "check_name": "Subtotal vs Total",
                "displayed_value": f"Subtotal: ${subtotal:.2f}, Total: ${grand_total:.2f}",
                "calculated_value": "Subtotal <= Grand Total",
                "status": "mismatch",
                "explanation": f"Subtotal (${subtotal:.2f}) exceeds grand total (${grand_total:.2f})."
            })

    # Determine overall status
    if has_inconsistency:
        status = "content_inconsistency_detected"
        summary = "Financial arithmetic checks detected numerical discrepancies in invoice totals."
    elif checks and any(c["status"] == "match" for c in checks):
        status = "content_consistent"
        summary = "Extracted invoice financial totals and arithmetic relationships are consistent."
    else:
        status = "insufficient_information"
        summary = "Document did not contain sufficient discrete numerical fields for complete invoice arithmetic verification."

    return {
        "status": status,
        "summary": summary,
        "checks": checks
    }

