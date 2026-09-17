"""
DocForensics AI — Document Type Classifier
Conservatively classifies documents based on OCR keyword distributions and spatial patterns.
"""

from typing import List, Dict, Any, Tuple


ACADEMIC_KEYWORDS = {
    "sgpa", "cgpa", "grade", "credits", "semester", "marksheet", "mark sheet",
    "grade card", "transcript", "roll no", "enrollment", "course code", "subject code",
    "internal", "external", "theory", "practical", "examination", "result", "pass", "fail",
    "curriculum", "university", "institute", "college", "gpa", "percentage"
}

INVOICE_KEYWORDS = {
    "invoice", "bill to", "ship to", "subtotal", "sub-total", "tax", "vat", "gst",
    "total amount", "amount due", "balance due", "unit price", "qty", "quantity",
    "rate", "item description", "line total", "payment terms", "due date", "bill of sale"
}

CERTIFICATE_KEYWORDS = {
    "certificate", "certify", "certified", "hereby", "presented to", "conferred upon",
    "successfully completed", "completion", "achievement", "appreciation", "participation"
}


def detect_document_type(ocr_results: List[Dict[str, Any]]) -> Tuple[str, float]:
    """
    Classifies document type conservatively from OCR extracted tokens.
    
    Returns:
        document_type: 'academic_result' | 'invoice' | 'certificate' | 'generic_document' | 'unknown'
        confidence: Confidence score [0.0 - 1.0]
    """
    if not ocr_results:
        return "unknown", 0.0

    full_text_lower = " ".join([r.get("text", "").lower() for r in ocr_results])
    
    academic_score = 0
    for kw in ACADEMIC_KEYWORDS:
        if kw in full_text_lower:
            # High-signal keywords weigh more
            if kw in {"sgpa", "cgpa", "grade card", "marksheet", "mark sheet", "transcript", "semester"}:
                academic_score += 3
            else:
                academic_score += 1

    invoice_score = 0
    for kw in INVOICE_KEYWORDS:
        if kw in full_text_lower:
            if kw in {"invoice", "bill to", "subtotal", "tax", "total amount", "unit price"}:
                invoice_score += 3
            else:
                invoice_score += 1

    cert_score = 0
    for kw in CERTIFICATE_KEYWORDS:
        if kw in full_text_lower:
            if kw in {"certificate", "hereby", "presented to", "successfully completed"}:
                cert_score += 3
            else:
                cert_score += 1

    # Conservative thresholding
    if academic_score >= 4 and academic_score > invoice_score:
        conf = min(1.0, 0.4 + academic_score * 0.08)
        return "academic_result", round(conf, 2)

    if invoice_score >= 4 and invoice_score > academic_score:
        conf = min(1.0, 0.4 + invoice_score * 0.08)
        return "invoice", round(conf, 2)

    if cert_score >= 3:
        conf = min(1.0, 0.4 + cert_score * 0.1)
        return "certificate", round(conf, 2)

    if len(ocr_results) > 5:
        return "generic_document", 0.5

    return "unknown", 0.0
