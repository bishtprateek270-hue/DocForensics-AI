"""
DocForensics AI — Document Collection & Synthetic Base Generator
Generates realistic multi-family clean documents (Invoices, Receipts, ID Cards,
Certificates, Tax Forms, Legal Agreements) with authentic layouts, typography,
stamps, and structured fields, and saves them to data/raw/authentic/.
"""

import os
import random
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import cv2

from config import SYNTHETIC_ORIGINALS_DIR, cfg


AUTHENTIC_RAW_DIR = SYNTHETIC_ORIGINALS_DIR
AUTHENTIC_RAW_DIR.mkdir(parents=True, exist_ok=True)


def get_font(size: int = 20, bold: bool = False):
    """Attempt to load a clean system font or fallback to default."""
    font_names = [
        "arialbd.ttf" if bold else "arial.ttf",
        "calibrib.ttf" if bold else "calibri.ttf",
        "segoeuib.ttf" if bold else "segoeui.ttf",
        "timesbd.ttf" if bold else "times.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for fn in font_names:
        try:
            return ImageFont.truetype(fn, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def add_document_texture(img: Image.Image, noise_level: float = 0.03) -> Image.Image:
    """Add subtle paper texture and slight scan grain."""
    arr = np.array(img, dtype=np.float32)
    h, w, c = arr.shape
    noise = np.random.normal(0, noise_level * 255, (h, w, c))
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def create_invoice_template(doc_id: str) -> Image.Image:
    """Generate a clean commercial invoice."""
    w, h = 800, 1050
    img = Image.new("RGB", (w, h), color=(252, 252, 250))
    draw = ImageDraw.Draw(img)

    f_title = get_font(32, bold=True)
    f_sub = get_font(18, bold=True)
    f_body = get_font(16, bold=False)
    f_small = get_font(13, bold=False)

    # Header Bar
    draw.rectangle([40, 40, w - 40, 110], fill=(30, 41, 59))
    draw.text((60, 55), "GLOBAL CORP LOGISTICS INC.", fill=(255, 255, 255), font=f_title)
    draw.text((w - 240, 65), "TAX INVOICE", fill=(203, 213, 225), font=f_sub)

    # Metadata
    inv_num = f"INV-2026-{random.randint(10000, 99999)}"
    date_str = f"2026-0{random.randint(1, 9)}-{random.randint(10, 28)}"
    due_str = f"2026-0{random.randint(1, 9)}-{random.randint(10, 28)}"

    draw.text((50, 140), "BILLED TO:", fill=(71, 85, 105), font=f_sub)
    draw.text((50, 170), f"Acme Industrial Solutions Ltd.\n102 Silicon Parkway, Suite {random.randint(100, 999)}\nSan Francisco, CA 94105", fill=(15, 23, 42), font=f_body)

    draw.text((500, 140), f"Invoice No : {inv_num}", fill=(15, 23, 42), font=f_body)
    draw.text((500, 170), f"Issue Date : {date_str}", fill=(15, 23, 42), font=f_body)
    draw.text((500, 200), f"Due Date   : {due_str}", fill=(15, 23, 42), font=f_body)

    # Table Header
    draw.rectangle([40, 270, w - 40, 310], fill=(226, 232, 240))
    draw.text((50, 282), "ITEM DESCRIPTION", fill=(30, 41, 59), font=f_sub)
    draw.text((450, 282), "QTY", fill=(30, 41, 59), font=f_sub)
    draw.text((550, 282), "UNIT PRICE", fill=(30, 41, 59), font=f_sub)
    draw.text((680, 282), "TOTAL", fill=(30, 41, 59), font=f_sub)

    items = [
        ("Cloud Infrastructure Server Cluster", 4, 1250.00),
        ("Cybersecurity Vulnerability Audit", 1, 4500.00),
        ("Enterprise Storage Array Backup (10TB)", 2, 850.00),
        ("High-Speed Dedicated Fiber Uplink", 12, 320.00),
    ]

    y = 330
    subtotal = 0.0
    for desc, qty, unit in items:
        total = qty * unit
        subtotal += total
        draw.text((50, y), desc, fill=(15, 23, 42), font=f_body)
        draw.text((460, y), str(qty), fill=(15, 23, 42), font=f_body)
        draw.text((550, y), f"${unit:,.2f}", fill=(15, 23, 42), font=f_body)
        draw.text((680, y), f"${total:,.2f}", fill=(15, 23, 42), font=f_body)
        draw.line([40, y + 30, w - 40, y + 30], fill=(241, 245, 249), width=1)
        y += 45

    # Totals Box
    tax = subtotal * 0.08
    grand_total = subtotal + tax

    draw.rectangle([480, y + 20, w - 40, y + 160], fill=(248, 250, 252), outline=(203, 213, 225), width=1)
    draw.text((500, y + 35), "Subtotal:", fill=(71, 85, 105), font=f_body)
    draw.text((680, y + 35), f"${subtotal:,.2f}", fill=(15, 23, 42), font=f_body)

    draw.text((500, y + 70), "Tax (8.0%):", fill=(71, 85, 105), font=f_body)
    draw.text((680, y + 70), f"${tax:,.2f}", fill=(15, 23, 42), font=f_body)

    draw.line([490, y + 105, w - 50, y + 105], fill=(203, 213, 225), width=2)
    draw.text((500, y + 120), "TOTAL DUE:", fill=(15, 23, 42), font=f_sub)
    draw.text((660, y + 118), f"${grand_total:,.2f}", fill=(185, 28, 28), font=f_title)

    # Footer & Stamp Placeholder
    draw.text((50, h - 140), "Authorized Signature:", fill=(71, 85, 105), font=f_small)
    draw.line([50, h - 70, 260, h - 70], fill=(15, 23, 42), width=2)
    draw.text((60, h - 100), "David K. Sterling (CFO)", fill=(30, 58, 138), font=f_sub)

    # Official Stamp (Blue circle)
    draw.ellipse([w - 230, h - 190, w - 90, h - 50], outline=(30, 64, 175), width=4)
    draw.ellipse([w - 220, h - 180, w - 100, h - 60], outline=(30, 64, 175), width=1)
    draw.text((w - 200, h - 130), "VERIFIED\nOFFICIAL", fill=(30, 64, 175), font=f_small, align="center")

    return add_document_texture(img)


def create_id_card_template(doc_id: str) -> Image.Image:
    """Generate an official ID badge / identity credential."""
    w, h = 900, 600
    img = Image.new("RGB", (w, h), color=(248, 250, 252))
    draw = ImageDraw.Draw(img)

    f_title = get_font(28, bold=True)
    f_sub = get_font(18, bold=True)
    f_body = get_font(16, bold=False)
    f_small = get_font(12, bold=False)

    # ID Top Header Banner
    draw.rectangle([0, 0, w, 80], fill=(15, 76, 129))
    draw.text((50, 24), "DEPARTMENT OF IDENTIFICATION & LICENSING", fill=(255, 255, 255), font=f_title)

    # Photo Box
    draw.rectangle([50, 120, 260, 380], fill=(226, 232, 240), outline=(71, 85, 105), width=2)
    # Draw simple avatar shape
    draw.ellipse([105, 160, 205, 260], fill=(148, 163, 184))
    draw.chord([80, 250, 230, 420], 180, 360, fill=(100, 116, 139))
    draw.text((100, 345), "[OFFICIAL PHOTO]", fill=(71, 85, 105), font=f_small)

    # Holder Information
    id_num = f"ID-{random.randint(10000000, 99999999)}"
    dob = f"19{random.randint(75, 99)}-{random.randint(10, 12)}-{random.randint(10, 28)}"
    expiry = f"203{random.randint(0, 5)}-06-30"

    draw.text((300, 120), "IDENTITY IDENTIFIER", fill=(100, 116, 139), font=f_small)
    draw.text((300, 140), id_num, fill=(185, 28, 28), font=f_title)

    draw.text((300, 200), "FULL NAME:", fill=(100, 116, 139), font=f_small)
    draw.text((300, 220), "ALEXANDER M. VANDERBILT", fill=(15, 23, 42), font=f_sub)

    draw.text((300, 270), "DATE OF BIRTH:", fill=(100, 116, 139), font=f_small)
    draw.text((300, 290), dob, fill=(15, 23, 42), font=f_body)

    draw.text((550, 270), "NATIONALITY:", fill=(100, 116, 139), font=f_small)
    draw.text((550, 290), "UNITED STATES", fill=(15, 23, 42), font=f_body)

    draw.text((300, 340), "EXPIRATION DATE:", fill=(100, 116, 139), font=f_small)
    draw.text((300, 360), expiry, fill=(15, 23, 42), font=f_body)

    draw.text((550, 340), "SECURITY CLEARANCE:", fill=(100, 116, 139), font=f_small)
    draw.text((550, 360), "LEVEL-IV (RESTRICTED)", fill=(21, 128, 61), font=f_sub)

    # Bottom Hologram & Security Bar
    draw.rectangle([0, 460, w, h], fill=(226, 232, 240))
    # Simulated MRZ barcode lines
    draw.text((50, 480), f"IDUSA{id_num}<<<<<<<<<<<<<<<<<<<<<<098234", fill=(30, 41, 59), font=f_body)
    draw.text((50, 520), f"8812044M3101017USA<<<<<<<<<<<<<<<<4", fill=(30, 41, 59), font=f_body)

    return add_document_texture(img)


def create_certificate_template(doc_id: str) -> Image.Image:
    """Generate an official academic / accreditation certificate."""
    w, h = 950, 700
    img = Image.new("RGB", (w, h), color=(255, 253, 245))
    draw = ImageDraw.Draw(img)

    f_title = get_font(36, bold=True)
    f_sub = get_font(22, bold=True)
    f_body = get_font(18, bold=False)
    f_script = get_font(26, bold=True)
    f_small = get_font(14, bold=False)

    # Ornate Border
    draw.rectangle([20, 20, w - 20, h - 20], outline=(180, 130, 30), width=6)
    draw.rectangle([30, 30, w - 30, h - 30], outline=(200, 160, 60), width=2)

    # Header
    draw.text((w // 2 - 250, 70), "NATIONAL INSTITUTE OF TECHNOLOGY", fill=(120, 53, 15), font=f_sub)
    draw.text((w // 2 - 220, 120), "CERTIFICATE OF ACHIEVEMENT", fill=(15, 23, 42), font=f_title)

    # Body
    draw.text((w // 2 - 130, 200), "This is to certify that", fill=(71, 85, 105), font=f_body)
    draw.text((w // 2 - 200, 250), "ELIZABETH CLAIRE MORGAN", fill=(30, 58, 138), font=f_script)

    cert_text = (
        "has successfully fulfilled all institutional requirements and demonstrated excellence in\n"
        "ADVANCED CYBERSECURITY & DIGITAL FORENSICS ENGINEERING\n"
        "awarded this 15th day of October, 2025 with Highest Distinction."
    )
    draw.text((w // 2 - 340, 320), cert_text, fill=(30, 41, 59), font=f_body, align="center")

    # Signatures and Gold Seal
    draw.line([100, 560, 320, 560], fill=(15, 23, 42), width=2)
    draw.text((120, 570), "Prof. Marcus Thorne, Ph.D.\nDean of Engineering", fill=(71, 85, 105), font=f_small, align="center")

    draw.line([w - 320, 560, w - 100, 560], fill=(15, 23, 42), width=2)
    draw.text((w - 300, 570), "Dr. Sarah Jenkins\nChancellor of the Board", fill=(71, 85, 105), font=f_small, align="center")

    # Gold Seal
    draw.ellipse([w // 2 - 60, 490, w // 2 + 60, 610], fill=(217, 119, 6), outline=(180, 83, 9), width=4)
    draw.ellipse([w // 2 - 50, 500, w // 2 + 50, 600], outline=(254, 240, 138), width=2)
    draw.text((w // 2 - 38, 540), "OFFICIAL\nSEAL", fill=(255, 255, 255), font=f_small, align="center")

    return add_document_texture(img)


def create_contract_template(doc_id: str) -> Image.Image:
    """Generate a legal agreement / contract."""
    w, h = 800, 1050
    img = Image.new("RGB", (w, h), color=(253, 253, 252))
    draw = ImageDraw.Draw(img)

    f_title = get_font(26, bold=True)
    f_sub = get_font(18, bold=True)
    f_body = get_font(14, bold=False)
    f_small = get_font(12, bold=False)

    # Header
    draw.text((w // 2 - 200, 50), "MUTUAL NON-DISCLOSURE AGREEMENT", fill=(15, 23, 42), font=f_title)
    draw.line([50, 90, w - 50, 90], fill=(15, 23, 42), width=2)

    date_str = f"March {random.randint(1, 28)}, 2026"
    draw.text((50, 110), f"Effective Date: {date_str} | Contract Ref: NDA-2026-{doc_id}", fill=(71, 85, 105), font=f_small)

    text_p1 = (
        "1. PARTIES: This Non-Disclosure Agreement (the 'Agreement') is entered into by and between\n"
        "Vanguard Systems Corporation ('Disclosing Party') and Apex Analytics Group ('Receiving Party').\n\n"
        "2. CONFIDENTIAL INFORMATION: Confidential Information refers to any proprietary information,\n"
        "trade secrets, machine learning models, source code, and customer records disclosed directly or indirectly.\n\n"
        "3. OBLIGATIONS: The Receiving Party agrees to protect the Confidential Information with a standard of care\n"
        "no less than reasonable care, and agrees not to disclose such information to any third party.\n\n"
        "4. FINANCIAL PENALTY & REMEDIES: In the event of an unauthorized breach, the breaching party shall be liable\n"
        "for liquidated damages in the amount of $250,000.00 USD plus direct legal fees incurred.\n\n"
        "5. TERM & TERMINATION: The obligations under this Agreement shall survive for a period of 5 years\n"
        "following the Effective Date specified above."
    )
    draw.text((50, 160), text_p1, fill=(15, 23, 42), font=f_body)

    # Signature Block
    draw.text((50, 680), "IN WITNESS WHEREOF, the parties hereto have executed this Agreement.", fill=(15, 23, 42), font=f_body)

    draw.text((50, 750), "DISCLOSING PARTY:", fill=(71, 85, 105), font=f_sub)
    draw.line([50, 830, 300, 830], fill=(15, 23, 42), width=2)
    draw.text((60, 800), "Jonathan R. Vance", fill=(30, 58, 138), font=get_font(20, bold=True))
    draw.text((50, 840), "Jonathan R. Vance, Executive VP\nVanguard Systems Corp", fill=(71, 85, 105), font=f_small)

    draw.text((450, 750), "RECEIVING PARTY:", fill=(71, 85, 105), font=f_sub)
    draw.line([450, 830, 700, 830], fill=(15, 23, 42), width=2)
    draw.text((460, 800), "Helena Troy-Smith", fill=(30, 58, 138), font=get_font(20, bold=True))
    draw.text((450, 840), "Helena Troy-Smith, Managing Director\nApex Analytics Group", fill=(71, 85, 105), font=f_small)

    return add_document_texture(img)


def create_receipt_template(doc_id: str) -> Image.Image:
    """Generate a thermal / retail store receipt."""
    w, h = 600, 900
    img = Image.new("RGB", (w, h), color=(250, 250, 248))
    draw = ImageDraw.Draw(img)

    f_title = get_font(24, bold=True)
    f_sub = get_font(16, bold=True)
    f_body = get_font(15, bold=False)
    f_small = get_font(12, bold=False)

    # Store Header
    draw.text((w // 2 - 120, 40), "METRO MART SUPERSTORE", fill=(15, 23, 42), font=f_title)
    draw.text((w // 2 - 90, 75), "Store #0482 - Terminal #03", fill=(71, 85, 105), font=f_small)
    draw.text((w // 2 - 110, 95), "4050 Grand Ave, Phoenix AZ", fill=(71, 85, 105), font=f_small)

    draw.line([40, 130, w - 40, 130], fill=(100, 116, 139), width=1)
    rec_id = f"REC-994-{random.randint(100000, 999999)}"
    draw.text((50, 145), f"Receipt: {rec_id}", fill=(15, 23, 42), font=f_body)
    draw.text((380, 145), "Date: 2026-04-12 14:32", fill=(15, 23, 42), font=f_body)
    draw.line([40, 175, w - 40, 175], fill=(100, 116, 139), width=1)

    items = [
        ("ORGANIC ESPRESSO BEANS 1KG", 24.99),
        ("ALMOND MILK UNSWEETENED 1L", 4.49),
        ("PREMIUM WIRELESS MOUSE", 39.99),
        ("USB-C FAST CHARGING CABLE 2M", 18.50),
        ("ERGONOMIC DESK MAT (BLACK)", 29.99),
    ]

    y = 200
    subtotal = sum(price for _, price in items)
    for name, price in items:
        draw.text((50, y), name, fill=(15, 23, 42), font=f_body)
        draw.text((480, y), f"${price:.2f}", fill=(15, 23, 42), font=f_body)
        y += 40

    draw.line([40, y + 10, w - 40, y + 10], fill=(100, 116, 139), width=1)
    tax = subtotal * 0.0825
    total = subtotal + tax

    y += 30
    draw.text((320, y), "SUBTOTAL:", fill=(71, 85, 105), font=f_sub)
    draw.text((480, y), f"${subtotal:.2f}", fill=(15, 23, 42), font=f_body)

    y += 35
    draw.text((320, y), "TAX (8.25%):", fill=(71, 85, 105), font=f_sub)
    draw.text((480, y), f"${tax:.2f}", fill=(15, 23, 42), font=f_body)

    y += 40
    draw.text((320, y), "TOTAL AMOUNT:", fill=(15, 23, 42), font=f_title)
    draw.text((460, y), f"${total:.2f}", fill=(15, 23, 42), font=f_title)

    # Barcode
    y += 90
    draw.rectangle([100, y, w - 100, y + 60], fill=(226, 232, 240))
    # Simulated vertical barcode bars
    for x in range(120, w - 120, random.choice([3, 5, 8])):
        draw.line([x, y + 5, x, y + 55], fill=(15, 23, 42), width=random.choice([1, 2, 3]))
    draw.text((w // 2 - 70, y + 70), "* 0482 9942 1085 *", fill=(71, 85, 105), font=f_small)

    return add_document_texture(img)


GENERATOR_MAP = {
    "invoice": create_invoice_template,
    "id_card": create_id_card_template,
    "certificate": create_certificate_template,
    "contract": create_contract_template,
    "receipt": create_receipt_template,
}


def generate_document_family(family_index: int, doc_type: str, count_per_family: int = 1) -> list[tuple[str, Image.Image]]:
    """Generate base document instances for a given document family."""
    gen_fn = GENERATOR_MAP.get(doc_type, create_invoice_template)
    family_id = f"doc_family_{doc_type}_{family_index:03d}"
    results = []
    for i in range(count_per_family):
        doc_id = f"{family_id}_inst_{i:02d}"
        img = gen_fn(doc_id)
        results.append((doc_id, family_id, img))
    return results


def build_raw_authentic_corpus(num_families_per_type: int = 20) -> list[dict]:
    """Generate raw clean documents across all document types."""
    np.random.seed(cfg.dataset.seed)
    random.seed(cfg.dataset.seed)

    saved_records = []
    for doc_type in GENERATOR_MAP.keys():
        for f_idx in range(1, num_families_per_type + 1):
            instances = generate_document_family(f_idx, doc_type)
            for doc_id, family_id, img in instances:
                save_path = AUTHENTIC_RAW_DIR / f"{doc_id}.png"
                img.save(save_path, "PNG")
                saved_records.append({
                    "doc_id": doc_id,
                    "family_id": family_id,
                    "doc_type": doc_type,
                    "file_path": str(save_path),
                    "width": img.width,
                    "height": img.height,
                })
    return saved_records


if __name__ == "__main__":
    records = build_raw_authentic_corpus(num_families_per_type=20)
    print(f"Generated {len(records)} authentic documents across {len(GENERATOR_MAP)} document types in {AUTHENTIC_RAW_DIR}")
