"""
DocForensics AI — Multi-Category Document Tampering Engine (Phase 3)
Implements 10 distinct realistic document manipulation operations with pixel-accurate
ground-truth binary masks and precise bounding boxes:

1. Text Replacement (text_replacement)
2. Number/Amount/Marks/CGPA Replacement (number_amount_replacement)
3. Date Replacement (date_replacement)
4. Copy-Move within Same Document (copy_move_intra)
5. Copy-Paste from Another Region/Document (copy_paste_inter)
6. Signature Insertion/Replacement (signature_manipulation)
7. Photo/Image Replacement (photo_replacement)
8. Stamp/Seal Insertion/Replacement (stamp_seal_manipulation)
9. Region/Object Removal (region_removal)
10. Local Inpainting/Removal (local_inpainting)
"""

import io
import random
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import cv2


def get_font(size: int = 18, bold: bool = False):
    """Load system font for text manipulation."""
    font_names = [
        "arialbd.ttf" if bold else "arial.ttf",
        "calibrib.ttf" if bold else "calibri.ttf",
        "timesbd.ttf" if bold else "times.ttf",
        "segoeuib.ttf" if bold else "segoeui.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for fn in font_names:
        try:
            return ImageFont.truetype(fn, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def sample_background_color(img_arr: np.ndarray, x: int, y: int, w: int, h: int) -> tuple[int, int, int]:
    """Sample the surrounding background color near the target region."""
    ih, iw, _ = img_arr.shape
    y_start = max(0, y - 6)
    y_end = min(ih, y + h + 6)
    x_start = max(0, x - 6)
    x_end = min(iw, x + w + 6)

    surrounding = []
    if y > 6:
        surrounding.append(img_arr[y_start:y, x:x + w])
    if y + h + 6 < ih:
        surrounding.append(img_arr[y + h:y_end, x:x + w])
    if x > 6:
        surrounding.append(img_arr[y:y + h, x_start:x])
    if x + w + 6 < iw:
        surrounding.append(img_arr[y:y + h, x + w:x_end])

    if surrounding:
        combined = np.concatenate([s.reshape(-1, 3) for s in surrounding if s.size > 0], axis=0)
        mean_col = np.median(combined, axis=0).astype(int)
        return int(mean_col[0]), int(mean_col[1]), int(mean_col[2])
    return 252, 252, 250


# =========================================================================
# 1. Text Replacement
# =========================================================================
def tamper_text_replacement(img: Image.Image, seed: int = None) -> tuple[Image.Image, np.ndarray, list[int], str]:
    """Replace an existing text clause, recipient name, or line with modified text."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    img_arr = np.array(img).copy()
    h, w, c = img_arr.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    box_w = random.randint(int(w * 0.20), int(w * 0.40))
    box_h = random.randint(int(h * 0.03), int(h * 0.06))
    box_x = random.randint(50, w - box_w - 50)
    box_y = random.randint(int(h * 0.20), int(h * 0.70))

    bg_color = sample_background_color(img_arr, box_x, box_y, box_w, box_h)
    pil_doc = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(pil_doc)
    draw.rectangle([box_x, box_y, box_x + box_w, box_y + box_h], fill=bg_color)

    texts = [
        "AUTHORIZED BY GLOBAL HEADQUARTERS",
        "Apex Systems & Analytics International",
        "EXCLUSIVE & UNCONDITIONAL WAIVER",
        "CLEARANCE LEVEL: TOP SECRET / VOID",
        "DEPARTMENT OF REVENUE & COMPLIANCE",
    ]
    font = get_font(size=random.choice([15, 17, 19]), bold=random.choice([True, False]))
    text_color = random.choice([(15, 23, 42), (30, 41, 59), (185, 28, 28), (30, 58, 138)])
    draw.text((box_x + 6, box_y + 4), random.choice(texts), fill=text_color, font=font)

    mask[box_y:box_y + box_h, box_x:box_x + box_w] = 255
    bbox = [box_x, box_y, box_x + box_w, box_y + box_h]
    return pil_doc, mask, bbox, "text_replacement"


# =========================================================================
# 2. Number / Amount / Marks / CGPA Replacement
# =========================================================================
def tamper_number_amount_replacement(img: Image.Image, seed: int = None) -> tuple[Image.Image, np.ndarray, list[int], str]:
    """Alter numeric figures such as totals, prices, invoice amounts, CGPA or marks."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    img_arr = np.array(img).copy()
    h, w, c = img_arr.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    box_w = random.randint(int(w * 0.14), int(w * 0.26))
    box_h = random.randint(int(h * 0.03), int(h * 0.055))
    box_x = random.randint(int(w * 0.45), w - box_w - 40)
    box_y = random.randint(int(h * 0.30), int(h * 0.78))

    bg_color = sample_background_color(img_arr, box_x, box_y, box_w, box_h)
    pil_doc = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(pil_doc)
    draw.rectangle([box_x, box_y, box_x + box_w, box_y + box_h], fill=bg_color)

    amounts = [
        f"${random.randint(45, 99)},{random.randint(100, 999)}.{random.randint(10, 99)}",
        f"CGPA: {random.uniform(3.75, 3.99):.2f} / 4.00",
        f"TOTAL: ${random.randint(75000, 98500):,}.00",
        f"SCORE: {random.randint(95, 100)}/100 (A+)",
        f"${random.randint(12000, 48000):,}.50",
    ]
    font = get_font(size=random.choice([16, 18, 20, 22]), bold=True)
    text_color = random.choice([(185, 28, 28), (15, 23, 42), (30, 58, 138)])
    draw.text((box_x + 4, box_y + 3), random.choice(amounts), fill=text_color, font=font)

    mask[box_y:box_y + box_h, box_x:box_x + box_w] = 255
    bbox = [box_x, box_y, box_x + box_w, box_y + box_h]
    return pil_doc, mask, bbox, "number_amount_replacement"


# =========================================================================
# 3. Date Replacement
# =========================================================================
def tamper_date_replacement(img: Image.Image, seed: int = None) -> tuple[Image.Image, np.ndarray, list[int], str]:
    """Alter dates such as issue dates, expiration dates, or execution timestamps."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    img_arr = np.array(img).copy()
    h, w, c = img_arr.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    box_w = random.randint(int(w * 0.16), int(w * 0.28))
    box_h = random.randint(int(h * 0.028), int(h * 0.05))
    box_x = random.randint(int(w * 0.40), w - box_w - 40)
    box_y = random.randint(int(h * 0.12), int(h * 0.40))

    bg_color = sample_background_color(img_arr, box_x, box_y, box_w, box_h)
    pil_doc = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(pil_doc)
    draw.rectangle([box_x, box_y, box_x + box_w, box_y + box_h], fill=bg_color)

    dates = [
        f"2030-12-{random.randint(10, 28)}",
        f"October {random.randint(10, 28)}, 2029",
        f"EXP: 08/{random.randint(28, 35)}",
        f"VALID THRU: 2032-06-30",
        f"Date: 2028-04-{random.randint(10, 28)}",
    ]
    font = get_font(size=random.choice([15, 17, 19]), bold=False)
    draw.text((box_x + 4, box_y + 3), random.choice(dates), fill=(15, 23, 42), font=font)

    mask[box_y:box_y + box_h, box_x:box_x + box_w] = 255
    bbox = [box_x, box_y, box_x + box_w, box_y + box_h]
    return pil_doc, mask, bbox, "date_replacement"


# =========================================================================
# 4. Copy-Move within the Same Document
# =========================================================================
def tamper_copy_move_intra(img: Image.Image, seed: int = None) -> tuple[Image.Image, np.ndarray, list[int], str]:
    """Clone a snippet from one part of the document and paste it onto another valid coordinate."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    img_arr = np.array(img).copy()
    h, w, c = img_arr.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    patch_w = random.randint(int(w * 0.12), int(w * 0.32))
    patch_h = random.randint(int(h * 0.035), int(h * 0.09))

    src_x = random.randint(40, w - patch_w - 40)
    src_y = random.randint(int(h * 0.15), int(h * 0.65))

    dst_x = random.randint(40, w - patch_w - 40)
    dst_y = random.randint(int(h * 0.15), int(h * 0.85))

    patch = img_arr[src_y:src_y + patch_h, src_x:src_x + patch_w].copy()
    img_arr[dst_y:dst_y + patch_h, dst_x:dst_x + patch_w] = patch
    mask[dst_y:dst_y + patch_h, dst_x:dst_x + patch_w] = 255

    bbox = [dst_x, dst_y, dst_x + patch_w, dst_y + patch_h]
    return Image.fromarray(img_arr), mask, bbox, "copy_move_intra"


# =========================================================================
# 5. Copy-Paste from Another Region / Document (Inter-Document Splicing)
# =========================================================================
def tamper_copy_paste_inter(img: Image.Image, donor_img: Image.Image = None, seed: int = None) -> tuple[Image.Image, np.ndarray, list[int], str]:
    """Splice a patch from a donor document or synthetic foreign header/table snippet."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    img_arr = np.array(img).copy()
    h, w, c = img_arr.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    patch_w = random.randint(int(w * 0.18), int(w * 0.35))
    patch_h = random.randint(int(h * 0.04), int(h * 0.08))

    # Generate synthetic foreign badge/clause
    foreign_patch = Image.new("RGB", (patch_w, patch_h), color=(241, 245, 249))
    draw = ImageDraw.Draw(foreign_patch)
    draw.rectangle([0, 0, patch_w, patch_h], outline=(148, 163, 184), width=1)
    draw.text((8, 6), f"EXTERNAL AUDIT #EA-{random.randint(1000, 9999)}", fill=(30, 41, 59), font=get_font(14, bold=True))

    dst_x = random.randint(40, w - patch_w - 40)
    dst_y = random.randint(int(h * 0.20), int(h * 0.80))

    img_arr[dst_y:dst_y + patch_h, dst_x:dst_x + patch_w] = np.array(foreign_patch)
    mask[dst_y:dst_y + patch_h, dst_x:dst_x + patch_w] = 255

    bbox = [dst_x, dst_y, dst_x + patch_w, dst_y + patch_h]
    return Image.fromarray(img_arr), mask, bbox, "copy_paste_inter"


# =========================================================================
# 6. Signature Insertion / Replacement
# =========================================================================
def tamper_signature_manipulation(img: Image.Image, seed: int = None) -> tuple[Image.Image, np.ndarray, list[int], str]:
    """Insert or overwrite an executive or official signature."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    img_arr = np.array(img).copy()
    h, w, c = img_arr.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    sig_w = random.randint(int(w * 0.22), int(w * 0.38))
    sig_h = random.randint(int(h * 0.05), int(h * 0.10))

    dst_x = random.randint(50, w - sig_w - 50)
    dst_y = random.randint(int(h * 0.65), h - sig_h - 40)

    bg_color = sample_background_color(img_arr, dst_x, dst_y, sig_w, sig_h)
    pil_doc = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(pil_doc)
    draw.rectangle([dst_x, dst_y, dst_x + sig_w, dst_y + sig_h], fill=bg_color)

    # Render cursive-style forged signature
    sig_names = ["Robert M. Sterling", "Elena Rostova", "Dr. Kenneth Clarke", "Victoria H. Sterling"]
    f_sig = get_font(size=random.choice([22, 24, 26]), bold=True)
    draw.text((dst_x + 10, dst_y + 8), random.choice(sig_names), fill=(30, 58, 138), font=f_sig)
    draw.line([dst_x + 5, dst_y + sig_h - 10, dst_x + sig_w - 5, dst_y + sig_h - 10], fill=(100, 116, 139), width=1)

    mask[dst_y:dst_y + sig_h, dst_x:dst_x + sig_w] = 255
    bbox = [dst_x, dst_y, dst_x + sig_w, dst_y + sig_h]
    return pil_doc, mask, bbox, "signature_manipulation"


# =========================================================================
# 7. Photo / Image Replacement
# =========================================================================
def tamper_photo_replacement(img: Image.Image, seed: int = None) -> tuple[Image.Image, np.ndarray, list[int], str]:
    """Replace an identity portrait or credential photo with a forged photo box."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    img_arr = np.array(img).copy()
    h, w, c = img_arr.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    photo_w = random.randint(int(w * 0.18), int(w * 0.28))
    photo_h = random.randint(int(h * 0.22), int(h * 0.35))

    dst_x = random.randint(40, int(w * 0.35))
    dst_y = random.randint(int(h * 0.18), int(h * 0.55))

    pil_doc = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(pil_doc)

    # Forged avatar photo replacement
    bg_avatar = random.choice([(186, 230, 253), (254, 215, 170), (221, 214, 254)])
    draw.rectangle([dst_x, dst_y, dst_x + photo_w, dst_y + photo_h], fill=bg_avatar, outline=(30, 41, 59), width=2)
    # Draw simple silhouette
    draw.ellipse([dst_x + int(photo_w * 0.25), dst_y + int(photo_h * 0.15),
                  dst_x + int(photo_w * 0.75), dst_y + int(photo_h * 0.55)], fill=(71, 85, 105))
    draw.chord([dst_x + int(photo_w * 0.10), dst_y + int(photo_h * 0.50),
                dst_x + int(photo_w * 0.90), dst_y + int(photo_h * 1.10)], 180, 360, fill=(51, 65, 85))

    mask[dst_y:dst_y + photo_h, dst_x:dst_x + photo_w] = 255
    bbox = [dst_x, dst_y, dst_x + photo_w, dst_y + photo_h]
    return pil_doc, mask, bbox, "photo_replacement"


# =========================================================================
# 8. Stamp / Seal Insertion or Replacement
# =========================================================================
def tamper_stamp_seal_manipulation(img: Image.Image, seed: int = None) -> tuple[Image.Image, np.ndarray, list[int], str]:
    """Insert or overwrite an official certification seal, notary stamp, or validation emblem."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    img_arr = np.array(img).copy()
    h, w, c = img_arr.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    stamp_size = random.randint(int(min(h, w) * 0.15), int(min(h, w) * 0.25))
    stamp_img = Image.new("RGBA", (stamp_size, stamp_size), (0, 0, 0, 0))
    stamp_draw = ImageDraw.Draw(stamp_img)

    colors = [
        (220, 38, 38, 230),   # Red
        (37, 99, 235, 230),   # Blue
        (22, 163, 74, 230),   # Green
        (217, 119, 6, 230),   # Gold
    ]
    scol = random.choice(colors)

    stamp_draw.ellipse([3, 3, stamp_size - 3, stamp_size - 3], outline=scol, width=4)
    stamp_draw.ellipse([10, 10, stamp_size - 10, stamp_size - 10], outline=scol, width=1)
    
    f_s = get_font(max(9, stamp_size // 8), bold=True)
    stamp_draw.text((stamp_size // 2 - 32, stamp_size // 2 - 18), "AUTHENTIC\nVERIFIED", fill=scol, font=f_s, align="center")

    dst_x = random.randint(int(w * 0.35), w - stamp_size - 40)
    dst_y = random.randint(int(h * 0.40), h - stamp_size - 40)

    pil_doc = Image.fromarray(img_arr)
    pil_doc.paste(stamp_img, (dst_x, dst_y), stamp_img)

    stamp_alpha = np.array(stamp_img)[:, :, 3]
    binary_stamp = (stamp_alpha > 30).astype(np.uint8) * 255
    mask[dst_y:dst_y + stamp_size, dst_x:dst_x + stamp_size] = binary_stamp

    bbox = [dst_x, dst_y, dst_x + stamp_size, dst_y + stamp_size]
    return pil_doc, mask, bbox, "stamp_seal_manipulation"


# =========================================================================
# 9. Region / Object Removal
# =========================================================================
def tamper_region_removal(img: Image.Image, seed: int = None) -> tuple[Image.Image, np.ndarray, list[int], str]:
    """Erase a prominent document object (e.g. barcode, disclaimer box, or seal)."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    img_arr = np.array(img).copy()
    h, w, c = img_arr.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    box_w = random.randint(int(w * 0.20), int(w * 0.45))
    box_h = random.randint(int(h * 0.05), int(h * 0.12))
    box_x = random.randint(40, w - box_w - 40)
    box_y = random.randint(int(h * 0.40), int(h * 0.88))

    bg_color = sample_background_color(img_arr, box_x, box_y, box_w, box_h)
    pil_doc = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(pil_doc)
    draw.rectangle([box_x, box_y, box_x + box_w, box_y + box_h], fill=bg_color)

    mask[box_y:box_y + box_h, box_x:box_x + box_w] = 255
    bbox = [box_x, box_y, box_x + box_w, box_y + box_h]
    return pil_doc, mask, bbox, "region_removal"


# =========================================================================
# 10. Local Inpainting / Removal
# =========================================================================
def tamper_local_inpainting(img: Image.Image, seed: int = None) -> tuple[Image.Image, np.ndarray, list[int], str]:
    """Remove small text, amounts, or serial numbers using Navier-Stokes/Telea inpainting."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    img_arr = np.array(img).copy()
    h, w, c = img_arr.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    box_w = random.randint(int(w * 0.14), int(w * 0.30))
    box_h = random.randint(int(h * 0.025), int(h * 0.06))
    box_x = random.randint(50, w - box_w - 50)
    box_y = random.randint(int(h * 0.22), int(h * 0.78))

    inpaint_mask = np.zeros((h, w), dtype=np.uint8)
    inpaint_mask[box_y:box_y + box_h, box_x:box_x + box_w] = 255

    inpainted = cv2.inpaint(img_arr, inpaint_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    mask[box_y:box_y + box_h, box_x:box_x + box_w] = 255
    bbox = [box_x, box_y, box_x + box_w, box_y + box_h]
    return Image.fromarray(inpainted), mask, bbox, "local_inpainting"


# =========================================================================
# Dispatcher & Category Registry
# =========================================================================
MANIPULATION_REGISTRY = {
    "text_replacement": tamper_text_replacement,
    "number_amount_replacement": tamper_number_amount_replacement,
    "date_replacement": tamper_date_replacement,
    "copy_move_intra": tamper_copy_move_intra,
    "copy_paste_inter": tamper_copy_paste_inter,
    "signature_manipulation": tamper_signature_manipulation,
    "photo_replacement": tamper_photo_replacement,
    "stamp_seal_manipulation": tamper_stamp_seal_manipulation,
    "region_removal": tamper_region_removal,
    "local_inpainting": tamper_local_inpainting,
}

MANIPULATION_CATEGORIES = list(MANIPULATION_REGISTRY.keys())


def apply_manipulation(
    img: Image.Image,
    manipulation_type: str = "random",
    seed: int = None
) -> tuple[Image.Image, np.ndarray, list[int], str]:
    """
    Apply requested manipulation category to clean document.
    Returns: (tampered_image, binary_mask, bounding_box, manipulation_type)
    """
    if manipulation_type in MANIPULATION_REGISTRY:
        fn = MANIPULATION_REGISTRY[manipulation_type]
        return fn(img, seed=seed)
    else:
        chosen_type = random.choice(MANIPULATION_CATEGORIES)
        fn = MANIPULATION_REGISTRY[chosen_type]
        return fn(img, seed=seed)


if __name__ == "__main__":
    from src.preprocessing.data_collector import create_invoice_template
    base = create_invoice_template("test_phase3")
    for cat in MANIPULATION_CATEGORIES:
        t_img, m, bb, c = apply_manipulation(base, manipulation_type=cat, seed=42)
        print(f"[{c:<26}] Pixels: {np.sum(m > 0):<6} | BBox: {bb}")
