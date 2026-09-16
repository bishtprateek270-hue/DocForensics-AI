"""
DocForensics AI — Multi-Technique Document Tampering Engine
Implements realistic document forgery methods and generates pixel-accurate ground-truth binary masks:
1. Copy-Move Forgery
2. Splicing / Foreign Element Insertion
3. Content Erasure / Inpainting
4. Text / Digit Alteration & Overwriting
5. Forensic Noise & JPEG Compression Artifact Injection
"""

import io
import random
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import cv2

from config import SYNTHETIC_DATA_DIR, cfg


def get_font(size: int = 18, bold: bool = False):
    """Load system font for text manipulation."""
    font_names = [
        "arialbd.ttf" if bold else "arial.ttf",
        "calibrib.ttf" if bold else "calibri.ttf",
        "timesbd.ttf" if bold else "times.ttf",
    ]
    for fn in font_names:
        try:
            return ImageFont.truetype(fn, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def apply_jpeg_artifact(patch: np.ndarray, quality: int = 40) -> np.ndarray:
    """Simulate local compression mismatch by recompressing patch at low quality."""
    pil_patch = Image.fromarray(patch)
    buf = io.BytesIO()
    pil_patch.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    recompressed = Image.open(buf)
    return np.array(recompressed.convert("RGB"))


def tamper_copy_move(img: Image.Image, seed: int = None) -> tuple[Image.Image, np.ndarray, str]:
    """
    Extract a patch from within the document (e.g. text line, signature, number)
    and paste it onto another valid document location.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    img_arr = np.array(img).copy()
    h, w, c = img_arr.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    # Pick patch size (10% to 30% of width, 3% to 10% of height)
    patch_w = random.randint(int(w * 0.12), int(w * 0.35))
    patch_h = random.randint(int(h * 0.04), int(h * 0.10))

    # Source coordinate (ensure within bounds and non-trivial area)
    src_x = random.randint(50, w - patch_w - 50)
    src_y = random.randint(int(h * 0.15), int(h * 0.70))

    # Target coordinate (must differ significantly from source)
    attempts = 0
    while attempts < 20:
        dst_x = random.randint(40, w - patch_w - 40)
        dst_y = random.randint(int(h * 0.10), int(h * 0.85))
        if abs(dst_x - src_x) > 50 or abs(dst_y - src_y) > 40:
            break
        attempts += 1

    patch = img_arr[src_y:src_y + patch_h, src_x:src_x + patch_w]
    
    # Optional slight rotation or scaling
    if random.random() > 0.5:
        patch = apply_jpeg_artifact(patch, quality=random.randint(45, 75))

    img_arr[dst_y:dst_y + patch_h, dst_x:dst_x + patch_w] = patch
    mask[dst_y:dst_y + patch_h, dst_x:dst_x + patch_w] = 255

    return Image.fromarray(img_arr), mask, "copy_move"


def tamper_splicing(img: Image.Image, seed: int = None) -> tuple[Image.Image, np.ndarray, str]:
    """
    Insert a foreign official stamp, seal, signature, or badge with alpha blending
    and distinct forensic noise footprint.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    img_arr = np.array(img).copy()
    h, w, c = img_arr.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    # Generate synthetic foreign stamp
    stamp_size = random.randint(int(min(h, w) * 0.15), int(min(h, w) * 0.28))
    stamp_img = Image.new("RGBA", (stamp_size, stamp_size), (0, 0, 0, 0))
    stamp_draw = ImageDraw.Draw(stamp_img)

    colors = [
        (220, 38, 38, 220),   # Red stamp
        (37, 99, 235, 220),   # Blue stamp
        (22, 163, 74, 220),   # Green stamp
        (147, 51, 234, 220),  # Purple seal
    ]
    stamp_color = random.choice(colors)

    # Circle stamp
    stamp_draw.ellipse([4, 4, stamp_size - 4, stamp_size - 4], outline=stamp_color, width=4)
    stamp_draw.ellipse([12, 12, stamp_size - 12, stamp_size - 12], outline=stamp_color, width=1)
    
    font_s = get_font(max(10, stamp_size // 9), bold=True)
    stamp_draw.text((stamp_size // 2 - 35, stamp_size // 2 - 20), "APPROVED\nFORGED", fill=stamp_color, font=font_s, align="center")

    # Position on document
    dst_x = random.randint(int(w * 0.40), w - stamp_size - 40)
    dst_y = random.randint(int(h * 0.45), h - stamp_size - 40)

    # Blend onto target
    pil_doc = Image.fromarray(img_arr)
    pil_doc.paste(stamp_img, (dst_x, dst_y), stamp_img)
    img_arr = np.array(pil_doc)

    # Mark mask for alpha > 0
    stamp_alpha = np.array(stamp_img)[:, :, 3]
    binary_stamp_mask = (stamp_alpha > 30).astype(np.uint8) * 255
    mask[dst_y:dst_y + stamp_size, dst_x:dst_x + stamp_size] = np.maximum(
        mask[dst_y:dst_y + stamp_size, dst_x:dst_x + stamp_size],
        binary_stamp_mask
    )

    return Image.fromarray(img_arr), mask, "splicing"


def tamper_erasure_inpainting(img: Image.Image, seed: int = None) -> tuple[Image.Image, np.ndarray, str]:
    """
    Erase text, amounts, or dates using OpenCV inpainting algorithms (Telea / Navier-Stokes)
    or localized background cloning.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    img_arr = np.array(img).copy()
    h, w, c = img_arr.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    # Choose a rectangular region likely containing text (middle or financial total area)
    box_w = random.randint(int(w * 0.15), int(w * 0.35))
    box_h = random.randint(int(h * 0.03), int(h * 0.08))

    box_x = random.randint(int(w * 0.20), w - box_w - 40)
    box_y = random.randint(int(h * 0.25), int(h * 0.80))

    inpaint_mask = np.zeros((h, w), dtype=np.uint8)
    inpaint_mask[box_y:box_y + box_h, box_x:box_x + box_w] = 255

    # Run OpenCV Telea Inpainting
    inpainted = cv2.inpaint(img_arr, inpaint_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    
    # Add subtle residual noise
    noise = np.random.normal(0, 3, (box_h, box_w, c))
    patch = inpainted[box_y:box_y + box_h, box_x:box_x + box_w].astype(np.float32) + noise
    inpainted[box_y:box_y + box_h, box_x:box_x + box_w] = np.clip(patch, 0, 255).astype(np.uint8)

    mask[box_y:box_y + box_h, box_x:box_x + box_w] = 255

    return Image.fromarray(inpainted), mask, "erasure_inpainting"


def tamper_text_alteration(img: Image.Image, seed: int = None) -> tuple[Image.Image, np.ndarray, str]:
    """
    Alter numbers, financial amounts, dates, or names by overwriting target text lines
    with mismatched fonts, colors, or sizes.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    img_arr = np.array(img).copy()
    h, w, c = img_arr.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    # Region for alteration
    box_w = random.randint(int(w * 0.18), int(w * 0.32))
    box_h = random.randint(int(h * 0.03), int(h * 0.06))

    box_x = random.randint(int(w * 0.45), w - box_w - 30)
    box_y = random.randint(int(h * 0.30), int(h * 0.75))

    # Background color sample near target
    bg_sample = img_arr[max(0, box_y - 5):box_y, box_x:box_x + box_w]
    bg_color = tuple(np.mean(bg_sample, axis=(0, 1)).astype(int)) if bg_sample.size > 0 else (255, 255, 255)

    pil_doc = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(pil_doc)

    # Cover existing text
    draw.rectangle([box_x, box_y, box_x + box_w, box_y + box_h], fill=bg_color)

    # Forged text
    forged_values = [
        f"${random.randint(50, 99)},{random.randint(100, 999)}.{random.randint(10, 99)}",
        f"2029-12-{random.randint(10, 28)}",
        f"AMOUNT DUE: ${random.randint(85000, 99000)}.00",
        f"ID-FORGED-{random.randint(100000, 999999)}",
        "CLEARANCE LEVEL: TOP SECRET / VOID",
    ]
    forged_str = random.choice(forged_values)
    font_alt = get_font(size=random.choice([16, 18, 20, 22]), bold=random.choice([True, False]))
    text_color = random.choice([(15, 23, 42), (185, 28, 28), (30, 58, 138), (0, 0, 0)])

    draw.text((box_x + 5, box_y + 4), forged_str, fill=text_color, font=font_alt)
    img_arr = np.array(pil_doc)

    mask[box_y:box_y + box_h, box_x:box_x + box_w] = 255

    return Image.fromarray(img_arr), mask, "text_alteration"


TAMPERING_TECHNIQUES = [
    tamper_copy_move,
    tamper_splicing,
    tamper_erasure_inpainting,
    tamper_text_alteration,
]


def apply_tampering(img: Image.Image, technique: str = "random", seed: int = None) -> tuple[Image.Image, np.ndarray, str]:
    """
    Apply single or combined tampering techniques to an authentic document.
    """
    if technique == "copy_move":
        return tamper_copy_move(img, seed)
    elif technique == "splicing":
        return tamper_splicing(img, seed)
    elif technique == "erasure_inpainting":
        return tamper_erasure_inpainting(img, seed)
    elif technique == "text_alteration":
        return tamper_text_alteration(img, seed)
    else:
        # Pick random technique
        chosen_fn = random.choice(TAMPERING_TECHNIQUES)
        return chosen_fn(img, seed)


if __name__ == "__main__":
    from src.preprocessing.data_collector import create_invoice_template
    base = create_invoice_template("test_01")
    tampered_img, mask, tech = apply_tampering(base, technique="random", seed=42)
    print(f"Sample tampering generated: technique='{tech}', tampered pixels={np.sum(mask > 0)}")
