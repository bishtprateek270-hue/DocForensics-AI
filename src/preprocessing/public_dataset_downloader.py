"""
DocForensics AI — Public Dataset Downloader (RealText-V2)
Integrates the official RealText-V2 public document tampering benchmark:
- Dataset Name: RealText-V2 (ACM MM 2026 GenText-Forensics Challenge)
- Source: Hugging Face (https://huggingface.co/datasets/vankey/RealText-V2)
- License: CC-BY-NC 4.0
- Ground Truth: Pixel-level binary localization masks (0: authentic, 255: tampered)
"""

import os
import io
import json
import urllib.request
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm

from config import RAW_DATA_DIR, cfg


PUBLIC_RAW_DIR = RAW_DATA_DIR / "public_realtext_v2"
PUBLIC_IMAGES_DIR = PUBLIC_RAW_DIR / "images"
PUBLIC_MASKS_DIR = PUBLIC_RAW_DIR / "masks"

for d in [PUBLIC_RAW_DIR, PUBLIC_IMAGES_DIR, PUBLIC_MASKS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

HF_BASE_URL = "https://huggingface.co/datasets/vankey/RealText-V2/resolve/main"

DATASET_PROVENANCE = {
    "official_dataset_name": "RealText-V2: A Large-Scale Multilingual Document Forgery Analysis Benchmark",
    "short_name": "RealText-V2",
    "official_source": "https://huggingface.co/datasets/vankey/RealText-V2",
    "organization": "ACM MM 2026 MGC: GenText-Forensics Challenge",
    "license": "CC-BY-NC-4.0",
    "license_url": "https://creativecommons.org/licenses/by-nc/4.0/",
    "data_type": "Real and AI-manipulated multilingual document images with ground-truth binary tampering masks",
    "ground_truth_mask_convention": "0 = authentic / background, 255 = tampered pixel region",
    "supported_tasks": ["tampering_classification", "pixel_level_localization", "multilingual_ocr_forensics"]
}


def download_file(url: str) -> bytes:
    """Download binary data with proper User-Agent headers."""
    req = urllib.request.Request(url, headers={"User-Agent": "DocForensicsAI/1.0 (Research)"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def download_realtext_v2_subset(num_tampered: int = 30, num_authentic: int = 20) -> list[dict]:
    """
    Download a verified subset of RealText-V2 document images and ground-truth masks.
    """
    print("=" * 70)
    print(f"[*] Fetching Public Dataset: {DATASET_PROVENANCE['official_dataset_name']}")
    print(f"[*] Source: {DATASET_PROVENANCE['official_source']} | License: {DATASET_PROVENANCE['license']}")
    print("=" * 70)

    # Save provenance metadata
    with open(PUBLIC_RAW_DIR / "DATASET_INFO.json", "w", encoding="utf-8") as f:
        json.dump(DATASET_PROVENANCE, f, indent=2)

    # Query file index from Hugging Face API
    req = urllib.request.Request("https://huggingface.co/api/datasets/vankey/RealText-V2", headers={"User-Agent": "DocForensicsAI/1.0"})
    with urllib.request.urlopen(req) as resp:
        hf_data = json.loads(resp.read().decode("utf-8"))
    
    siblings = [x["rfilename"] for x in hf_data.get("siblings", [])]
    
    # Map images and masks
    image_files = {s.split("/")[-1].split(".")[0]: s for s in siblings if s.startswith("train/image/")}
    mask_files = {s.split("/")[-1].replace("_mask.png", ""): s for s in siblings if s.startswith("train/mask/")}
    
    tampered_keys = sorted(list(set(image_files.keys()) & set(mask_files.keys())))[:num_tampered]
    authentic_keys = sorted(list(set(image_files.keys()) - set(mask_files.keys())))[:num_authentic]

    downloaded_records = []

    # 1. Download Tampered Documents with Ground-Truth Masks
    print(f"[*] Downloading {len(tampered_keys)} tampered document images with pixel masks...")
    for file_id in tqdm(tampered_keys, desc="Tampered RealText-V2"):
        img_rel = image_files[file_id]
        mask_rel = mask_files[file_id]

        img_url = f"{HF_BASE_URL}/{img_rel}"
        mask_url = f"{HF_BASE_URL}/{mask_rel}"

        ext = img_rel.split(".")[-1]
        local_img_path = PUBLIC_IMAGES_DIR / f"{file_id}.{ext}"
        local_mask_path = PUBLIC_MASKS_DIR / f"{file_id}.png"

        try:
            if not local_img_path.exists():
                img_bytes = download_file(img_url)
                with open(local_img_path, "wb") as f:
                    f.write(img_bytes)

            if not local_mask_path.exists():
                mask_bytes = download_file(mask_url)
                with open(local_mask_path, "wb") as f:
                    f.write(mask_bytes)

            with Image.open(local_img_path) as im:
                w, h = im.size
            
            mask_arr = cv2.imread(str(local_mask_path), cv2.IMREAD_GRAYSCALE)
            if mask_arr is None or mask_arr.shape != (h, w):
                continue

            tampered_pixels = int(np.sum(mask_arr > 0))
            area_ratio = float(tampered_pixels / (w * h))

            downloaded_records.append({
                "sample_id": f"public_realtext_{file_id}",
                "document_id": file_id,
                "document_family_id": f"public_family_{file_id}",
                "doc_type": "public_document",
                "is_tampered": 1,
                "tampering_type": "public_realtext_tampered",
                "image_path": str(local_img_path),
                "mask_path": str(local_mask_path),
                "width": w,
                "height": h,
                "tampered_pixel_count": tampered_pixels,
                "tampered_area_ratio": area_ratio,
                "data_source": "public_realtext_v2",
                "is_synthetic": 0,
                "license": DATASET_PROVENANCE["license"],
            })
        except Exception as e:
            print(f"[ERROR] Failed downloading {file_id}: {e}")

    # 2. Download Authentic Documents
    print(f"[*] Downloading {len(authentic_keys)} authentic document images...")
    for file_id in tqdm(authentic_keys, desc="Authentic RealText-V2"):
        img_rel = image_files[file_id]
        img_url = f"{HF_BASE_URL}/{img_rel}"

        ext = img_rel.split(".")[-1]
        local_img_path = PUBLIC_IMAGES_DIR / f"{file_id}.{ext}"
        local_mask_path = PUBLIC_MASKS_DIR / f"{file_id}.png"

        try:
            if not local_img_path.exists():
                img_bytes = download_file(img_url)
                with open(local_img_path, "wb") as f:
                    f.write(img_bytes)

            with Image.open(local_img_path) as im:
                w, h = im.size

            if not local_mask_path.exists():
                zero_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.imwrite(str(local_mask_path), zero_mask)

            downloaded_records.append({
                "sample_id": f"public_realtext_{file_id}",
                "document_id": file_id,
                "document_family_id": f"public_family_{file_id}",
                "doc_type": "public_document",
                "is_tampered": 0,
                "tampering_type": "none",
                "image_path": str(local_img_path),
                "mask_path": str(local_mask_path),
                "width": w,
                "height": h,
                "tampered_pixel_count": 0,
                "tampered_area_ratio": 0.0,
                "data_source": "public_realtext_v2",
                "is_synthetic": 0,
                "license": DATASET_PROVENANCE["license"],
            })
        except Exception as e:
            print(f"[ERROR] Failed downloading authentic {file_id}: {e}")

    print(f"\n[+] Successfully downloaded and verified {len(downloaded_records)} public document samples into {PUBLIC_RAW_DIR}")
    return downloaded_records


if __name__ == "__main__":
    download_realtext_v2_subset(num_tampered=30, num_authentic=20)
