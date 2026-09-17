"""DocTamper Dataset Download and Preparation Assistant.

Guides downloading and setting up the DocTamper dataset from Kaggle or local archives:
https://www.kaggle.com/code/datascienceimcomming/doctamper-dataset-overview/input
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DocTamperSetup")

DEFAULT_TARGET_DIR = Path("data/raw/doctamper")


def check_kaggle_auth() -> bool:
    """Check if Kaggle API authentication credentials exist."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    env_token = os.environ.get("KAGGLE_API_TOKEN") or (
        os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
    )
    return kaggle_json.exists() or bool(env_token)


def download_from_kaggle(
    dataset_handle: str = "datascienceimcomming/doctamper-dataset-overview",
    target_dir: Path = DEFAULT_TARGET_DIR,
) -> bool:
    """Attempt download of dataset using the official Kaggle CLI."""
    target_dir.mkdir(parents=True, exist_ok=True)

    if not check_kaggle_auth():
        logger.warning("Kaggle credentials not detected.")
        logger.info(
            "To enable automatic download from Kaggle:\n"
            "  1. Go to https://www.kaggle.com/settings/api\n"
            "  2. Click 'Create New Token' to download kaggle.json\n"
            "  3. Place kaggle.json in ~/.kaggle/kaggle.json (or set KAGGLE_USERNAME and KAGGLE_KEY env vars)\n"
        )
        return False

    try:
        import kaggle
        logger.info("Authenticating with Kaggle API...")
        kaggle.api.authenticate()

        logger.info("Downloading dataset '%s' into %s...", dataset_handle, target_dir)
        # Try dataset download
        kaggle.api.dataset_download_files(
            dataset_handle,
            path=str(target_dir),
            unzip=True,
            quiet=False,
        )
        logger.info("Dataset successfully downloaded and extracted to: %s", target_dir)
        return True
    except Exception as e:
        logger.error("Kaggle download failed: %s", e)
        return False


def inspect_doctamper_dir(target_dir: Path) -> dict:
    """Inspect the target directory to verify LMDB or image/mask pairs."""
    status = {
        "exists": target_dir.exists(),
        "is_lmdb": False,
        "lmdb_files": [],
        "image_count": 0,
        "mask_count": 0,
        "valid": False,
    }

    if not target_dir.exists():
        return status

    # Check for LMDB files
    mdb_files = list(target_dir.rglob("*.mdb"))
    if mdb_files:
        status["is_lmdb"] = True
        status["lmdb_files"] = [str(f) for f in mdb_files]
        status["valid"] = True
        logger.info("Found %d LMDB file(s) in %s", len(mdb_files), target_dir)
        return status

    # Check for image/mask directories
    images = list(target_dir.rglob("*.jpg")) + list(target_dir.rglob("*.png"))
    status["image_count"] = len(images)
    if len(images) > 0:
        status["valid"] = True
        logger.info("Found %d image files in %s", len(images), target_dir)

    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="DocTamper Dataset Setup Helper")
    parser.add_argument(
        "--dataset",
        type=str,
        default="datascienceimcomming/doctamper-dataset-overview",
        help="Kaggle dataset handle or URL",
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        default="data/raw/doctamper",
        help="Destination directory",
    )
    parser.add_argument(
        "--inspect_only",
        action="store_true",
        help="Only inspect current local directory without downloading",
    )

    args = parser.parse_args()
    target_path = Path(args.target_dir)

    info = inspect_doctamper_dir(target_path)
    if info["valid"]:
        logger.info("DocTamper dataset is already present and valid at %s!", target_path)
        return

    if not args.inspect_only:
        success = download_from_kaggle(args.dataset, target_path)
        if not success:
            logger.info(
                "\n--- MANUAL SETUP INSTRUCTIONS ---\n"
                "1. Download the DocTamper dataset / files from:\n"
                "   https://www.kaggle.com/code/datascienceimcomming/doctamper-dataset-overview/input\n"
                "2. Place the downloaded files or LMDB folder into:\n"
                f"   {target_path.resolve()}\n"
                "3. Run training with:\n"
                f"   python src/training/train_doctamper.py --data_path {target_path}\n"
            )


if __name__ == "__main__":
    main()
