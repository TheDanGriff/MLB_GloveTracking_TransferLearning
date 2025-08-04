#!/usr/bin/env python3
"""
Download the baseball_rubber_home_glove dataset and the glove tracking
pretrained weights.  This module automates fetching the zip archive and
PyTorch weight file from the external links referenced in the BaseballCV
repository.  If the download fails because the hosting site returns a 403
response, the user can manually place the files in the appropriate
directories and re-run this script with `--extract` to unpack the dataset.
"""

import argparse
import os
import sys
import urllib.request
from pathlib import Path
import zipfile

# External URLs defined in BaseballCV pointer files
DATA_URL = "https://data.balldatalab.com/index.php/s/pLy7sZqqMdx3jj7/download/baseball_rubber_home_glove.zip"
WEIGHTS_URL = "https://data.balldatalab.com/index.php/s/BwwWJbSsesFSBDa/download/glove_tracking_v4_YOLOv11.pt"

def download_file(url: str, dest: Path) -> bool:
    """Download a file from a URL to the destination path.

    Returns True if the download succeeded, False otherwise.
    """
    try:
        print(f"Downloading {url} -> {dest} ...")
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest)
    except Exception as exc:
        print(f"Error downloading {url}: {exc}")
        return False
    return True

def extract_zip(zip_path: Path, out_dir: Path) -> None:
    """Extract a zip archive into a given directory."""
    print(f"Extracting {zip_path} to {out_dir}")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(out_dir)

def create_dataset_yaml(dataset_dir: Path, yaml_path: Path) -> None:
    """Create a dataset YAML file for Ultralytics YOLO.

    A BaseballCV dataset extracted from the ``baseball_rubber_home_glove``
    archive follows the structure:

    ``dataset_dir/``
        ``train/images``
        ``train/labels``
        ``val`` or ``valid``
        ``test`` (optional)

    This helper inspects the extracted directory to choose the correct
    validation folder name (``val`` or ``valid``).  It then writes
    a YAML configuration pointing YOLO to the appropriate sub‑directories.
    The ``path`` field points to the dataset root, and ``train``, ``val``
    and ``test`` fields are relative to this root.
    """

    # Determine which validation directory exists: ``val`` or ``valid``
    if (dataset_dir / "val").exists():
        val_dir = "val"
    elif (dataset_dir / "valid").exists():
        val_dir = "valid"
    else:
        # If no validation folder is present, reuse the train set for validation
        val_dir = "train"

    # Use 'test' if present, otherwise fall back to validation set
    if (dataset_dir / "test").exists():
        test_dir = "test"
    else:
        test_dir = val_dir

    yaml_lines = [
        f"path: {dataset_dir}",
        f"train: {Path('train') / 'images'}",
        f"val: {Path(val_dir) / 'images'}",
        f"test: {Path(test_dir) / 'images'}",
        "",
        "nc: 1",
        "names: ['glove']",
    ]
    yaml_text = "\n".join(yaml_lines) + "\n"
    yaml_path.write_text(yaml_text)
    print(f"Wrote dataset YAML to {yaml_path}")

def main(args: argparse.Namespace) -> None:
    root = Path(__file__).resolve().parents[1]  # repository root
    raw_dir = root / "data" / "raw"
    weights_dir = root / "models"
    # When manually downloading the zip, place it directly under ``data``.
    # We default to the file in ``data/raw`` for backward compatibility but also
    # check ``data/baseball_rubber_home_glove.zip`` when extracting.
    dataset_zip = raw_dir / "baseball_rubber_home_glove.zip"
    weights_file = weights_dir / "glove_tracking_v4_YOLOv11.pt"

    if args.download:
        # Download dataset zip
        if dataset_zip.exists():
            print(f"Dataset zip already exists at {dataset_zip}")
        else:
            ok = download_file(DATA_URL, dataset_zip)
            if not ok:
                print("\nUnable to download dataset. Please manually download the file from"
                      f" {DATA_URL} and place it at {dataset_zip}")

        # Download weights file
        if weights_file.exists():
            print(f"Weights file already exists at {weights_file}")
        else:
            ok = download_file(WEIGHTS_URL, weights_file)
            if not ok:
                print("\nUnable to download weights. Please manually download the file from"
                      f" {WEIGHTS_URL} and place it at {weights_file}")

    if args.extract:
        # Extract dataset
        # If the zip is not found in the raw directory, fall back to the data directory
        if not dataset_zip.exists():
            alt_zip = root / "data" / "baseball_rubber_home_glove.zip"
            if alt_zip.exists():
                dataset_zip = alt_zip
            else:
                print(f"Dataset zip does not exist at {dataset_zip} or {alt_zip}. Download it first.")
                sys.exit(1)
        dataset_dir = root / "data" / "baseball_rubber_home_glove"
        extract_zip(dataset_zip, dataset_dir)
        # The archive contains a nested folder with the same name; if so, descend
        nested = dataset_dir / "baseball_rubber_home_glove"
        if nested.exists() and nested.is_dir():
            dataset_dir = nested
        # Create YAML file for YOLO with proper train/val/test splits
        yaml_path = root / "data" / "baseball_rubber_home_glove.yaml"
        create_dataset_yaml(dataset_dir, yaml_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download and prepare data for glove tracking")
    parser.add_argument('--download', action='store_true', help="Download dataset and weights")
    parser.add_argument('--extract', action='store_true', help="Extract dataset and create YAML")
    args = parser.parse_args()
    if not args.download and not args.extract:
        # default to download only
        args.download = True
    main(args)
