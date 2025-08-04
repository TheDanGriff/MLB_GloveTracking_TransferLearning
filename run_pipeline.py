#!/usr/bin/env python3
"""
End-to-end glove-tracking transfer learning pipeline.

It will:
1) Use the provided ZIP (if present) OR an already-extracted dataset folder OR an existing YAML.
2) Auto-detect nested dataset folder structure (…/baseball_rubber_home_glove/baseball_rubber_home_glove/…).
3) Write a correct YOLO YAML (train/val/test) if missing.
4) Train an advanced fine-tune configuration (AdamW, cosine LR, warmup, EMA, richer aug).
5) Evaluate baseline vs fine-tuned on the same dataset and print a concise comparison.

Examples (Windows CMD/Powershell):
  python run_pipeline.py --weights models/glove_tracking_v4_YOLOv11.pt --epochs 50 --batch 16 --imgsz 640 --name glove_ft_pipeline
  python run_pipeline.py --zip-path data/baseball_rubber_home_glove.zip --weights models/glove_tracking_v4_YOLOv11.pt
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

from ultralytics import YOLO
import yaml
import zipfile
import shutil


# -----------------------------
# Utilities
# -----------------------------
def extract_zip(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)

def find_dataset_root(base_dir: Path) -> Path:
    """
    Return the dataset root that contains 'train', 'val/valid', 'test' subfolders.
    Handles nested e.g. base/base/train by descending one level if needed.
    """
    def has_splits(d: Path) -> bool:
        return (d / "train").exists() and ((d / "val").exists() or (d / "valid").exists() or (d / "validation").exists())

    cand = base_dir
    if has_splits(cand):
        return cand

    # Common nested case: data/baseball_rubber_home_glove/baseball_rubber_home_glove/...
    nested = cand / "baseball_rubber_home_glove"
    if nested.exists() and has_splits(nested):
        return nested

    # Search one more level deep (defensive)
    for p in cand.iterdir():
        if p.is_dir() and has_splits(p):
            return p

    # If nothing found, still return base_dir (YAML build will try sensible defaults)
    return base_dir

def make_yolo_yaml(dataset_root: Path, yaml_path: Path) -> None:
    """
    Create a YOLO dataset YAML that points to /images for train/val/test if they exist.
    """
    def split_dir(name: str) -> Optional[Path]:
        for key in [name, "valid", "validation", "val"]:
            d = dataset_root / key
            if d.exists():
                return d
        return None

    train_dir = split_dir("train")
    val_dir = split_dir("val")
    test_dir = split_dir("test")  # may be None

    def images_subdir(d: Optional[Path]) -> Optional[str]:
        if d is None:
            return None
        imgs = d / "images"
        if imgs.exists():
            return imgs.as_posix()
        # Fallback: allow images directly in split folder (rare)
        return d.as_posix()

    data = {
        "path": dataset_root.as_posix(),
        "train": images_subdir(train_dir),
        "val": images_subdir(val_dir),
        "test": images_subdir(test_dir),
        "nc": 1,
        "names": ["glove"],
    }
    # If val missing, reuse train as val (not ideal but prevents crashes)
    if data["val"] is None:
        data["val"] = data["train"]
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)

def load_metrics_from_results(res) -> dict:
    """Extract key metrics from an Ultralytics results object."""
    # Ultralytics YOLO v8 returns a Results object with .metrics dict-like
    m = getattr(res, "results_dict", None) or getattr(res, "metrics", None) or {}
    # Normalize keys we care about
    out = {
        "map50": float(m.get("metrics/mAP50(B)", m.get("mAP50", 0.0))),     # mAP@0.5
        "map50_95": float(m.get("metrics/mAP50-95(B)", m.get("mAP50-95", 0.0))),  # mAP@0.5:0.95
        "precision": float(m.get("metrics/precision(B)", m.get("precision", 0.0))),
        "recall": float(m.get("metrics/recall(B)", m.get("recall", 0.0))),
    }
    return out


# -----------------------------
# Main
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Glove-tracking transfer learning pipeline")
    p.add_argument("--zip-path", type=str, default=None, help="Optional path to dataset ZIP in data/")
    p.add_argument("--dataset-dir", type=str, default="data/baseball_rubber_home_glove", help="Dataset directory (already extracted)")
    p.add_argument("--yaml", type=str, default="data/baseball_rubber_home_glove.yaml", help="Dataset YAML path (auto-written if missing)")
    p.add_argument("--weights", type=str, required=True, help="Pretrained weights path for baseline AND fine-tuning start")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--name", type=str, default="glove_ft_advanced")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()

def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent

    # ---------------- Step 1: Prepare dataset ----------------
    dataset_dir = (root / args.dataset_dir).resolve()
    yaml_path = (root / args.yaml).resolve()

    # If ZIP provided and exists -> extract
    if args.zip_path:
        zip_path = (root / args.zip_path).resolve()
        if zip_path.exists():
            print(f"Extracting {zip_path} -> {dataset_dir} ...")
            if dataset_dir.exists() and any(dataset_dir.iterdir()):
                # Clean previous extraction to avoid stale nested paths
                shutil.rmtree(dataset_dir)
            extract_zip(zip_path, dataset_dir)
        else:
            print(f"[INFO] ZIP not found at {zip_path}, will use existing dataset folder or YAML.")

    # If no dataset folder yet, but YAML exists, we’ll rely on YAML; otherwise try to find split root.
    if not dataset_dir.exists():
        print(f"[INFO] Dataset directory {dataset_dir} not found. If {yaml_path} exists, it will be used as-is.")
    else:
        ds_root = find_dataset_root(dataset_dir)
        # Write YAML if missing or stale
        if not yaml_path.exists():
            print(f"Writing dataset YAML -> {yaml_path}")
            make_yolo_yaml(ds_root, yaml_path)
        else:
            print(f"Using existing YAML: {yaml_path}")

    if not yaml_path.exists():
        print(f"ERROR: No YAML found at {yaml_path} and no dataset directory at {dataset_dir}.")
        print("       Place your extracted dataset under data/baseball_rubber_home_glove (with train/val/test) or supply --zip-path.")
        sys.exit(1)

    # ---------------- Step 2: Evaluate baseline ----------------
    baseline_weights = (root / args.weights).resolve()
    if not baseline_weights.exists():
        print(f"ERROR: Baseline/starting weights not found at {baseline_weights}")
        sys.exit(1)

    print("\n=== Baseline evaluation ===")
    baseline_model = YOLO(baseline_weights.as_posix())
    base_res = baseline_model.val(
        data=yaml_path.as_posix(),
        imgsz=args.imgsz,
        device=args.device,
        split="val",
        verbose=True,
    )
    base_metrics = load_metrics_from_results(base_res)
    print("[Baseline]", json.dumps(base_metrics, indent=2))

    # ---------------- Step 3: Fine-tune (advanced recipe) ----------------
    print("\n=== Fine-tuning (transfer learning) ===")
    model = YOLO(baseline_weights.as_posix())

    # Advanced training settings; all recognized by Ultralytics YOLOv8
    # (AdamW, cosine LR, warmup, EMA, stronger aug, label smoothing)
    train_kwargs = dict(
        data=yaml_path.as_posix(),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        seed=args.seed,

        optimizer="AdamW",
        lr0=3e-4,         # base LR
        lrf=0.1,          # final LR = lr0 * lrf
        cos_lr=True,      # cosine LR schedule
        warmup_epochs=3,
        momentum=0.9,
        weight_decay=0.01,

        amp=True,
        ema=True,

        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=3.0, translate=0.05, scale=0.2, shear=2.0,
        perspective=0.0005, flipud=0.0, fliplr=0.5,
        mosaic=0.8, mixup=0.1, copy_paste=0.1,
        erasing=0.1, # cutout-like

        box=7.5, cls=0.5, dfl=1.5,
        label_smoothing=0.005,

        patience=max(10, args.epochs // 5),  # early stop patience
        close_mosaic=5,
        workers=4,
        save=True,
        plots=True,
        verbose=True,
    )

    model.train(**train_kwargs)

    best_ft = root / "runs" / "detect" / args.name / "weights" / "best.pt"
    if not best_ft.exists():
        # fallback to generic path (Ultralytics may create 'runs/train')
        best_ft = root / "runs" / "train" / args.name / "weights" / "best.pt"
    if not best_ft.exists():
        print(f"ERROR: Fine-tuned weights not found under runs/, name={args.name}. Check training logs.")
        sys.exit(1)

    # ---------------- Step 4: Evaluate fine-tuned ----------------
    print("\n=== Fine-tuned evaluation ===")
    ft_model = YOLO(best_ft.as_posix())
    ft_res = ft_model.val(
        data=yaml_path.as_posix(),
        imgsz=args.imgsz,
        device=args.device,
        split="val",
        verbose=True,
    )
    ft_metrics = load_metrics_from_results(ft_res)
    print("[Fine-tuned]", json.dumps(ft_metrics, indent=2))

    # ---------------- Step 5: Comparison summary ----------------
    print("\n=== Summary (Baseline vs Fine-tuned) ===")
    summary = {
        "baseline": base_metrics,
        "fine_tuned": ft_metrics,
        "runs_dir": str(best_ft.parent.parent.parent),  # .../runs/...
        "best_weights": best_ft.as_posix(),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
