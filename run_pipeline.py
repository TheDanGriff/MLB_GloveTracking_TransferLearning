#!/usr/bin/env python3
"""
Run the entire glove‑tracking transfer learning pipeline from end to end.

This script orchestrates the following steps:

1. Extract the manually downloaded dataset archive ``baseball_rubber_home_glove.zip``
   from the ``data/`` directory and create a YOLO dataset YAML with correct
   train/val/test splits.  If the archive contains a nested folder, the
   nested path is handled automatically.
2. Train a YOLO model using the provided baseline weights on the custom
   dataset, with configurable hyper‑parameters (epochs, batch size,
   learning rate, etc.).
3. Evaluate the baseline model and the fine‑tuned model on the same
   dataset and print key detection metrics (mAP@0.5, mAP@0.5:0.95,
   precision and recall).

Usage example:

```
python run_pipeline.py \
  --zip-path data/baseball_rubber_home_glove.zip \
  --weights models/glove_tracking_v4_YOLOv11.pt \
  --epochs 50 --batch 16 --imgsz 640 --name glove_ft_full
```

This script expects that ``baseball_rubber_home_glove.zip`` and the
pretrained weights have been manually downloaded and placed in the
``data/`` and ``models/`` directories, respectively.  It does not
attempt to download external resources.
"""

import argparse
import sys
from pathlib import Path
from ultralytics import YOLO

# Import helper functions from our own modules
from data.download_data import extract_zip, create_dataset_yaml
from scripts.evaluate import run_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full transfer learning pipeline")
    parser.add_argument('--zip-path', type=str, default='data/baseball_rubber_home_glove.zip',
                        help="Path to the manually downloaded dataset ZIP")
    parser.add_argument('--weights', type=str, default='models/glove_tracking_v4_YOLOv11.pt',
                        help="Path to the pretrained baseline weights")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--batch', type=int, default=16, help="Batch size for training")
    parser.add_argument('--imgsz', type=int, default=640, help="Image size for training and evaluation")
    parser.add_argument('--lr0', type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument('--lrf', type=float, default=0.1, help="Final learning rate fraction")
    parser.add_argument('--momentum', type=float, default=0.937, help="Optimizer momentum")
    parser.add_argument('--weight-decay', type=float, default=5e-4, help="Weight decay (L2 penalty)")
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam', 'AdamW'],
                        help="Optimizer to use")
    parser.add_argument('--freeze', type=int, default=0, help="Freeze first N layers during training")
    parser.add_argument('--patience', type=int, default=20, help="Early stopping patience")
    parser.add_argument('--device', type=str, default='0', help="Computation device")
    parser.add_argument('--name', type=str, default='glove_ft_pipeline', help="Run name for training outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[0]

    # Step 1: extract dataset and create YAML
    zip_path = root / args.zip_path
    if not zip_path.exists():
        print(f"Dataset zip not found at {zip_path}. Please download it and place it here.")
        sys.exit(1)
    dataset_dir = root / 'data' / 'baseball_rubber_home_glove'
    extract_zip(zip_path, dataset_dir)
    nested = dataset_dir / 'baseball_rubber_home_glove'
    if nested.exists() and nested.is_dir():
        dataset_dir = nested
    yaml_path = root / 'data' / 'baseball_rubber_home_glove.yaml'
    create_dataset_yaml(dataset_dir, yaml_path)

    # Step 2: train the model
    print("\n=== Training model ===")
    model = YOLO(str(args.weights))
    train_kwargs = {
        'data': str(yaml_path),
        'epochs': args.epochs,
        'batch': args.batch,
        'imgsz': args.imgsz,
        'lr0': args.lr0,
        'lrf': args.lrf,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'optimizer': args.optimizer,
        'freeze': args.freeze if args.freeze > 0 else None,
        'patience': args.patience,
        'device': args.device,
        'name': args.name,
        'val': True,
    }
    # Remove None values (e.g. freeze when 0)
    train_kwargs = {k: v for k, v in train_kwargs.items() if v is not None}
    print("Training configuration:")
    for k, v in train_kwargs.items():
        print(f"  {k}: {v}")
    results = model.train(**train_kwargs)
    # Determine fine‑tuned weights path
    ft_weights = Path('runs/train') / args.name / 'weights' / 'best.pt'
    if not ft_weights.exists():
        print(f"Fine‑tuned weights not found at {ft_weights}. Training may have failed.")
        sys.exit(1)

    # Step 3: evaluate baseline vs fine‑tuned model
    print("\n=== Evaluating models ===")
    baseline_metrics = run_evaluation(str(args.weights), str(yaml_path), args.imgsz, args.device)
    ft_metrics = run_evaluation(str(ft_weights), str(yaml_path), args.imgsz, args.device)

    print("\nResults:")
    hdr = f"{'Model':>12} {'mAP@0.5':>10} {'mAP@0.5:0.95':>12} {'Precision':>10} {'Recall':>10}"
    print(hdr)
    print('-' * len(hdr))
    base_line = f"{'Baseline':>12} {baseline_metrics['mAP50']:>10.3f} {baseline_metrics['mAP50_95']:>12.3f} {baseline_metrics['precision']:>10.3f} {baseline_metrics['recall']:>10.3f}"
    ft_line = f"{'Fine‑tuned':>12} {ft_metrics['mAP50']:>10.3f} {ft_metrics['mAP50_95']:>12.3f} {ft_metrics['precision']:>10.3f} {ft_metrics['recall']:>10.3f}"
    print(base_line)
    print(ft_line)


if __name__ == '__main__':
    main()
