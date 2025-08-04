#!/usr/bin/env python3
"""
Fine‑tune the glove‑tracking model on the `baseball_rubber_home_glove` dataset.

This script wraps Ultralytics YOLO’s high‑level API.  It loads a base model
from pre‑trained weights, then trains on a custom dataset described by a
YAML file.  Hyper‑parameters such as number of epochs, image size and batch
size can be passed on the command line.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

def parse_args() -> argparse.Namespace:
    """Parse command line arguments for training.

    This parser exposes a number of hyper‑parameters commonly tuned when fine‑tuning
    object detection models.  Sensible defaults are provided for ease of use,
    but users can override any of them to experiment with different
    configurations.  See the Ultralytics YOLO documentation for details on
    each option.
    """
    parser = argparse.ArgumentParser(description="Train glove‑tracking YOLO model")
    parser.add_argument('--data', type=str, required=True,
                        help="Path to dataset YAML (e.g. data/baseball_rubber_home_glove.yaml)")
    parser.add_argument('--weights', type=str, required=True,
                        help="Path to pre‑trained weights (e.g. models/glove_tracking_v4_YOLOv11.pt)")
    parser.add_argument('--epochs', type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument('--imgsz', type=int, default=640,
                        help="Input image size for training (pixels)")
    parser.add_argument('--batch', type=int, default=16,
                        help="Training batch size")
    parser.add_argument('--lr0', type=float, default=1e-4,
                        help="Initial learning rate (lr0) for the optimizer")
    parser.add_argument('--lrf', type=float, default=0.1,
                        help="Final learning rate fraction (scheduler end LR = lr0*lrf)")
    parser.add_argument('--momentum', type=float, default=0.937,
                        help="SGD/Adam momentum")
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help="Weight decay for optimizer (L2 penalty)")
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam', 'AdamW'],
                        help="Optimizer to use")
    parser.add_argument('--freeze', type=int, default=0,
                        help="Freeze first N layers during training (0=no freeze)")
    parser.add_argument('--patience', type=int, default=20,
                        help="Early stopping patience (number of epochs without improvement before stopping)")
    parser.add_argument('--device', type=str, default="0",
                        help="Computation device: GPU index or 'cpu'")
    parser.add_argument('--name', type=str, default="glove_ft",
                        help="Name of training run (for output folder)")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    # Load base model
    model = YOLO(args.weights)

    # Build training hyper‑parameter dictionary.  Only include keys when
    # explicitly set to avoid overriding Ultralytics defaults unintentionally.
    train_kwargs = {
        'data': args.data,
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'batch': args.batch,
        'lr0': args.lr0,
        'lrf': args.lrf,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'optimizer': args.optimizer,
        'device': args.device,
        'name': args.name,
        'patience': args.patience,
        'val': True,
    }

    # Apply layer freezing if requested
    if args.freeze > 0:
        # Freeze the first N layers; Ultralytics accepts a list of layer indices
        # or an integer specifying number of layers to freeze.  See docs.
        train_kwargs['freeze'] = args.freeze

    # Kick off training
    print("Starting training with configuration:\n" + "\n".join(f"  {k}: {v}" for k, v in train_kwargs.items()))
    results = model.train(**train_kwargs)
    # Summarize results
    best_map50 = results.metrics.get('best_map50', None)
    print("Training completed. Best model saved in runs/train/{}/weights/best.pt".format(args.name))
    if best_map50 is not None:
        print(f"Best mAP@0.5 achieved: {best_map50:.4f}")

if __name__ == '__main__':
    main()
