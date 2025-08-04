#!/usr/bin/env python3
"""
Evaluate baseline and fine‑tuned glove‑tracking models on a given dataset.

This script computes detection metrics using Ultralytics YOLO’s built‑in
`val()` method.  It accepts paths to the dataset YAML, the baseline
weights (pre‑trained model) and the fine‑tuned weights.  Metrics for both
models are printed to stdout for easy comparison.
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

def parse_args() -> argparse.Namespace:
    """Parse command line arguments for evaluation.

    In addition to specifying the dataset and weight files, users can choose to save
    a JSON summary of the metrics and optionally plot a confusion matrix.  The
    defaults perform evaluation on GPU 0 with 640×640 images.
    """
    parser = argparse.ArgumentParser(description="Evaluate baseline and fine‑tuned models")
    parser.add_argument('--data', type=str, required=True,
                        help="Path to dataset YAML")
    parser.add_argument('--weights', type=str, required=True,
                        help="Baseline weights file")
    parser.add_argument('--fine_tuned', type=str, required=True,
                        help="Fine‑tuned weights file (e.g. runs/train/exp/weights/best.pt)")
    parser.add_argument('--imgsz', type=int, default=640, help="Image size for evaluation")
    parser.add_argument('--device', type=str, default="0", help="Computation device")
    parser.add_argument('--save-json', type=str, default=None,
                        help="Optional path to save metrics as JSON")
    parser.add_argument('--plot', action='store_true',
                        help="If set, save a confusion matrix plot for the fine‑tuned model")
    return parser.parse_args()

def run_evaluation(model_path: str, data: str, imgsz: int, device: str, save_dir: str | None = None) -> dict:
    """Run validation on a model and return key metrics.

    If ``save_dir`` is provided, the underlying Ultralytics validation routine will
    save per‑image predictions and optional plots in that directory.  Otherwise
    artifacts are not persisted.
    """
    model = YOLO(model_path)
    metrics = model.val(data=data, imgsz=imgsz, device=device, save=bool(save_dir), save_dir=save_dir)
    return {
        'mAP50': float(metrics.box.map50),
        'mAP50_95': float(metrics.box.map),
        'precision': float(metrics.box.p),
        'recall': float(metrics.box.r)
    }

def main() -> None:
    args = parse_args()
    print("Evaluating baseline model...")
    baseline_metrics = run_evaluation(args.weights, args.data, args.imgsz, args.device)
    print("Evaluating fine‑tuned model...")
    # When plotting we save outputs to a temporary directory under runs/val for the fine‑tuned model
    save_dir = None
    if args.plot:
        save_dir = f"runs/val/{Path(args.fine_tuned).stem}"  # unique dir based on weights name
    ft_metrics = run_evaluation(args.fine_tuned, args.data, args.imgsz, args.device, save_dir)

    print("\nResults:")
    hdr = "{:>12} {:>10} {:>12} {:>10} {:>10}".format('Model','mAP@0.5','mAP@0.5:0.95','Precision','Recall')
    print(hdr)
    print("-"*len(hdr))
    base_line = "{:>12} {:>10.3f} {:>12.3f} {:>10.3f} {:>10.3f}".format(
        'Baseline', baseline_metrics['mAP50'], baseline_metrics['mAP50_95'],
        baseline_metrics['precision'], baseline_metrics['recall']
    )
    ft_line = "{:>12} {:>10.3f} {:>12.3f} {:>10.3f} {:>10.3f}".format(
        'Fine‑tuned', ft_metrics['mAP50'], ft_metrics['mAP50_95'],
        ft_metrics['precision'], ft_metrics['recall']
    )
    print(base_line)
    print(ft_line)

    # Save metrics to JSON if requested
    if args.save_json:
        import json
        summary = {
            'baseline': baseline_metrics,
            'fine_tuned': ft_metrics
        }
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open('w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved metrics to {out_path}")

if __name__ == '__main__':
    main()
