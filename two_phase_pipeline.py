from __future__ import annotations
import argparse, json, os, sys, shutil, time, random
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import yaml, torch
from ultralytics import YOLO

try:
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False

def _has_splits(d: Path) -> bool:
    return (d / "train").exists() and ((d / "val").exists() or (d / "valid").exists() or (d / "test").exists())

def find_dataset_root(base_dir: Path) -> Path:
    if _has_splits(base_dir):
        return base_dir
    nested = base_dir / "baseball_rubber_home_glove"
    if nested.exists() and _has_splits(nested):
        return nested
    for p in base_dir.glob("*"):
        if p.is_dir() and _has_splits(p):
            return p
    return base_dir

def _split_dir(dataset_root: Path, name: str) -> Optional[Path]:
    for key in [name, "val", "valid", "validation"]:
        d = dataset_root / key
        if d.exists():
            return d
    return None

def _image_files_of(split_dir: Optional[Path]) -> List[Path]:
    if not split_dir: return []
    imgs = split_dir / "images"
    if not imgs.exists(): return []
    return sorted([p for p in imgs.rglob("*.*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])

def _label_files_of(split_dir: Optional[Path]) -> List[Path]:
    if not split_dir: return []
    labels = split_dir / "labels"
    if not labels.exists(): return []
    return sorted([p for p in labels.rglob("*.txt") if p.is_file()])

def erase_caches(root_dir: Path) -> None:
    for p in root_dir.rglob("*.cache"):
        try: p.unlink()
        except Exception: pass

# quick stats 
def label_stats(dataset_root: Path) -> Dict[str, Dict[str, int]]:
    stats: Dict[str, Dict[str, int]] = {}
    for sp in ["train", "val", "test"]:
        d = _split_dir(dataset_root, sp)
        imgs = _image_files_of(d)
        labs = _label_files_of(d)
        n_boxes = 0
        for lf in labs:
            try:
                lines = [ln for ln in lf.read_text(encoding="utf-8").splitlines() if ln.strip()]
            except Exception:
                lines = []
            n_boxes += len(lines)
        stats[sp] = {"images": len(imgs), "label_files": len(labs), "boxes": n_boxes}
    return stats

def write_multiclass_yaml(dataset_root: Path, yaml_out: Path, nc: int = 5, names: Optional[List[str]] = None) -> Path:
    def img_dir(name: str) -> Optional[str]:
        d = _split_dir(dataset_root, name)
        if not d: return None
        imgs = d / "images"
        return imgs.as_posix() if imgs.exists() else None
    if names is None or len(names) != nc:
        names = [f"class_{i}" for i in range(nc)]
        names[0] = "glove"
    data = {"path": dataset_root.as_posix(), "train": img_dir("train"),
            "val": img_dir("val"), "test": img_dir("test"), "nc": nc, "names": names}
    yaml_out.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_out, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return yaml_out

def write_glove_yaml(glove_root: Path, yaml_out: Path) -> Path:
    def img_dir(name: str) -> Optional[str]:
        d = glove_root / name / "images"
        return d.as_posix() if d.exists() else None
    data = {"path": glove_root.as_posix(), "train": img_dir("train"),
            "val": img_dir("val") or img_dir("train"), "test": img_dir("test"),
            "nc": 1, "names": ["glove"]}
    yaml_out.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_out, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return yaml_out

# glove-only dataset
def detect_glove_index_from_model(weights_path: Path) -> int:
    try:
        m = YOLO(weights_path.as_posix())
        names = getattr(m, "names", None)
        if isinstance(names, dict):
            for k, v in names.items():
                if isinstance(v, str) and "glove" in v.lower():
                    return int(k)
        elif isinstance(names, list):
            for i, v in enumerate(names):
                if isinstance(v, str) and "glove" in v.lower():
                    return i
    except Exception:
        pass
    return 0

def make_glove_only_dataset(src_root: Path, out_root: Path, glove_idx: Optional[int]) -> Path:
    if out_root.exists(): shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    def process_split(split_name: str) -> None:
        src_split = _split_dir(src_root, split_name)
        if not src_split: return
        dst_split = out_root / split_name
        (dst_split / "images").mkdir(parents=True, exist_ok=True)
        (dst_split / "labels").mkdir(parents=True, exist_ok=True)
        # copy images
        for img in _image_files_of(src_split):
            rel = img.relative_to(src_split / "images")
            dst = (dst_split / "images" / rel).with_suffix(img.suffix)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, dst)
        # filter labels to glove only
        for lab in _label_files_of(src_split):
            rel = lab.relative_to(src_split / "labels")
            dst_lab = (dst_split / "labels" / rel).with_suffix(".txt")
            dst_lab.parent.mkdir(parents=True, exist_ok=True)
            try:
                lines_in = lab.read_text(encoding="utf-8").splitlines()
            except Exception:
                lines_in = []
            keep: List[str] = []
            for line in lines_in:
                line = line.strip()
                if not line: continue
                parts = line.split()
                try:
                    cls = int(float(parts[0]))
                except Exception:
                    continue
                if glove_idx is None or cls == glove_idx:
                    parts[0] = "0"
                    keep.append(" ".join(parts))
            if keep:
                dst_lab.write_text("\n".join(keep) + "\n", encoding="utf-8")
    for split in ["train", "val", "test"]:
        process_split(split)
    erase_caches(out_root)
    return out_root

def load_metrics(res) -> Dict[str, float]:
    m = getattr(res, "results_dict", None) or getattr(res, "metrics", None) or {}
    return {
        "map50": float(m.get("metrics/mAP50(B)", m.get("mAP50", 0.0))),
        "map50_95": float(m.get("metrics/mAP50-95(B)", m.get("mAP50-95", 0.0))),
        "precision": float(m.get("metrics/precision(B)", m.get("precision", 0.0))),
        "recall": float(m.get("metrics/recall(B)", m.get("recall", 0.0))),
    }

def names_from_weights(weights_path: Path) -> Optional[List[str]]:
    try:
        mtmp = YOLO(weights_path.as_posix())
        nm = getattr(mtmp, "names", None)
        if isinstance(nm, dict): return [nm[i] for i in sorted(nm.keys())]
        elif isinstance(nm, list): return nm
    except Exception:
        pass
    return None

def sanity_read_random_images(ds_root: Path, n: int = 32, timeout_s: float = 15.0) -> Tuple[int,int]:
    if not _HAS_PIL:
        print("[SANITY] PIL not available, skipping.")
        return (0,0)
    train_dir = _split_dir(ds_root, "train")
    if not train_dir: return (0,0)
    imgs = _image_files_of(train_dir)
    if not imgs: return (0,0)
    sample = random.sample(imgs, min(n, len(imgs)))
    ok, fail = 0, 0
    t0 = time.time()
    print(f"[SANITY] PIL open() check on {len(sample)} train images ...")
    for i, p in enumerate(sample, 1):
        if time.time() - t0 > timeout_s:
            print("[SANITY] Timed out; moving on.")
            break
        try:
            with Image.open(p) as im:
                im.verify() 
            ok += 1
        except Exception:
            fail += 1
        if i % 8 == 0:
            print(f"[SANITY] ... {i}/{len(sample)} checked")
    print(f"[SANITY] Done. ok={ok}, fail={fail}")
    return ok, fail

# CLI 
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Two-phase pipeline (stable, glove-first)")
    p.add_argument("--dataset-dir", type=str, default="data/baseball_rubber_home_glove")
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--glove-class-index", type=int, default=None)

    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=12)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--cache", type=str, default="none", choices=["none", "ram", "disk"])
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--name", type=str, default="glove_two_phase")

    p.add_argument("--skip-phase1", type=int, default=1, help="1=skip P1 (recommended), 0=run P1")
    p.add_argument("--phase1-epochs", type=int, default=6)
    p.add_argument("--phase2-epochs", type=int, default=20)
    p.add_argument("--fraction", type=float, default=1.0)

    p.add_argument("--tta", action="store_true")
    p.add_argument("--rect", action="store_true", default=False)
    return p.parse_args()

# main 
def main() -> None:
    root = Path(__file__).resolve().parent
    args = parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    if args.device is None: args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device = {args.device}")
    cache_mode = False if args.cache == "none" else args.cache
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    dataset_dir = (root / args.dataset_dir.replace("/", os.sep)).resolve()
    if not dataset_dir.exists():
        print(f"[ERROR] dataset dir not found: {dataset_dir}"); sys.exit(1)
    ds_root = find_dataset_root(dataset_dir)

    weights_path = (root / args.weights).resolve()
    if not weights_path.exists():
        print(f"[ERROR] weights not found: {weights_path}"); sys.exit(1)

    model_names = names_from_weights(weights_path)
    erase_caches(ds_root)

    fixed_yaml = (root / "data" / "baseball_rubber_home_glove_fixed.yaml").resolve()
    fixed_yaml.parent.mkdir(parents=True, exist_ok=True)
    write_multiclass_yaml(ds_root, fixed_yaml, nc=5, names=model_names)

    glove_idx = args.glove_class_index or detect_glove_index_from_model(weights_path)
    print(f"[INFO] glove_idx = {glove_idx}")
    glove_root = make_glove_only_dataset(ds_root, root / "data" / "glove_only", glove_idx=glove_idx)
    glove_yaml = write_glove_yaml(glove_root, root / "data" / "glove_only.yaml")

    # baseline
    print("\n=== Baseline evaluation (glove-only) ===")
    baseline_model = YOLO(weights_path.as_posix())
    val_half = args.device != "cpu"
    baseline_res = baseline_model.val(
        data=glove_yaml.as_posix(), imgsz=args.imgsz, device=args.device,
        split="val", verbose=True, half=val_half, augment=args.tta, rect=False,
    )
    baseline = load_metrics(baseline_res)
    print("[Baseline]", json.dumps(baseline, indent=2))

    sanity_read_random_images(ds_root, n=32, timeout_s=15.0)

    # Phase 1 
    if not args.skip_phase1:
        print("\n=== Phase 1: Multi-class (nc=5) fine-tune ===")
        model_p1 = YOLO(weights_path.as_posix())
        hp1_warm = dict(
            imgsz=args.imgsz, epochs=1, batch=args.batch, workers=args.workers,
            optimizer="AdamW", lr0=3e-4, lrf=0.2, warmup_epochs=1, momentum=0.9, weight_decay=0.01,
            amp=True, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
            degrees=0.5, translate=0.02, scale=0.10, shear=0.5,
            perspective=0.0002, flipud=0.0, fliplr=0.5,
            mosaic=0.0, mixup=0.0, copy_paste=0.0, erasing=0.15,
            box=7.5, cls=0.5, dfl=1.5,
            patience=max(2, args.patience),
            close_mosaic=0, save=True, plots=False, val=True,
            fraction=args.fraction, rect=False, cache=False, cos_lr=True,
            name=f"{args.name}_p1_warm", seed=0,
        )
        model_p1.train(data=fixed_yaml.as_posix(), device=args.device, **hp1_warm)

        hp1 = dict(
            imgsz=args.imgsz, epochs=args.phase1_epochs, batch=args.batch, workers=args.workers,
            optimizer="AdamW", lr0=3e-4, lrf=0.2, warmup_epochs=2, momentum=0.9, weight_decay=0.01,
            amp=True, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
            degrees=2.0, translate=0.05, scale=0.20, shear=1.0,
            perspective=0.0005, flipud=0.0, fliplr=0.5,
            mosaic=0.2, mixup=0.05, copy_paste=0.05, erasing=0.25,
            box=7.5, cls=0.5, dfl=1.5,
            patience=max(4, args.patience),
            close_mosaic=max(2, args.phase1_epochs // 2),
            save=True, plots=False, val=True,
            fraction=args.fraction, rect=False, cache=cache_mode, cos_lr=True,
            name=f"{args.name}_p1", seed=0,
        )
        model_p1.train(data=fixed_yaml.as_posix(), device=args.device, **hp1)

        p1_best = (root / "runs" / "detect" / f"{args.name}_p1" / "weights" / "best.pt")
        if not p1_best.exists():
            p1_best = (root / "runs" / "train" / f"{args.name}_p1" / "weights" / "best.pt")
        if not p1_best.exists():
            print(f"[ERROR] Phase-1 best weights not found for name={args.name}_p1"); sys.exit(1)

        print("\n=== Phase 1 evaluation (glove-only val) ===")
        p1_model = YOLO(p1_best.as_posix())
        p1_res = p1_model.val(
            data=glove_yaml.as_posix(), imgsz=args.imgsz, device=args.device,
            split="val", verbose=True, half=val_half, augment=args.tta, rect=False,
        )
        phase1 = load_metrics(p1_res)
        print("[Phase 1]", json.dumps(phase1, indent=2))
        p2_start = p1_best
    else:
        p2_start = weights_path

    # Phase 2
    print("\n=== Phase 2: Glove-only (nc=1) fine-tune ===")
    model_p2 = YOLO(p2_start.as_posix())

    hp2_warm = dict(
        imgsz=args.imgsz, epochs=1, batch=args.batch, workers=args.workers,
        optimizer="AdamW", lr0=2e-4, lrf=0.2, warmup_epochs=1, momentum=0.9, weight_decay=0.01,
        amp=True, hsv_h=0.012, hsv_s=0.6, hsv_v=0.35,
        degrees=0.5, translate=0.02, scale=0.10, shear=0.5,
        perspective=0.0002, flipud=0.0, fliplr=0.5,
        mosaic=0.0, mixup=0.0, copy_paste=0.0, erasing=0.15,
        box=7.5, cls=0.5, dfl=1.5,
        patience=max(2, args.patience),
        close_mosaic=0, save=True, plots=False, val=True,
        fraction=args.fraction, rect=False, cache=False, cos_lr=True,
        name=f"{args.name}_p2_warm", seed=0,
    )
    model_p2.train(data=glove_yaml.as_posix(), device=args.device, **hp2_warm)

    hp2 = dict(
        imgsz=args.imgsz, epochs=args.phase2_epochs, batch=args.batch, workers=args.workers,
        optimizer="AdamW", lr0=2e-4, lrf=0.2, warmup_epochs=2, momentum=0.9, weight_decay=0.01,
        amp=True, hsv_h=0.012, hsv_s=0.6, hsv_v=0.35,
        degrees=2.0, translate=0.05, scale=0.15, shear=1.5,
        perspective=0.0005, flipud=0.0, fliplr=0.5,
        mosaic=0.3, mixup=0.05, copy_paste=0.05, erasing=0.30,
        box=7.5, cls=0.5, dfl=1.5,
        patience=max(5, args.patience),
        close_mosaic=max(2, args.phase2_epochs // 2),
        save=True, plots=False, val=True,
        fraction=args.fraction, rect=False, cache=cache_mode, cos_lr=True,
        name=f"{args.name}_p2", seed=0,
    )
    model_p2.train(data=glove_yaml.as_posix(), device=args.device, **hp2)

    p2_best = (root / "runs" / "detect" / f"{args.name}_p2" / "weights" / "best.pt")
    if not p2_best.exists():
        p2_best = (root / "runs" / "train" / f"{args.name}_p2" / "weights" / "best.pt")
    if not p2_best.exists():
        print(f"[ERROR] Phase-2 best weights not found for name={args.name}_p2"); sys.exit(1)

    print("\n=== Phase 2 evaluation (glove-only val) ===")
    p2_model = YOLO(p2_best.as_posix())
    p2_res = p2_model.val(
        data=glove_yaml.as_posix(), imgsz=args.imgsz, device=args.device,
        split="val", verbose=True, half=val_half, augment=args.tta, rect=False,
    )
    phase2 = load_metrics(p2_res)
    print("[Phase 2]", json.dumps(phase2, indent=2))

    print("\n=== Summary ===")
    out = {
        "dataset_stats": label_stats(ds_root),
        "baseline": baseline,
        "phase2": phase2,
        "phase2_best": str(p2_best.resolve()),
        "device": args.device, "imgsz": args.imgsz, "batch": args.batch,
        "workers": args.workers, "cache": args.cache,
        "skip_phase1": args.skip_phase1,
    }
    if not args.skip_phase1:
        out["phase1_best"] = str(p2_start.resolve())
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
