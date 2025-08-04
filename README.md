# MLB Glove‑Tracking Transfer Learning

This repository contains all code, scripts and documentation to fine‑tune a
YOLO‑based glove‑tracking model for Major League Baseball (MLB) using the
`baseball_rubber_home_glove` dataset.  The goal of this project is to
improve the accuracy of an existing glove‑tracking model by performing
transfer learning on a domain‑specific dataset of pitchers’ gloves in the
``rubber/home`` view from MLB broadcasts.  The work draws on the
pre‑trained model provided in the original [`BaseballCV` project](https://github.com/BaseballCV/BaseballCV) and uses the Ultralytics
YOLO implementation for simplicity and reproducibility.

## Contents

| Path | Description |
| --- | --- |
| `data/` | Utility script for downloading the dataset and pre‑trained weights.  **The raw dataset is not included in this repository** because the download server requires authentication.  Use `python data/download_data.py` to fetch the zip file. |
| `models/` | Placeholder for pre‑trained weights.  After running `download_data.py` the file `glove_tracking_v4_YOLOv11.pt` will be placed here. |
| `scripts/train.py` | Training script for fine‑tuning the glove‑tracking model on the new dataset. |
| `scripts/evaluate.py` | Script to evaluate the fine‑tuned model against the baseline model. |
| `report.md` | Detailed write‑up of the modelling pipeline, including assumptions, data preparation, training choices and evaluation. |
| `requirements.txt` | Python dependencies required to run the scripts. |

## Quick start

1. **Clone this repository** (private) and install dependencies into a clean
   Python environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **Download the dataset and pre‑trained weights.**  The `BaseballCV`
   repository stores a pointer to the dataset rather than the images
   themselves【49790347083117†L209-L215】; similarly the glove‑tracking
   weights file is provided via a download link.  Run:

   ```bash
   python data/download_data.py
   ```

   This script downloads the `baseball_rubber_home_glove.zip` file and
   `glove_tracking_v4_YOLOv11.pt` into the appropriate subfolders.  If
   the download fails because the remote server returns a **403 Forbidden**, you
   may need to manually download the zip file from
   <https://data.balldatalab.com/index.php/s/pLy7sZqqMdx3jj7/download/baseball_rubber_home_glove.zip> and
   place it in the `data/raw/` directory.  The pre‑trained weights can be
   obtained from <https://data.balldatalab.com/index.php/s/BwwWJbSsesFSBDa/download/glove_tracking_v4_YOLOv11.pt>.

3. **Unzip the dataset**.  After downloading, the dataset will be saved to
   `data/raw/baseball_rubber_home_glove.zip`.  Unzip it using:

   ```bash
   python data/download_data.py --extract
   ```

   This will create a directory `data/baseball_rubber_home_glove/` containing
   `images/` and `labels/` folders in YOLO format.

4. **Fine‑tune the model.**  Run the training script with reasonable
   hyper‑parameters:

   ```bash
   python scripts/train.py --data data/baseball_rubber_home_glove.yaml \
       --weights models/glove_tracking_v4_YOLOv11.pt --epochs 50 --imgsz 640
   ```

   The script uses Ultralytics YOLO (v8) under the hood.  Adjust the number of
   epochs, batch size and image size based on available GPU resources.

5. **Evaluate the fine‑tuned model.**  After training completes, evaluate
   performance relative to the baseline using:

   ```bash
   python scripts/evaluate.py --data data/baseball_rubber_home_glove.yaml \
       --weights models/glove_tracking_v4_YOLOv11.pt --fine_tuned runs/train/exp/weights/best.pt
   ```

   The evaluation script outputs mean average precision (mAP), precision and
   recall for both the baseline and fine‑tuned models.

## Notes

* This repository does **not** embed the raw images or large weight files
  because of size constraints and authentication required by the hosting
  server.  The provided download script will fetch them when possible.
* Ensure you have CUDA‑enabled GPUs available if training for many epochs.  If
  using CPU only, you may need to reduce image size and epochs to avoid
  excessive runtime.
* See `report.md` for a detailed discussion of the modelling choices and
  evaluation results.