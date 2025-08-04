# MLB Glove‑Tracking Transfer Learning

This repository contains all code, scripts and documentation to fine‑tune a YOLO‑based glove‑tracking model for Major League Baseball (MLB) using the `baseball_rubber_home_glove` dataset.  The goal of this project is to improve the accuracy of an existing glove‑tracking model by performing transfer learning on a domain‑specific dataset of pitchers’ gloves in the ``rubber/home`` view from MLB broadcasts.  The work draws on the pre‑trained model provided in the original [`BaseballCV` project](https://github.com/BaseballCV/BaseballCV) and uses the Ultralytics YOLO implementation for simplicity and reproducibility.

## Contents

| Path | Description |
| --- | --- |
| `data/` | Utility script for downloading **and preparing** the dataset and pre‑trained weights. **The raw dataset is not included** because the download server requires authentication. Use `python data/download_data.py` to fetch the zip file and `--extract` to unzip and build the dataset YAML. |
| `models/` | Placeholder for pre‑trained weights. After running `download_data.py` the file `glove_tracking_v4_YOLOv11.pt` will be placed here. |
| `scripts/train.py` | Training script for fine‑tuning the glove‑tracking model on the new dataset. |
| `scripts/evaluate.py` | Script to evaluate the fine‑tuned model against the baseline model. |
| `report.md` | Detailed write‑up of the modelling pipeline, including assumptions, data preparation, training choices and evaluation. |
| `run_pipeline.py` | One‑stop script that extracts the dataset, creates the YAML file, trains the model and evaluates baseline vs. fine‑tuned results. |
| `requirements.txt` | Python dependencies required to run the scripts. |

## Quick start

1. **Clone this repository** (private) and install dependencies into a clean Python environment:

   ```bash
   # clone the repository (requires access to your private repo)
   git clone https://github.com/TheDanGriff/MLB_GloveTracking_TransferLearning.git
   cd MLB_GloveTracking_TransferLearning

   # create and activate a virtual environment
   python3 -m venv .venv
   source .venv/bin/activate  # on Windows use ".venv\\Scripts\\activate"

   # install dependencies
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **Download and prepare the dataset and weights.**  The external links in BaseballCV point to the dataset and pre‑trained weights【49790347083117†L209-L215】. Run:

   ```bash
   python data/download_data.py
   ```

   This will attempt to fetch `baseball_rubber_home_glove.zip` and `glove_tracking_v4_YOLOv11.pt` into the `data/raw/` and `models/` folders respectively.  If the server returns **403 Forbidden** you will need to download the zip and weight files manually from:

   - Dataset: <https://data.balldatalab.com/index.php/s/pLy7sZqqMdx3jj7/download/baseball_rubber_home_glove.zip>
   - Weights: <https://data.balldatalab.com/index.php/s/BwwWJbSsesFSBDa/download/glove_tracking_v4_YOLOv11.pt>

   Place the ZIP file in either `data/raw/` or directly under `data/`, and the weights file in `models/`.

3. **Extract the dataset and build the dataset YAML.**  Once you have the zip file, run:

   ```bash
   python data/download_data.py --extract
   ```

   This unzips the archive into `data/baseball_rubber_home_glove/`, handles nested directories automatically and writes `data/baseball_rubber_home_glove.yaml` with the correct `train/val/test` splits.

4. **Run the full training and evaluation pipeline.**  Instead of executing training and evaluation separately, you can run everything with a single command:

   ```bash
   python run_pipeline.py --zip-path data/baseball_rubber_home_glove.zip \
       --weights models/glove_tracking_v4_YOLOv11.pt --epochs 50 --batch 16 \
       --imgsz 640 --name glove_ft_pipeline
   ```

   On Windows `cmd.exe` you should either use the caret (`^`) for line continuations or put the entire command on one line. For example:

   ```cmd
   python run_pipeline.py --zip-path data/baseball_rubber_home_glove.zip --weights models/glove_tracking_v4_YOLOv11.pt --epochs 50 --batch 16 --imgsz 640 --name glove_ft_pipeline
   ```

   The script will extract the dataset (if needed), fine‑tune the model with Ultralytics YOLO and then evaluate both the baseline and fine‑tuned models, printing mAP@0.5, mAP@0.5:0.95, precision and recall for each.

5. **(Alternative) Run training and evaluation manually.**  If you prefer to invoke the scripts yourself, use the following commands after the extraction step:

   ```bash
   # Train
   python scripts/train.py --data data/baseball_rubber_home_glove.yaml \
       --weights models/glove_tracking_v4_YOLOv11.pt --epochs 50 --batch 16 --imgsz 640 --name glove_ft_custom

   # Evaluate
   python scripts/evaluate.py --data data/baseball_rubber_home_glove.yaml \
       --weights models/glove_tracking_v4_YOLOv11.pt \
       --fine_tuned runs/train/glove_ft_custom/weights/best.pt --imgsz 640
   ```

   This yields the same metrics as the unified pipeline but gives you finer control over training options.

## Notes

* This repository does **not** embed the raw images or large weight files because of size constraints and authentication required by the hosting server. The provided download script will fetch them when possible.
* Ensure you have CUDA‑enabled GPUs available if training for many epochs. If using CPU only, you may need to reduce image size and epochs to avoid excessive runtime.
* See `report.md` for a detailed discussion of the modelling choices and evaluation results.
