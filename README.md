# MLB Glove-Tracking — Transfer Learning (Ultralytics YOLO)

This project improves the **glove-tracking** component of a pre-trained YOLO model using the
`baseball_rubber_home_glove` dataset from BaseballCV. We leverage **transfer learning** to adapt the
pretrained detector to MLB broadcast “rubber/home” views while keeping the training stable on Windows + RTX GPUs.

> **What you submit:** a private GitHub repository containing all **code** and **documentation** to run the pipeline and a write-up (this README, or a notebook) that justifies each modeling decision and compares your fine-tuned model against the **original pretrained** model.

---

## Table of contents

1. [Goals & constraints](#goals--constraints)  
2. [Data & task](#data--task)  
3. [Model & transfer strategy](#model--transfer-strategy)  
4. [Why a two-phase pipeline?](#why-a-two-phase-pipeline)  
5. [Environment & installation](#environment--installation)  
6. [How to run (one-liners)](#how-to-run-one-liners)  
7. [What each phase does](#what-each-phase-does)  
8. [Hyperparameters & design choices](#hyperparameters--design-choices)  
9. [Evaluation methodology](#evaluation-methodology)  
10. [Results & interpretation](#results--interpretation)  
11. [Ablations & “best accuracy” recipe](#ablations--best-accuracy-recipe)  
12. [Troubleshooting & known pitfalls](#troubleshooting--known-pitfalls)  
13. [Reproducibility & submission checklist](#reproducibility--submission-checklist)  
14. [Repo layout](#repo-layout)  
15. [License](#license)

---

## Goals & constraints

- **Goal:** Improve **glove detection** quality (mAP@0.50 and mAP@0.50:0.95, precision/recall) over the **pretrained** BaseballCV YOLO weights in a way that’s **reproducible** and **well-justified**.
- **Constraints we account for:**
  - Windows + NVIDIA RTX (PyTorch + CUDA 12.1 wheels).
  - Occasional Windows **data-loader stalls**: mitigated with a **warm-up epoch**, `workers=0`, `cache=none`, and delayed mosaic.
  - Dataset is smallish; careful **augmentation scheduling** and light regularization help avoid over/under-fit.

---

## Data & task

- Dataset ZIP: `baseball_rubber_home_glove.zip` (images + YOLO text labels).
- Native labels include **five classes** (`glove`, `homeplate`, `baseball`, `rubber`, `na`).  
  **Our target task** is **glove detection**; we evaluate on a **glove-only view** (nc=1).
- We keep the dataset splits (`train/val/test`) intact. The pipeline creates two YAMLs at runtime:
  - **Multi-class YAML (nc=5)** for optional Phase-1.
  - **Glove-only YAML (nc=1)** for Phase-2 training and all glove-only evaluations.

---

## Model & transfer strategy

- Base detector: **Ultralytics YOLOv11x** weights from BaseballCV (`glove_tracking_v4_YOLOv11.pt`).  
  We load these weights into Ultralytics > 8.3.x.
- **Transfer learning** path:
  - **Recommended (default):** **Phase-2 only** (glove-only fine-tune, nc=1). This is the most **stable** and **fast** way to improve glove detection on Windows.
  - **Optional:** **Phase-1 multi-class** (nc=5) → then Phase-2 (nc=1). Phase-1 slightly adapts general features to the dataset distribution before specializing on glove. Useful if you have time and want to squeeze extra recall in busy scenes; can be slower/less stable on Windows.

---

## Why a two-phase pipeline?

1. **Phase-1 (optional):** Multi-class fine-tune (nc=5) reminds the model of **contextual co-occurrences** (baseball, rubber, homeplate). This can reduce some false positives/negatives on ambiguous gloves because features remain aligned to the full scene semantics.
2. **Phase-2 (recommended):** Switch to **glove-only (nc=1)** and specialize the final heads while gently training the backbone. This focuses capacity on glove localization and improves glove-only metrics.

> In practice (your machine, dataset size, Windows/RTX stack), **Phase-2 alone** delivered strong and stable results. Phase-1 is exposed as a switch for completeness and ablations.

---

## Environment & installation

> Python 3.10–3.12; these instructions were validated with **Python 3.11.7** on Windows + RTX 4060.

```bash
git clone https://github.com/<your_org>/<your_repo>.git
cd <your_repo>

# Create & activate venv
python -m venv .venv
. .venv/Scripts/Activate.ps1   # PowerShell
# or: .venv\Scripts\activate   # cmd.exe

# Install PyTorch for CUDA 12.1 (RTX 40-series)
pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121

# Then the rest
pip install --upgrade pip
pip install -r requirements.txt
