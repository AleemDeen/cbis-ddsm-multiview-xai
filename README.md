# CBIS-DDSM Multi-View XAI

Explainable AI system for mammogram malignancy classification using the CBIS-DDSM dataset. Takes paired **CC** (cranio-caudal) and **MLO** (medio-lateral oblique) view DICOMs, classifies them as **benign or malignant**, and produces spatial localisation overlays highlighting the suspicious region.

*University of Exeter undergraduate dissertation project.*

---

## Quick Start

The quickest way to run the application is using the platform-specific launch script. It will automatically set up the Python virtual environment, install all dependencies, and start both the backend and frontend in one step.

**Prerequisites:** [Python 3.10+](https://www.python.org/downloads/) and [Node.js (LTS)](https://nodejs.org/) must be installed before running the script.

---

### Windows

Double-click `start.bat` in the project root, or run it from a terminal:

```cmd
start.bat
```

This opens two separate terminal windows — one for the backend, one for the frontend — and launches the application in your browser automatically. Close those windows to shut down the servers.

---

### macOS

The script needs execute permission the first time. Open a terminal in the project root and run:

```bash
chmod +x start.sh
./start.sh
```

From the second run onwards, just:

```bash
./start.sh
```

Both servers run in the background within the same terminal. Press **Ctrl+C** to shut them both down cleanly.

---

### Linux

Same as macOS — grant execute permission once, then run:

```bash
chmod +x start.sh
./start.sh
```

If Python or Node are not installed, the script will print the relevant install commands for your package manager (apt for Ubuntu/Debian, dnf for Fedora).

Press **Ctrl+C** to shut down both servers.

---

Once running, the application is available at:

| Service | URL |
|---|---|
| Frontend | http://localhost:5173 |
| Backend API | http://localhost:8000 |
| API docs | http://localhost:8000/docs |

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Installation](#2-installation)
3. [Project Structure](#3-project-structure)
4. [Dataset Setup](#4-dataset-setup)
5. [Required Directories](#5-required-directories)
6. [Dataset Preparation](#6-dataset-preparation)
7. [Data Preprocessing](#7-data-preprocessing)
8. [Model Training](#8-model-training)
9. [Model Evaluation](#9-model-evaluation)
10. [Backend Setup](#10-backend-setup)
11. [Frontend Setup](#11-frontend-setup)
12. [Full Pipeline](#12-full-pipeline)
13. [GPU Configuration](#13-gpu-configuration)

---

## 1. Project Overview

This system implements a **dual-branch ResNet18** architecture that processes CC and MLO mammogram views simultaneously. The two branches are concatenated for joint malignancy classification, and a **U-Net segmentation decoder** with skip connections produces per-pixel ROI probability masks that highlight where in the image the model predicts malignancy.

Three model architectures are included:

| Model | Description |
|---|---|
| `ResNet18SingleView` | Single DICOM → malignancy score + GradCAM heatmap |
| `ResNet18MultiView` | CC + MLO DICOMs → malignancy score + GradCAM heatmaps |
| `ResNet18MultiViewSeg` | CC + MLO DICOMs → malignancy score + U-Net ROI masks |

Four trained checkpoints are provided in `models/`:

| File | Architecture | AUC |
|---|---|---|
| `sv_baseline.pt` | ResNet18SingleView | 0.7867 |
| `sv_best.pt` | ResNet18SingleView | 0.7569 |
| `mv_baseline.pt` | ResNet18MultiView | 0.8321 |
| `mv_best.pt` | ResNet18MultiViewSeg | 0.8276 |

The system includes a **FastAPI backend** and **React frontend** for interactive inference via a web interface.

---

## 2. Installation

### Prerequisites

- Python 3.10+
- Node.js 18+ and npm (for the frontend)
- CUDA-capable GPU recommended (CPU works but is slow for training)

### Clone and install

```bash
git clone <repository-url>
cd cbis-ddsm-multiview-xai-main
```

Create and activate a virtual environment:

```bash
# Windows
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

> **PyTorch note:** `requirements.txt` installs CPU-only PyTorch by default. See [Section 13](#13-gpu-configuration) to enable GPU acceleration.

---

## 3. Project Structure

```
cbis-ddsm-multiview-xai-main/
├── models/                        # Trained model checkpoints
│   ├── sv_baseline.pt             # SV Baseline  (AUC 0.7867)
│   ├── sv_best.pt                 # SV Best      (AUC 0.7569)
│   ├── mv_baseline.pt             # MV Baseline  (AUC 0.8321)
│   └── mv_best.pt                 # MV Best      (AUC 0.8276)
├── data_processed/                # Preprocessed .pt tensors + CSV index (generated)
├── splits/                        # Train / val / test case ID lists
├── examples/                      # 5 example patient DICOM pairs for quick testing
├── architectures/                 # Architecture diagram PNGs
├── graphs/                        # Model comparison graphs
├── debug/                         # Sanity checks and debug scripts
├── src/
│   ├── models/
│   │   ├── resnet18_single_view.py      # Single-view ResNet18
│   │   └── resnet18_multi_view.py       # Dual-branch ResNet18 + U-Net SegDecoder
│   ├── data/
│   │   ├── create_multi_view_csv.py     # Build indexed_multi_view_cases.csv
│   │   ├── preprocess_multi_view_to_pt.py  # Convert DICOMs → .pt tensors
│   │   ├── preprocess_to_pt.py          # Single-view DICOM → .pt conversion
│   │   ├── multi_view_dataset.py        # Multi-view dataset class
│   │   ├── mammogram_dataset.py         # Single-view dataset class
│   │   ├── create_splits.py             # Generate train/val/test splits
│   │   └── build_case_level_labels.py   # Build case-level label CSV
│   ├── train/
│   │   ├── train_single_view.py         # Train single-view classifier → sv_best.pt
│   │   ├── train_multi_view.py          # Train dual-branch classifier → mv_baseline.pt
│   │   └── finetune_multi_view_seg_v4.py  # Fine-tune U-Net seg heads → mv_best.pt
│   ├── eval/
│   │   ├── evaluate_single_view.py      # Evaluate single-view model
│   │   └── evaluate_multi_view.py       # Evaluate multi-view model (GradCAM or seg)
│   ├── xai/
│   │   ├── gradcam.py                   # GradCAM implementation
│   │   ├── scorecam.py                  # ScoreCAM implementation
│   │   └── run_gradcam_single_view.py   # GradCAM visualisation script
│   ├── api/
│   │   ├── server.py                    # FastAPI app
│   │   └── model_runner.py              # Inference, GradCAM, mask overlay
│   └── frontend/                        # React + Vite web interface
│       ├── src/
│       │   ├── App.jsx
│       │   └── components/
│       │       ├── ModelSelector.jsx
│       │       ├── UploadArea.jsx
│       │       └── ResultsPanel.jsx
│       ├── package.json
│       └── vite.config.js
├── generate_graphs.py             # Reproduce model comparison graphs
├── generate_architectures.py      # Reproduce architecture diagrams
├── export_example_masks.py        # Export ground-truth ROI overlays for examples/
├── requirements.txt
└── README.md
```

---

## 4. Dataset Setup

This project uses the **CBIS-DDSM** (Curated Breast Imaging Subset of DDSM) dataset.

Download it from the [TCIA repository](https://www.cancerimagingarchive.net/collection/cbis-ddsm/).

Place the DICOM files and CSV metadata inside a `data/` folder at the project root:

```
data/
├── CBIS-DDSM/
│   └── <case folders with DICOMs>
└── cbis_ddsm_metadata/
    ├── mass_case_description_train_set.csv
    ├── mass_case_description_test_set.csv
    ├── calc_case_description_train_set.csv
    └── calc_case_description_test_set.csv
```

> The `data/` directory is excluded from git (see `.gitignore`). It must be set up locally on each machine.

---

## 5. Required Directories

These directories are not tracked by git and must be created manually before running any scripts:

```bash
mkdir data
mkdir data_processed
mkdir models
mkdir runs
mkdir results
```

---

## 6. Dataset Preparation

### Step 1 — Build case-level labels

```bash
python -m src.data.build_case_level_labels
```

Outputs a CSV mapping each patient case to a benign/malignant label.

### Step 2 — Build multi-view index CSV

```bash
python -m src.data.create_multi_view_csv
```

Outputs: `data_processed/indexed_multi_view_cases.csv`

Each row links a patient's CC DICOM path, MLO DICOM path, ROI mask paths, and label.

### Step 3 — Create train/val/test splits

```bash
python -m src.data.create_splits
```

Outputs case-level split files to `splits/`:
- `splits/train_cases.txt` — 850 cases
- `splits/val_cases.txt` — 166 cases
- `splits/test_cases.txt` — 181 cases

Splits are **patient-level** to prevent data leakage between sets.

---

## 7. Data Preprocessing

Converting DICOMs to pre-cached `.pt` tensors is optional but **strongly recommended** — it speeds up training by ~10× by eliminating per-epoch DICOM decoding.

```bash
python -m src.data.preprocess_multi_view_to_pt
```

Outputs: `data_processed/indexed_multi_view_cases_pt.csv`

Each row adds `cc_pt_path`, `mlo_pt_path`, `cc_mask_pt_path`, `mlo_mask_pt_path` columns pointing to the saved tensors. All training and evaluation scripts auto-detect this file and prefer it over raw DICOMs.

---

## 8. Model Training

All models are saved to the `models/` directory.

### Single-view classifier

```bash
python -m src.train.train_single_view
```

Saves: `models/sv_best.pt`

### Multi-view classifier (MV Baseline)

```bash
python -m src.train.train_multi_view
```

Saves: `models/mv_baseline.pt`

### U-Net segmentation head fine-tuning (MV Best)

Train the MV Baseline first, then fine-tune the U-Net decoders:

```bash
python -m src.train.finetune_multi_view_seg_v4 --base-model models/mv_baseline.pt --epochs 40
```

**What this does:**
- Loads the pre-trained classification backbone (`mv_baseline.pt`)
- Freezes `conv1`, `layer1`, `layer2`, and the classifier head
- Unfreezes `layer3` and `layer4` so features become spatially discriminative
- Trains two U-Net decoders (one per view) with skip connections from all four ResNet stages
- Combined loss: classification BCE + weighted segmentation BCE + Dice
- Backbone LR: `1e-5` | Decoder LR: `1e-3` | Batch size: `8`

Saves: `models/mv_best.pt`

---

## 9. Model Evaluation

### Single-view

```bash
python -m src.eval.evaluate_single_view --model-path models/sv_best.pt
```

### Multi-view (GradCAM)

```bash
python -m src.eval.evaluate_multi_view --model-path models/mv_baseline.pt
```

### Multi-view with U-Net seg heads

```bash
python -m src.eval.evaluate_multi_view --model-path models/mv_best.pt --seg-head
```

Reports AUC, accuracy, recall, and soft/hard Dice scores for both CC and MLO views across all, malignant-only, and true-positive subsets.

### Results summary

| Model | AUC | Accuracy | Recall | Specificity |
|---|---|---|---|---|
| SV Baseline (`sv_baseline.pt`) | 0.7867 | 0.7074 | 0.7892 | 0.6348 |
| SV Best (`sv_best.pt`) | 0.7569 | 0.6843 | 0.7157 | 0.6565 |
| MV Baseline (`mv_baseline.pt`) | 0.8321 | 0.7735 | 0.8434 | 0.7143 |
| MV Best (`mv_best.pt`) | 0.8276 | 0.7680 | 0.7470 | 0.7857 |

`mv_best.pt` localisation (seg head, malignant subset):

| Metric | Value |
|---|---|
| CC Hard Dice | 0.4719 |
| MLO Hard Dice | 0.5129 |
| MLO Hard Dice (true positives) | 0.5545 |

---

## 10. Backend Setup

The FastAPI backend handles model loading, DICOM parsing, inference, and overlay generation.

### Install additional backend dependency

```bash
pip install fastapi uvicorn python-multipart
```

### Run the server

From the **project root**:

```bash
uvicorn src.api.server:app --reload --host 0.0.0.0 --port 8000
```

The server will be available at `http://localhost:8000`.

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/models` | List all `.pt` files in the `models/` folder |
| `POST` | `/predict` | Run inference on uploaded DICOM(s) |

### Model auto-detection

The backend detects model type from the filename:
- `sv_*.pt` → single-view model + GradCAM
- `mv_best.pt` → multi-view model + U-Net seg masks
- `mv_*.pt` → multi-view model + GradCAM

---

## 11. Frontend Setup

The React + Vite frontend provides an interactive web interface for uploading DICOMs and visualising results.

### Install dependencies

```bash
cd src/frontend
npm install
```

### Run the development server

```bash
npm run dev
```

The frontend will be available at `http://localhost:5173`.

Vite automatically proxies all `/api/*` requests to the FastAPI backend at `http://localhost:8000` — both servers must be running simultaneously.

### Interface features

- Model selector dropdown (lists all models from `models/`)
- CC and MLO DICOM file upload with drag-and-drop
- Malignant prediction: shows original image + ROI localisation overlay for both views
- Benign prediction: shows original image only (no localisation — clinically appropriate)
- Confidence score displayed as a circular progress indicator

---

## 12. Full Pipeline

End-to-end walkthrough from a fresh clone to a running system:

```bash
# 1. Clone and install
git clone <repository-url>
cd cbis-ddsm-multiview-xai-main
python -m venv .venv && .\.venv\Scripts\Activate.ps1   # Windows
pip install -r requirements.txt

# 2. Create required directories
mkdir data data_processed models runs results

# 3. Place CBIS-DDSM dataset in data/

# 4. Build labels and index
python -m src.data.build_case_level_labels
python -m src.data.create_multi_view_csv

# 5. Create splits
python -m src.data.create_splits

# 6. Preprocess DICOMs to .pt tensors (recommended)
python -m src.data.preprocess_multi_view_to_pt

# 7. Train MV Baseline
python -m src.train.train_multi_view

# 8. Fine-tune U-Net seg heads (MV Best)
python -m src.train.finetune_multi_view_seg_v4 --base-model models/mv_baseline.pt --epochs 40

# 9. Evaluate
python -m src.eval.evaluate_multi_view --model-path models/mv_best.pt --seg-head

# 10. Start backend (terminal 1)
uvicorn src.api.server:app --reload --host 0.0.0.0 --port 8000

# 11. Start frontend (terminal 2)
cd src/frontend && npm install && npm run dev

# 12. Open http://localhost:5173 in your browser
```

### Quick demo (no training required)

If you have the pre-trained model weights, place them in `models/` and skip steps 4–9. Use the example DICOMs in `examples/` to test the interface immediately.

```
examples/
├── P_00001_LEFT_malignant_CC.dcm   ← upload as CC view
├── P_00001_LEFT_malignant_MLO.dcm  ← upload as MLO view
├── P_00004_LEFT_benign_CC.dcm
├── P_00004_LEFT_benign_MLO.dcm
└── ...
```

---

## 13. GPU Configuration

By default `requirements.txt` installs CPU-only PyTorch. For GPU training:

```bash
# Remove CPU-only build
pip uninstall torch torchvision torchaudio -y

# Install CUDA 12.1 build
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify GPU is detected:

```python
import torch
print(torch.cuda.is_available())     # True
print(torch.cuda.get_device_name(0)) # e.g. NVIDIA GeForce RTX 3080
```

All training and evaluation scripts automatically use CUDA if available, falling back to CPU otherwise.

---

## Known Limitations

- **Diffuse localisation**: Hard Dice ~0.47–0.55 indicates the highlighted region covers a broad area of breast tissue rather than a tight spot around the lesion. This is a known limitation of training spatial localisation from a classification backbone with ~850 training cases.
- **Dataset size ceiling**: ~850 training cases limits the precision achievable for weakly-supervised localisation.
- **Single dataset**: Results are from CBIS-DDSM only and may not generalise to other mammography datasets or acquisition protocols.
