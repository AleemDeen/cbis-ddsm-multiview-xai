# CBIS-DDSM Multiview XAI

**CBIS-DDSM multiview (CC + MLO) mammography classification with explainable AI (Grad-CAM) and patch-level localisation evaluation (IoU, Dice, Pointing Game).**

*University of Exeter undergraduate dissertation project.*

This project implements a single-view baseline (with planned extension to multiview learning) using CNN-based classification on the CBIS-DDSM dataset, alongside explainability and quantitative ROI localisation using radiologist-annotated masks.

---

## Project Structure

| Directory | Description |
| :--- | :--- |
| `configs/` | YAML configuration files |
| `data_raw/` | Raw CBIS-DDSM dataset (not tracked) |
| `data_processed/` | Processed data (generated) |
| `runs/` | Training runs and logs (generated) |
| `results/` | Evaluation outputs (generated) |
| `splits/` | Train/validation/test case splits |
| `src/data/` | Dataset indexing, loaders, and preprocessing |
| `src/models/` | Model architectures |
| `src/train/` | Training scripts |
| `src/eval/` | Evaluation scripts |
| `src/explainability/` | Grad-CAM and XAI utilities |
| `src/utils/` | Config and reproducibility helpers |

> **Note:** Large files such as datasets, trained models, and experiment outputs are intentionally excluded via `.gitignore`.

---

## Setup

### 1. Initialise Virtual Environment
Run the following from the project root:

~~~powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
~~~

### 2. Create Required Directories
The following directories are ignored by Git and must be created locally:

```powershell
mkdir data_raw
mkdir data_processed
mkdir runs
mkdir results
```

### 3. Download and Place Dataset and Metadata
This project uses the CBIS-DDSM dataset.

Download CBIS-DDSM, or other relevant datasets, from the official TCIA repository.

Place the dataset inside `dataset/`.

If your dataset is stored elsewhere (e.g., an external drive), update the dataset path in `configs/base.yaml`.

Expected structure:

```powershell
dataset/
└── cbis_ddsm/
    └── manifest-xxxx/
        └── CBIS-DDSM/
```

As well as the metadata

```powershell
data_raw/
└── cbis_ddsm_dataset/
    └── xxxx.csv/
```

### 4. Verify Dataset and Environment
Run the smoke test to confirm paths, indexing, and environment setup:

```powershell
python -m src.data.smoke_test_setup
```

### 5. Create a CSV with labels for each patient

Run the following in the root directory (whilst in the virtual environment)
```
python -m src.data.build_case_level_labels
```

### 6. Create splits

Create train/val/test splits

```
python -m src.data.create_splits
```

### 7. (OPTIONAL) GPU Configuration

By default the programme is set ot train the model via the CPU. If you wish to enable GPU acceleration follow the steps:

Remove the CPU only PyTorch:

```
pip uninstall torch torchvision torchaudio -y
```
Re-install PyTorch with Cuda

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify the GPU by running 'python' in terminal and the following:

```
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

Which should return:

```
True
GPU Name xxxx
```