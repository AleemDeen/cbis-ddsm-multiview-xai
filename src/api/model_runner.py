"""
Model loading and inference for the mammogram XAI API.
Supports:
  - ResNet18SingleView  (single DICOM → GradCAM)
  - ResNet18MultiView   (CC + MLO DICOMs → GradCAM per view)
  - ResNet18MultiViewSeg (CC + MLO DICOMs → seg-head masks per view)
"""

import io
import base64
import numpy as np
import cv2
import pydicom
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image

# ── lazy imports (avoid loading torch models at import time) ──────────────────
_model_cache: dict = {}


# ─────────────────────────────── DICOM helpers ───────────────────────────────

def load_dicom_bytes(data: bytes, size: int = 512) -> np.ndarray:
    ds = pydicom.dcmread(io.BytesIO(data))
    img = ds.pixel_array.astype(np.float32)
    if (
        hasattr(ds, "PhotometricInterpretation")
        and ds.PhotometricInterpretation == "MONOCHROME1"
    ):
        img = img.max() - img
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img -= img.min()
    if img.max() > 0:
        img /= img.max()
    return img


# ─────────────────────────────── Image → base64 ──────────────────────────────

def _to_base64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def gray_to_base64(img: np.ndarray) -> str:
    return _to_base64((img * 255).astype(np.uint8))


def heatmap_overlay_base64(img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> str:
    img_rgb  = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    colored  = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay  = cv2.addWeighted(img_rgb, 1 - alpha, colored, alpha, 0)
    return _to_base64(overlay[:, :, ::-1])   # BGR→RGB


def mask_overlay_base64(img: np.ndarray, mask: np.ndarray, alpha: float = 0.55) -> str:
    """
    Jet-colormap overlay for segmentation mask.
    Normalises the mask to its own [min, max] range so relative activation
    differences are visible even when the seg head outputs uniformly high values.
    Per-pixel alpha ensures low-activation areas stay close to the original.
    """
    img_rgb = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

    # Normalise to full [0, 1] range to reveal relative hotspots
    m_min, m_max = mask.min(), mask.max()
    norm = (mask - m_min) / (m_max - m_min + 1e-8)

    colored = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    colored_rgb = colored[:, :, ::-1]  # BGR → RGB

    # Fixed alpha blend so the colormap is always visible across the full image
    overlay = cv2.addWeighted(img_rgb, 1.0 - alpha, colored_rgb, alpha, 0)
    return _to_base64(overlay.astype(np.uint8))


# ─────────────────────────────── GradCAM ────────────────────────────────────

def _gradcam(features: torch.Tensor, loss: torch.Tensor, retain: bool = False) -> np.ndarray:
    grads   = torch.autograd.grad(loss, features, retain_graph=retain, create_graph=False)[0]
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam     = F.relu((weights * features).sum(dim=1, keepdim=True))
    cam     = cam / (cam.amax(dim=(2, 3), keepdim=True) + 1e-8)
    cam_up  = F.interpolate(cam, size=(512, 512), mode="bilinear", align_corners=False)
    return cam_up[0, 0].detach().cpu().numpy()


# ─────────────────────────────── Model loader ────────────────────────────────

def _detect_type(name: str) -> str:
    n = name.lower()
    if "single" in n:
        return "single"
    if "seg" in n:
        return "multi_seg"
    if "multi" in n:
        return "multi"
    return "single"   # default guess


MODELS_DIR = Path("models")


def _load_model(model_name: str, device: torch.device):
    from src.models.resnet18_single_view import ResNet18SingleView
    from src.models.resnet18_multi_view import ResNet18MultiView, ResNet18MultiViewSeg

    mtype = _detect_type(model_name)
    # Resolve path: prefer models/ folder, fall back to cwd for absolute paths
    path = MODELS_DIR / model_name if not Path(model_name).is_absolute() else Path(model_name)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    if mtype == "single":
        m = ResNet18SingleView()
    elif mtype == "multi_seg":
        m = ResNet18MultiViewSeg()
    else:
        m = ResNet18MultiView()

    sd = torch.load(str(path), map_location=device, weights_only=True)
    m.load_state_dict(sd, strict=(mtype != "multi_seg"))
    m.to(device)
    return m, mtype


def get_model(model_name: str, device: torch.device):
    key = (model_name, str(device))
    if key not in _model_cache:
        _model_cache[key] = _load_model(model_name, device)
    return _model_cache[key]


# ─────────────────────────────── Model listing ───────────────────────────────

def list_available_models() -> list[str]:
    pts = sorted(MODELS_DIR.glob("*.pt"))
    return [p.name for p in pts]


# ─────────────────────────────── Inference ───────────────────────────────────

def _diagnosis_text(prob: float) -> str:
    label = "Malignant" if prob >= 0.5 else "Benign"
    conf  = prob * 100 if prob >= 0.5 else (1 - prob) * 100
    return f"{label} ({conf:.1f}% confidence)"


def run_inference(model_name: str, cc_bytes: bytes, mlo_bytes: bytes | None) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mtype = get_model(model_name, device)

    cc_img    = load_dicom_bytes(cc_bytes)
    cc_tensor = torch.from_numpy(cc_img).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,512,512)

    # ── Single-view ──────────────────────────────────────────────────────────
    if mtype == "single":
        model.eval()
        for p in model.parameters():
            p.requires_grad_(True)

        logits, features = model(cc_tensor, return_features=True)
        prob = torch.sigmoid(logits).item()
        loss = torch.sigmoid(logits).view(-1).mean()
        cam  = _gradcam(features, loss)

        return {
            "diagnosis":   _diagnosis_text(prob),
            "probability": round(prob, 4),
            "model_type":  "single",
            "original_cc": gray_to_base64(cc_img),
            "heatmap_cc":  heatmap_overlay_base64(cc_img, cam),
        }

    # ── Multi-view (no seg head) ──────────────────────────────────────────────
    if mtype == "multi":
        if mlo_bytes is None:
            raise ValueError("Multi-view model requires an MLO DICOM file.")
        mlo_img    = load_dicom_bytes(mlo_bytes)
        mlo_tensor = torch.from_numpy(mlo_img).unsqueeze(0).unsqueeze(0).to(device)

        model.eval()
        for p in model.parameters():
            p.requires_grad_(True)

        logits, cc_feat, mlo_feat = model(cc_tensor, mlo_tensor, return_features=True)
        prob = torch.sigmoid(logits).item()
        loss = torch.sigmoid(logits).view(-1).mean()
        cc_cam  = _gradcam(cc_feat,  loss, retain=True)
        mlo_cam = _gradcam(mlo_feat, loss, retain=False)

        return {
            "diagnosis":    _diagnosis_text(prob),
            "probability":  round(prob, 4),
            "model_type":   "multi",
            "original_cc":  gray_to_base64(cc_img),
            "original_mlo": gray_to_base64(mlo_img),
            "heatmap_cc":   heatmap_overlay_base64(cc_img,  cc_cam),
            "heatmap_mlo":  heatmap_overlay_base64(mlo_img, mlo_cam),
        }

    # ── Multi-view with segmentation heads ──────────────────────────────────
    if mtype == "multi_seg":
        if mlo_bytes is None:
            raise ValueError("Multi-view model requires an MLO DICOM file.")
        mlo_img    = load_dicom_bytes(mlo_bytes)
        mlo_tensor = torch.from_numpy(mlo_img).unsqueeze(0).unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            logits, cc_mask, mlo_mask = model(cc_tensor, mlo_tensor, return_masks=True)

        prob       = torch.sigmoid(logits).item()
        cc_mask_np = cc_mask[0, 0].cpu().numpy()
        mlo_mask_np= mlo_mask[0, 0].cpu().numpy()

        # Only render localization overlays when the model predicts malignant —
        # for benign cases the seg head output has no clinical meaning.
        is_malignant = prob >= 0.5
        result = {
            "diagnosis":     _diagnosis_text(prob),
            "probability":   round(prob, 4),
            "model_type":    "multi_seg",
            "original_cc":   gray_to_base64(cc_img),
            "original_mlo":  gray_to_base64(mlo_img),
        }
        if is_malignant:
            result["seg_mask_cc"]  = mask_overlay_base64(cc_img,  cc_mask_np)
            result["seg_mask_mlo"] = mask_overlay_base64(mlo_img, mlo_mask_np)
        return result

    raise ValueError(f"Unknown model type: {mtype}")
