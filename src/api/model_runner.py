"""
Model loading and inference for the mammogram XAI API.

Handles three model types:
  - ResNet18SingleView    (single DICOM → GradCAM heatmap)
  - ResNet18MultiView     (CC + MLO DICOMs → GradCAM per view)
  - ResNet18MultiViewSeg  (CC + MLO DICOMs → U-Net segmentation masks per view)
"""

from __future__ import annotations  # allow X | Y union syntax on Python 3.9

import io
import base64
import numpy as np
import cv2
import pydicom
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image

# Models are cached after first load so repeated API calls do not reload weights
_model_cache: dict = {}


# ─────────────────────────────── DICOM helpers ───────────────────────────────

def load_dicom_bytes(data: bytes, size: int = 512) -> np.ndarray:
    """
    Decode a DICOM file from raw bytes and return a normalised float32 array.

    Handles MONOCHROME1 photometric interpretation by inverting the pixel values,
    which ensures bright pixels always represent dense tissue regardless of the
    acquisition convention used by the scanner.
    """
    ds  = pydicom.dcmread(io.BytesIO(data))
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
    """Encode a uint8 numpy image as a base64 PNG string for JSON transport."""
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def gray_to_base64(img: np.ndarray) -> str:
    """Convert a [0, 1] greyscale float array to a base64 PNG."""
    return _to_base64((img * 255).astype(np.uint8))


def heatmap_overlay_base64(img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> str:
    """
    Blend a Grad-CAM heatmap onto a greyscale mammogram and return as base64 PNG.

    Uses a fixed alpha blend — the heatmap is always visible regardless of
    prediction confidence, as Grad-CAM intensity already encodes importance.
    """
    img_rgb = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_rgb, 1 - alpha, colored, alpha, 0)
    return _to_base64(overlay[:, :, ::-1])   # OpenCV uses BGR — convert to RGB for PIL


def mask_overlay_base64(img: np.ndarray, mask: np.ndarray,
                         alpha: float = 0.65,
                         percentile: float = 82.0) -> str:
    """
    Produce a focused segmentation mask overlay for display in the frontend.

    Processing pipeline:
      1. Suppress heatmap over the black background (non-breast) region.
         The breast boundary is eroded to avoid highlighting the sharp intensity
         edge at the breast margin, which the decoder tends to latch onto.
      2. Threshold to show only the top activations within the breast tissue.
         This removes low-confidence background noise from the visualisation.
      3. Gaussian smoothing to reduce high-frequency speckle in the mask.
      4. Per-pixel alpha blend: only activated regions are coloured; the rest
         shows the original greyscale mammogram unchanged.
    """
    img_rgb = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

    # Step 1: build a breast tissue mask and erode its boundary
    breast_binary = (img > 0.05).astype(np.uint8)
    kernel        = np.ones((25, 25), np.uint8)
    breast_eroded = cv2.erode(breast_binary, kernel, iterations=2)
    breast        = breast_eroded.astype(np.float32)
    mask          = mask * breast

    # Step 2: percentile threshold computed within the breast tissue area only
    breast_vals = mask[breast > 0]
    threshold   = float(np.percentile(breast_vals, percentile)) if len(breast_vals) > 0 \
                  else float(np.percentile(mask, percentile))
    focused = np.where(mask >= threshold, mask, 0.0)

    # Step 3: normalise then smooth
    m_max = focused.max()
    norm  = focused / (m_max + 1e-8)
    norm  = cv2.GaussianBlur(norm.astype(np.float32), (15, 15), 0)

    # Step 4: variable alpha blend — high activations are fully coloured,
    # low activations are transparent (show only original image)
    alpha_map   = (norm * alpha)[..., np.newaxis]           # (H, W, 1)
    colored     = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    colored_rgb = colored[:, :, ::-1]                       # BGR → RGB

    overlay = ((1.0 - alpha_map) * img_rgb + alpha_map * colored_rgb).clip(0, 255).astype(np.uint8)
    return _to_base64(overlay)


# ─────────────────────────────── GradCAM ────────────────────────────────────

def _gradcam(features: torch.Tensor, loss: torch.Tensor, retain: bool = False) -> np.ndarray:
    """
    Compute a Grad-CAM heatmap from a stored feature map and a scalar loss.

    retain=True keeps the computation graph alive for additional backward passes,
    which is needed when computing Grad-CAM for the second branch of a multi-view
    model before calling the final backward.
    """
    grads   = torch.autograd.grad(loss, features, retain_graph=retain, create_graph=False)[0]
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam     = F.relu((weights * features).sum(dim=1, keepdim=True))
    cam     = cam / (cam.amax(dim=(2, 3), keepdim=True) + 1e-8)
    cam_up  = F.interpolate(cam, size=(512, 512), mode="bilinear", align_corners=False)
    return cam_up[0, 0].detach().cpu().numpy()


# ─────────────────────────────── Model loader ────────────────────────────────

def _detect_type(name: str) -> str:
    """
    Infer the model architecture from the checkpoint filename.

    Convention:
      sv_*.pt      → single-view ResNet18
      mv_best.pt   → multi-view with U-Net segmentation heads
      mv_*.pt      → multi-view ResNet18 (GradCAM only)
    """
    n = name.lower()
    if n.startswith("sv_"):
        return "single"
    if n == "mv_best.pt":
        return "multi_seg"
    if n.startswith("mv_"):
        return "multi"
    # Legacy fallback for any checkpoints not following the naming convention
    if "single" in n:
        return "single"
    if "seg" in n:
        return "multi_seg"
    if "multi" in n:
        return "multi"
    return "single"


MODELS_DIR = Path("models")


def _load_model(model_name: str, device: torch.device):
    from src.models.resnet18_single_view import ResNet18SingleView
    from src.models.resnet18_multi_view import ResNet18MultiView, ResNet18MultiViewSeg

    mtype = _detect_type(model_name)
    path  = MODELS_DIR / model_name if not Path(model_name).is_absolute() else Path(model_name)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    if mtype == "single":
        m = ResNet18SingleView()
    elif mtype == "multi_seg":
        m = ResNet18MultiViewSeg()
    else:
        m = ResNet18MultiView()

    sd = torch.load(str(path), map_location=device, weights_only=True)
    # strict=False for multi_seg because a base mv_baseline checkpoint may lack seg head keys
    m.load_state_dict(sd, strict=(mtype != "multi_seg"))
    m.to(device)
    return m, mtype


def get_model(model_name: str, device: torch.device):
    """Return a cached (model, model_type) pair, loading from disk if necessary."""
    key = (model_name, str(device))
    if key not in _model_cache:
        _model_cache[key] = _load_model(model_name, device)
    return _model_cache[key]


# ─────────────────────────────── Model listing ───────────────────────────────

def list_available_models() -> list[str]:
    """Return a sorted list of all .pt filenames in the models/ directory."""
    pts = sorted(MODELS_DIR.glob("*.pt"))
    return [p.name for p in pts]


# ─────────────────────────────── Inference ───────────────────────────────────

def _diagnosis_text(prob: float) -> str:
    """Format a probability as a human-readable diagnosis string with confidence."""
    label = "Malignant" if prob >= 0.5 else "Benign"
    conf  = prob * 100 if prob >= 0.5 else (1 - prob) * 100
    return f"{label} ({conf:.1f}% confidence)"


def run_inference(model_name: str, cc_bytes: bytes, mlo_bytes: bytes | None) -> dict:
    """
    Run inference and return a response dictionary containing the diagnosis,
    probability, and base64-encoded overlay images appropriate for the model type.

    Single-view: one GradCAM overlay for the CC image.
    Multi-view:  GradCAM overlays for both CC and MLO.
    Multi-seg:   U-Net segmentation masks for both views (malignant cases only).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mtype = get_model(model_name, device)

    cc_img    = load_dicom_bytes(cc_bytes)
    cc_tensor = torch.from_numpy(cc_img).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,512,512)

    # ── Single-view ──────────────────────────────────────────────────────────
    if mtype == "single":
        model.eval()
        # Enable gradients on parameters so Grad-CAM can backpropagate
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

    # ── Multi-view (GradCAM) ─────────────────────────────────────────────────
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

        # retain=True for the CC backward because the graph is still needed for MLO
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

    # ── Multi-view with segmentation heads ───────────────────────────────────
    if mtype == "multi_seg":
        if mlo_bytes is None:
            raise ValueError("Multi-view model requires an MLO DICOM file.")
        mlo_img    = load_dicom_bytes(mlo_bytes)
        mlo_tensor = torch.from_numpy(mlo_img).unsqueeze(0).unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            # Forward pass on the original orientation
            logits, cc_mask, mlo_mask = model(cc_tensor, mlo_tensor, return_masks=True)

            # Test-time augmentation: average with horizontally flipped prediction.
            # Flipping smooths asymmetric noise in the mask without requiring
            # any additional training or architectural changes.
            cc_flip  = torch.flip(cc_tensor,  dims=[3])
            mlo_flip = torch.flip(mlo_tensor, dims=[3])
            _, cc_mask_flip, mlo_mask_flip = model(cc_flip, mlo_flip, return_masks=True)
            cc_mask  = (cc_mask  + torch.flip(cc_mask_flip,  dims=[3])) / 2.0
            mlo_mask = (mlo_mask + torch.flip(mlo_mask_flip, dims=[3])) / 2.0

        prob = torch.sigmoid(logits).item()

        # Logit gating: scale the mask intensity by how confident the model is that
        # the case is malignant. At 50% confidence the gate is 0 (mask hidden);
        # at 100% confidence the gate is 1 (full mask shown). This prevents noisy,
        # low-confidence masks from being displayed as if they are reliable.
        gate        = min(max((prob - 0.5) / 0.5, 0.0), 1.0)
        cc_mask_np  = cc_mask[0, 0].cpu().numpy()  * gate
        mlo_mask_np = mlo_mask[0, 0].cpu().numpy() * gate

        # Only show localisation overlays for malignant predictions — for benign
        # cases the segmentation head output carries no clinical meaning
        is_malignant = prob >= 0.5
        result = {
            "diagnosis":    _diagnosis_text(prob),
            "probability":  round(prob, 4),
            "model_type":   "multi_seg",
            "original_cc":  gray_to_base64(cc_img),
            "original_mlo": gray_to_base64(mlo_img),
        }
        if is_malignant:
            result["seg_mask_cc"]  = mask_overlay_base64(cc_img,  cc_mask_np)
            result["seg_mask_mlo"] = mask_overlay_base64(mlo_img, mlo_mask_np)
        return result

    raise ValueError(f"Unknown model type: {mtype}")
