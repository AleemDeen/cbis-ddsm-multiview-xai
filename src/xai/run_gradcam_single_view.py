import os
import sys
import torch
import numpy as np
import cv2
from pathlib import Path

from src.models.resnet18_single_view import ResNet18SingleView
from src.data.dataloaders import build_dataloaders
from src.xai.gradcam import GradCAM


def pick_model(default: str) -> str:
    """Interactively choose a .pt file from models/ when running in a terminal."""
    if not sys.stdin.isatty():
        return default
    candidates = sorted(Path("models").glob("*.pt"))
    if not candidates:
        return default
    print("\nAvailable models:")
    for i, p in enumerate(candidates, 1):
        marker = " (default)" if p.name == Path(default).name else ""
        print(f"  [{i}] {p.name}{marker}")
    print(f"  [Enter] use default ({Path(default).name})")
    choice = input("Select model: ").strip()
    if not choice:
        return default
    if choice.isdigit() and 1 <= int(choice) <= len(candidates):
        return str(candidates[int(choice) - 1])
    print(f"Invalid choice, using default: {Path(default).name}")
    return default


def overlay_heatmap(image, heatmap):
    """
    Blend a Grad-CAM heatmap onto a greyscale mammogram.

    Args:
        image:   (H, W) float array in [0, 1]
        heatmap: (H, W) float array in [0, 1]

    Returns:
        (H, W, 3) blended uint8 array
    """
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    # Stack the greyscale image into 3 channels so it can be blended with the coloured heatmap
    image_rgb = np.stack([image] * 3, axis=-1)
    overlay   = 0.6 * image_rgb + 0.4 * heatmap_color
    # Normalise to ensure the full [0, 1] range is used before converting to uint8
    overlay = overlay / overlay.max()
    return overlay


def main():
    model_path = pick_model("models/sv_best.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Model:        {model_path}")

    out_dir = "results/gradcam"
    os.makedirs(out_dir, exist_ok=True)

    # Load the test split only — Grad-CAM should never be evaluated on
    # images seen during training as this would not reflect generalisation
    _, _, test_loader = build_dataloaders(
        csv_path="data_processed/indexed_full_mammogram_images_with_labels.csv",
        splits_dir="splits",
        batch_size=1,         # batch size of 1 required for per-image Grad-CAM
        num_workers=2,
        pin_memory=True,
    )

    model = ResNet18SingleView().to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()

    # Gradients must be enabled for the backward pass inside Grad-CAM
    for p in model.parameters():
        p.requires_grad_(True)

    # Attach Grad-CAM to layer4 — the deepest spatial feature map, which
    # captures high-level semantic content whilst retaining some spatial resolution
    cam = GradCAM(model, model.model.layer4)

    saved = 0

    for i, batch in enumerate(test_loader):
        image = batch["image"].to(device)
        label = int(batch["label"].item())
        pid   = batch["patient_id"][0]

        print(f"Grad-CAM on {pid} | label={label}")

        heatmap = cam.generate(image)
        heatmap = heatmap.squeeze().detach().cpu().numpy()

        # Bring the image tensor back to a numpy array for visualisation
        img_np = image.squeeze().detach().cpu().numpy()
        img_np -= img_np.min()
        img_np /= (img_np.max() + 1e-8)

        # Upsample the coarse Grad-CAM output (16×16) to the full image resolution
        heatmap = cv2.resize(
            heatmap,
            (img_np.shape[1], img_np.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

        overlay   = overlay_heatmap(img_np, heatmap)
        out_path  = os.path.join(out_dir, f"{pid}_label{label}.png")
        cv2.imwrite(out_path, np.uint8(255 * overlay))

        saved += 1
        if saved >= 20:
            break

    print(f"\nSaved {saved} Grad-CAM images to {out_dir}/")


if __name__ == "__main__":
    main()
