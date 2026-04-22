"""
Architecture diagram generator for all 4 selected models.

Run from project root:
    python generate_architectures.py

Saves PNGs to architectures/.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path("architectures")
OUT.mkdir(exist_ok=True)

# ── Colour palette ────────────────────────────────────────────────────────────
C = {
    "input":    "#2C3E50",   # dark blue-grey
    "stem":     "#1A5276",   # dark blue
    "resblock": "#1F618D",   # medium blue
    "pool":     "#117A65",   # teal
    "dropout":  "#6C3483",   # purple
    "fc":       "#922B21",   # red
    "concat":   "#7D6608",   # gold
    "decoder":  "#1E8449",   # green
    "output":   "#2C3E50",   # dark
    "mask_out": "#117A65",   # teal
    "arrow":    "#555555",
    "text_light": "white",
    "text_dark":  "#2C3E50",
}


# ── Drawing primitives ────────────────────────────────────────────────────────

def box(ax, x, y, w, h, colour, label, sublabel="", fontsize=9, radius=0.015):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle=f"round,pad=0,rounding_size={radius}",
                          linewidth=1.2, edgecolor="white",
                          facecolor=colour, zorder=3)
    ax.add_patch(rect)
    if sublabel:
        ax.text(x, y + h * 0.12, label, ha="center", va="center",
                fontsize=fontsize, color=C["text_light"], fontweight="bold", zorder=4)
        ax.text(x, y - h * 0.22, sublabel, ha="center", va="center",
                fontsize=fontsize - 1.5, color=C["text_light"], alpha=0.85, zorder=4)
    else:
        ax.text(x, y, label, ha="center", va="center",
                fontsize=fontsize, color=C["text_light"], fontweight="bold", zorder=4)


def arrow(ax, x1, y1, x2, y2, label=""):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=C["arrow"],
                                lw=1.4, mutation_scale=12), zorder=2)
    if label:
        mx, my = (x1 + x2) / 2 + 0.015, (y1 + y2) / 2
        ax.text(mx, my, label, ha="left", va="center",
                fontsize=7, color="#555555", style="italic")


def bracket_arrow(ax, x1, y1, x2, y2, via_x, colour="#555555"):
    """L-shaped arrow for skip connections."""
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle="-|>", color=colour, lw=1.1,
                    connectionstyle=f"arc3,rad=0.0",
                    mutation_scale=10,
                ), zorder=2)


def save(fig, name):
    p = OUT / name
    fig.savefig(p, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  saved -> {p}")


# ── Shared ResNet18 block list ────────────────────────────────────────────────

SV_LAYERS = [
    ("input",    "Input\nDICOM",           "1 × 512 × 512"),
    ("stem",     "Conv 7×7  BN  ReLU",    "64 × 256 × 256"),
    ("stem",     "MaxPool 3×3",            "64 × 128 × 128"),
    ("resblock", "Layer 1\n2× BasicBlock", "64 × 128 × 128"),
    ("resblock", "Layer 2\n2× BasicBlock", "128 × 64 × 64"),
    ("resblock", "Layer 3\n2× BasicBlock", "256 × 32 × 32"),
    ("resblock", "Layer 4\n2× BasicBlock", "512 × 16 × 16"),
    ("pool",     "Global AvgPool",         "512"),
    ("dropout",  "Dropout  p = 0.3",       ""),
    ("fc",       "Linear 512 → 1",         ""),
    ("output",   "Sigmoid\nMalignancy Score", ""),
]


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Single-View Baseline
# ═══════════════════════════════════════════════════════════════════════════════

def draw_single_view(title, filename, note=""):
    fig, ax = plt.subplots(figsize=(4.5, 13))
    fig.patch.set_facecolor("#F4F6F7")
    ax.set_facecolor("#F4F6F7")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.985, title, ha="center", va="top", fontsize=12,
            fontweight="bold", color=C["text_dark"])
    if note:
        ax.text(0.5, 0.965, note, ha="center", va="top", fontsize=8,
                color="#7F8C8D", style="italic")

    n      = len(SV_LAYERS)
    top    = 0.935
    bottom = 0.025
    step   = (top - bottom) / n
    bw, bh = 0.72, step * 0.72

    centres = [top - i * step - step / 2 for i in range(n)]

    for i, (kind, label, dim) in enumerate(SV_LAYERS):
        cy = centres[i]
        box(ax, 0.5, cy, bw, bh, C[kind], label, dim, fontsize=8.5)
        if i < n - 1:
            arrow(ax, 0.5, cy - bh/2, 0.5, centres[i+1] + bh/2)

    fig.tight_layout(pad=0.3)
    save(fig, filename)


draw_single_view(
    "Single-View Baseline",
    "sv_baseline_architecture.png",
    note="ResNet18 — ImageNet pretrained | 1-channel grayscale input",
)
draw_single_view(
    "Single-View Best  (λ-loc = 0.0)",
    "sv_best_architecture.png",
    note="Same architecture as Baseline — trained with classification loss only",
)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Multi-View Baseline  (dual ResNet18, concat, classify)
# ═══════════════════════════════════════════════════════════════════════════════

def draw_mv_baseline():
    fig, ax = plt.subplots(figsize=(9, 12))
    fig.patch.set_facecolor("#F4F6F7")
    ax.set_facecolor("#F4F6F7")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.987, "Multi-View Baseline", ha="center", va="top",
            fontsize=13, fontweight="bold", color=C["text_dark"])
    ax.text(0.5, 0.967, "Dual ResNet18 branches  —  features concatenated  —  shared classifier",
            ha="center", va="top", fontsize=8.5, color="#7F8C8D", style="italic")

    BRANCH_LAYERS = [
        ("stem",     "Conv 7×7  BN  ReLU",    "64×256×256"),
        ("stem",     "MaxPool 3×3",            "64×128×128"),
        ("resblock", "Layer 1\n2× BasicBlock", "64×128×128"),
        ("resblock", "Layer 2\n2× BasicBlock", "128×64×64"),
        ("resblock", "Layer 3\n2× BasicBlock", "256×32×32"),
        ("resblock", "Layer 4\n2× BasicBlock", "512×16×16"),
        ("pool",     "Global AvgPool",         "512-d"),
    ]

    n    = len(BRANCH_LAYERS)
    top  = 0.94
    bot  = 0.38
    step = (top - bot) / n
    bw   = 0.36
    bh   = step * 0.70

    cx_cc  = 0.22
    cx_mlo = 0.78

    # Input boxes
    for cx, lbl in [(cx_cc, "CC View\n1×512×512"), (cx_mlo, "MLO View\n1×512×512")]:
        box(ax, cx, top + step * 0.5, bw, bh * 1.1, C["input"], lbl, fontsize=9)

    for i, (kind, label, dim) in enumerate(BRANCH_LAYERS):
        cy = top - i * step - step / 2
        for cx in (cx_cc, cx_mlo):
            box(ax, cx, cy, bw, bh, C[kind], label, dim, fontsize=8)
            if i == 0:
                arrow(ax, cx, top + step*0.5 - bh*0.55, cx, cy + bh/2)
            else:
                prev_cy = top - (i-1)*step - step/2
                arrow(ax, cx, prev_cy - bh/2, cx, cy + bh/2)

    # Concat
    last_cy = top - (n-1)*step - step/2
    concat_y = bot - 0.04
    box(ax, 0.5, concat_y, 0.45, bh * 1.15, C["concat"],
        "Concatenate", "512 + 512 = 1024-d", fontsize=9)
    for cx in (cx_cc, cx_mlo):
        ax.annotate("", xy=(0.5, concat_y + bh*0.575),
                    xytext=(cx, last_cy - bh/2),
                    arrowprops=dict(arrowstyle="-|>", color=C["arrow"],
                                    lw=1.3, mutation_scale=11,
                                    connectionstyle="arc3,rad=0.0"), zorder=2)

    # Classifier
    drop_y = concat_y - 0.065
    fc_y   = drop_y  - 0.065
    out_y  = fc_y    - 0.075

    box(ax, 0.5, drop_y, 0.38, bh, C["dropout"], "Dropout  p = 0.4", fontsize=9)
    box(ax, 0.5, fc_y,   0.38, bh, C["fc"],      "Linear 1024 → 1", fontsize=9)
    box(ax, 0.5, out_y,  0.38, bh * 1.1, C["output"],
        "Sigmoid\nMalignancy Score", fontsize=9)

    for y1, y2 in [(concat_y - bh*0.575, drop_y + bh/2),
                   (drop_y - bh/2,  fc_y + bh/2),
                   (fc_y   - bh/2,  out_y + bh*0.55)]:
        arrow(ax, 0.5, y1, 0.5, y2)

    # Branch labels
    for cx, lbl in [(cx_cc, "CC Branch"), (cx_mlo, "MLO Branch")]:
        ax.text(cx, top + step * 0.5 + bh * 0.7, lbl, ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=C["text_dark"])

    fig.tight_layout(pad=0.3)
    save(fig, "mv_baseline_architecture.png")


draw_mv_baseline()


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Multi-View Best  (dual ResNet18 + U-Net seg decoders)
# ═══════════════════════════════════════════════════════════════════════════════

def draw_mv_best():
    fig, ax = plt.subplots(figsize=(12, 15))
    fig.patch.set_facecolor("#F4F6F7")
    ax.set_facecolor("#F4F6F7")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.990, "Multi-View Best  (U-Net Seg Head)", ha="center", va="top",
            fontsize=13, fontweight="bold", color=C["text_dark"])
    ax.text(0.5, 0.973,
            "Dual ResNet18 encoder  +  U-Net decoder per view  "
            "→  classification logit  +  2 ROI probability masks",
            ha="center", va="top", fontsize=8.5, color="#7F8C8D", style="italic")

    # ── Backbone layers (with skip-connection points) ─────────────────────────
    BACKBONE = [
        ("stem",     "Conv 7×7  BN  ReLU",     "64×256×256",  None),
        ("stem",     "MaxPool 3×3",             "64×128×128",  None),
        ("resblock", "Layer 1  (f1)\n2× Block", "64×128×128",  "f1"),
        ("resblock", "Layer 2  (f2)\n2× Block", "128×64×64",   "f2"),
        ("resblock", "Layer 3  (f3)\n2× Block", "256×32×32",   "f3"),
        ("resblock", "Layer 4  (f4)\n2× Block", "512×16×16",   "f4"),
        ("pool",     "Global AvgPool",           "512-d",       None),
    ]

    # ── Decoder layers ────────────────────────────────────────────────────────
    DECODER = [
        ("decoder", "Up + Conv\n(f4 → 256)",   "256×32×32"),
        ("decoder", "Fuse f3\nConv 256",        "256×32×32"),
        ("decoder", "Up + Conv\n(→ 128)",       "128×64×64"),
        ("decoder", "Fuse f2\nConv 128",        "128×64×64"),
        ("decoder", "Up + Conv\n(→ 64)",        "64×128×128"),
        ("decoder", "Fuse f1\nConv 64",         "64×128×128"),
        ("decoder", "Up × 4 → Conv\nSigmoid",  "1×512×512"),
    ]

    n_back = len(BACKBONE)
    top    = 0.945
    bot    = 0.44
    step   = (top - bot) / n_back
    bh     = step * 0.68
    bw_enc = 0.26
    bw_dec = 0.20

    cx_cc      = 0.20   # CC encoder
    cx_mlo     = 0.80   # MLO encoder
    cx_cc_dec  = 0.40   # CC decoder (right of CC encoder)
    cx_mlo_dec = 0.60   # MLO decoder (left of MLO encoder)

    skip_colour = "#E67E22"

    # ── Draw both encoders ────────────────────────────────────────────────────
    enc_centres = {}
    for col_cx, col_name in [(cx_cc, "cc"), (cx_mlo, "mlo")]:
        # Input
        inp_y = top + step * 0.55
        box(ax, col_cx, inp_y, bw_enc, bh * 1.05, C["input"],
            f"{'CC' if col_name=='cc' else 'MLO'} View\n1×512×512", fontsize=8.5)

        for i, (kind, label, dim, skip_tag) in enumerate(BACKBONE):
            cy = top - i * step - step / 2
            enc_centres[(col_name, i)] = cy
            box(ax, col_cx, cy, bw_enc, bh, C[kind], label, dim, fontsize=7.5)
            if i == 0:
                arrow(ax, col_cx, inp_y - bh * 0.525, col_cx, cy + bh / 2)
            else:
                arrow(ax, col_cx, enc_centres[(col_name, i-1)] - bh/2, col_cx, cy + bh/2)

    # ── Draw both decoders ────────────────────────────────────────────────────
    n_dec     = len(DECODER)
    dec_top   = top - 2 * step - step / 2 - bh / 2 - 0.01   # start just below f1
    dec_step  = (dec_top - (bot - 0.02)) / n_dec
    dec_bh    = dec_step * 0.68

    dec_centres = {}
    for col_cx, col_name, enc_cx in [
        (cx_cc_dec,  "cc",  cx_cc),
        (cx_mlo_dec, "mlo", cx_mlo),
    ]:
        for i, (kind, label, dim) in enumerate(DECODER):
            cy = dec_top - i * dec_step - dec_step / 2
            dec_centres[(col_name, i)] = cy
            box(ax, col_cx, cy, bw_dec, dec_bh, C[kind], label, dim, fontsize=7.2)
            if i == 0:
                # First decoder box receives f4 from encoder layer 5 (index 5)
                f4_cy = enc_centres[(col_name, 5)]
                # Draw curved skip from encoder to decoder
                mid_x = (enc_cx + col_cx) / 2
                ax.annotate("", xy=(col_cx - bw_dec/2, cy),
                            xytext=(enc_cx, f4_cy),
                            arrowprops=dict(
                                arrowstyle="-|>", color=skip_colour, lw=1.2,
                                connectionstyle="arc3,rad=-0.25" if col_name=="mlo" else "arc3,rad=0.25",
                                mutation_scale=10), zorder=2)
                ax.text((enc_cx + col_cx) / 2,
                        (f4_cy + cy) / 2 + 0.01,
                        "f4", ha="center", fontsize=7,
                        color=skip_colour, fontweight="bold")
            else:
                arrow(ax, col_cx, dec_centres[(col_name, i-1)] - dec_bh/2,
                      col_cx, cy + dec_bh/2)

    # Skip connections f3, f2, f1 (decoder fuse layers = indices 1, 3, 5)
    SKIP_MAP = {
        "f3": (4, 1),   # encoder layer index 4 → decoder fuse index 1
        "f2": (3, 3),   # encoder layer 3 → decoder fuse 3
        "f1": (2, 5),   # encoder layer 2 → decoder fuse 5
    }
    for tag, (enc_i, dec_i) in SKIP_MAP.items():
        for col_name, enc_cx, dec_cx in [
            ("cc",  cx_cc,  cx_cc_dec),
            ("mlo", cx_mlo, cx_mlo_dec),
        ]:
            ey = enc_centres[(col_name, enc_i)]
            dy = dec_centres[(col_name, dec_i)]
            rad = 0.2 if col_name == "cc" else -0.2
            ax.annotate("", xy=(dec_cx - bw_dec/2 if col_name=="cc" else dec_cx + bw_dec/2, dy),
                        xytext=(enc_cx, ey),
                        arrowprops=dict(
                            arrowstyle="-|>", color=skip_colour, lw=1.0,
                            connectionstyle=f"arc3,rad={rad}",
                            mutation_scale=9), zorder=2)
            lx = (enc_cx + dec_cx) / 2
            ax.text(lx, (ey + dy) / 2, tag, ha="center", fontsize=6.5,
                    color=skip_colour, fontweight="bold")

    # ── Mask outputs ──────────────────────────────────────────────────────────
    last_dec_y = dec_centres[("cc", n_dec - 1)]
    mask_y     = last_dec_y - dec_bh * 0.8

    for col_cx, col_name, lbl in [
        (cx_cc_dec,  "cc",  "CC ROI Mask\n1×512×512"),
        (cx_mlo_dec, "mlo", "MLO ROI Mask\n1×512×512"),
    ]:
        box(ax, col_cx, mask_y, bw_dec + 0.02, dec_bh * 1.1,
            C["mask_out"], lbl, fontsize=8)
        arrow(ax, col_cx, dec_centres[(col_name, n_dec-1)] - dec_bh/2,
              col_cx, mask_y + dec_bh * 0.55)

    # ── Classifier (centre, below AvgPool) ───────────────────────────────────
    avg_cc_y  = enc_centres[("cc",  6)]
    avg_mlo_y = enc_centres[("mlo", 6)]
    concat_y  = min(avg_cc_y, avg_mlo_y) - step * 0.75
    drop_y    = concat_y - 0.048
    fc_y      = drop_y   - 0.048
    cls_y     = fc_y     - 0.055

    box(ax, 0.5, concat_y, 0.30, bh * 0.85, C["concat"],
        "Concatenate", "1024-d", fontsize=8.5)
    box(ax, 0.5, drop_y,   0.28, bh * 0.78, C["dropout"], "Dropout  p=0.4", fontsize=8.5)
    box(ax, 0.5, fc_y,     0.28, bh * 0.78, C["fc"],      "Linear 1024→1",  fontsize=8.5)
    box(ax, 0.5, cls_y,    0.28, bh * 0.90, C["output"],
        "Sigmoid\nMalignancy Score", fontsize=8.5)

    for y1, y2 in [
        (drop_y - bh*0.39, fc_y + bh*0.39),
        (fc_y   - bh*0.39, cls_y + bh*0.45),
    ]:
        arrow(ax, 0.5, y1, 0.5, y2)

    for enc_cx, col_name in [(cx_cc, "cc"), (cx_mlo, "mlo")]:
        avg_y = enc_centres[(col_name, 6)]
        ax.annotate("", xy=(0.5, concat_y + bh * 0.42),
                    xytext=(enc_cx, avg_y - bh / 2),
                    arrowprops=dict(
                        arrowstyle="-|>", color=C["arrow"], lw=1.3,
                        mutation_scale=11,
                        connectionstyle="arc3,rad=0.0"), zorder=2)

    for concat_from, concat_to in [
        (concat_y - bh*0.42, drop_y + bh*0.39),
    ]:
        arrow(ax, 0.5, concat_from, 0.5, concat_to)

    # ── Column headers ────────────────────────────────────────────────────────
    header_y = top + step * 0.55 + bh * 0.75
    for cx, lbl in [
        (cx_cc,      "CC Encoder"),
        (cx_cc_dec,  "CC Decoder"),
        (0.5,        "Classifier"),
        (cx_mlo_dec, "MLO Decoder"),
        (cx_mlo,     "MLO Encoder"),
    ]:
        ax.text(cx, header_y, lbl, ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=C["text_dark"])

    # ── Skip connection legend ────────────────────────────────────────────────
    lx, ly = 0.01, 0.04
    ax.plot([lx, lx+0.04], [ly, ly], color=skip_colour, lw=1.5)
    ax.annotate("", xy=(lx+0.04, ly), xytext=(lx+0.036, ly),
                arrowprops=dict(arrowstyle="-|>", color=skip_colour,
                                lw=1.0, mutation_scale=8))
    ax.text(lx+0.05, ly, "Skip connection (f1–f4)", va="center",
            fontsize=7.5, color=skip_colour)

    fig.tight_layout(pad=0.3)
    save(fig, "mv_best_architecture.png")


draw_mv_best()


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Overview comparison panel
# ═══════════════════════════════════════════════════════════════════════════════

def draw_overview():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.patch.set_facecolor("#F4F6F7")
    fig.suptitle("Model Architecture Overview", fontsize=14,
                 fontweight="bold", color=C["text_dark"], y=1.01)

    specs = [
        ("Single-View\n(Baseline & Best)",
         ["Input\n1×512×512", "ResNet18 Stem", "Layer1  64ch",
          "Layer2  128ch", "Layer3  256ch", "Layer4  512ch",
          "AvgPool  512-d", "Dropout 0.3", "Linear 512→1", "Output"],
         ["input","stem","resblock","resblock","resblock","resblock",
          "pool","dropout","fc","output"]),

        ("Multi-View Baseline",
         ["CC 1×512×512", "MLO 1×512×512",
          "ResNet18 × 2\n(independent)", "AvgPool × 2\n512-d each",
          "Concat  1024-d", "Dropout 0.4", "Linear 1024→1", "Output"],
         ["input","input","resblock","pool","concat","dropout","fc","output"]),

        ("Multi-View Best\n(+ U-Net Seg Head)",
         ["CC 1×512×512", "MLO 1×512×512",
          "ResNet18 × 2\n+ skip f1–f4", "AvgPool × 2\n512-d each",
          "Concat  1024-d", "Linear 1024→1",
          "U-Net Decoder × 2", "CC Mask\n1×512×512", "MLO Mask\n1×512×512"],
         ["input","input","resblock","pool","concat","fc",
          "decoder","mask_out","mask_out"]),
    ]

    for ax, (title, labels, kinds) in zip(axes, specs):
        ax.set_facecolor("#F4F6F7")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title(title, fontsize=10, fontweight="bold",
                     color=C["text_dark"], pad=6)

        n    = len(labels)
        step = 0.90 / n
        top  = 0.95
        bw, bh = 0.78, step * 0.72

        for i, (lbl, kind) in enumerate(zip(labels, kinds)):
            cy = top - i * step - step / 2
            box(ax, 0.5, cy, bw, bh, C[kind], lbl, fontsize=8)
            if i < n - 1:
                arrow(ax, 0.5, cy - bh/2, 0.5, top - (i+1)*step - step/2 + bh/2)

    fig.tight_layout(pad=0.5)
    save(fig, "overview_comparison.png")


draw_overview()

print(f"\nDone — {len(list(OUT.glob('*.png')))} PNG(s) saved to {OUT}/")
