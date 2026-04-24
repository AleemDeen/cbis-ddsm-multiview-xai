"""
Exports a random selection of example DICOM pairs from the dataset into the
examples/ folder, ready to drag and drop into the frontend.

Clears the examples/ folder first, then copies the requested number of cases
with clean filenames:
    P_00001_LEFT_MALIGNANT_CC.dcm
    P_00001_LEFT_MALIGNANT_MLO.dcm

Tries to balance malignant and benign cases evenly. If one class does not
have enough cases to fill its half, the remaining slots are taken from the
other class.

Run from the project root:
    python export_examples.py
"""

import random
import shutil
from pathlib import Path

import pandas as pd

ROOT      = Path(__file__).parent
CSV_PATH  = ROOT / "data_processed" / "indexed_multi_view_cases_pt.csv"
OUT_DIR   = ROOT / "examples"


def main():
    if not CSV_PATH.exists():
        print(f"[ERROR] CSV not found: {CSV_PATH}")
        print("        Run the data pipeline first (see README).")
        return

    df = pd.read_csv(CSV_PATH)
    df["case_id"] = df["patient_id"].str.extract(r"(P_\d+)")

    # ── Ask how many cases ────────────────────────────────────────────────────
    total_available = len(df)
    print(f"\nDataset contains {total_available} paired cases "
          f"({df['label'].sum()} malignant, {(df['label'] == 0).sum()} benign).")

    while True:
        raw = input(f"\nHow many example cases would you like? [1–{total_available}]: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= total_available:
            n = int(raw)
            break
        print(f"  Please enter a number between 1 and {total_available}.")

    # ── Balanced random sampling ───────────────────────────────────────────────
    malignant = df[df["label"] == 1].sample(frac=1, random_state=None)
    benign    = df[df["label"] == 0].sample(frac=1, random_state=None)

    # Aim for an even split; odd number gives the extra slot to malignant
    n_mal = (n + 1) // 2
    n_ben = n // 2

    # If one class is short, fill remaining slots from the other
    if len(malignant) < n_mal:
        n_mal = len(malignant)
        n_ben = min(n - n_mal, len(benign))
    elif len(benign) < n_ben:
        n_ben = len(benign)
        n_mal = min(n - n_ben, len(malignant))

    selected = pd.concat([
        malignant.head(n_mal),
        benign.head(n_ben),
    ]).sample(frac=1)   # shuffle so malignant/benign aren't grouped

    actual = len(selected)
    print(f"\nSelected {actual} cases  "
          f"({n_mal} malignant, {n_ben} benign) — randomised.")

    # ── Clear examples/ folder ────────────────────────────────────────────────
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir()
    print(f"Cleared and recreated {OUT_DIR}/\n")

    # ── Copy DICOMs with clean filenames ──────────────────────────────────────
    copied = 0
    skipped = 0

    for _, row in selected.iterrows():
        case_id  = row["case_id"]
        lat      = str(row["laterality"]).upper()
        dx       = "MALIGNANT" if int(row["label"]) == 1 else "BENIGN"
        prefix   = f"{case_id}_{lat}_{dx}"

        print(f"{prefix}")

        for view, col in (("CC", "cc_path"), ("MLO", "mlo_path")):
            src = Path(row[col])
            dst = OUT_DIR / f"{prefix}_{view}.dcm"

            if not src.exists():
                print(f"  [SKIP] source not found: {src}")
                skipped += 1
                continue

            shutil.copy2(src, dst)
            print(f"  -> {dst.name}")
            copied += 1

        print()

    print(f"Done — {copied} DICOM(s) exported to {OUT_DIR}/")
    if skipped:
        print(f"       {skipped} file(s) skipped (source DICOM not found).")


if __name__ == "__main__":
    main()
