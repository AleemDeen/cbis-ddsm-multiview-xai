from pathlib import Path
from src.utils.config import load_config
from src.utils.seed import set_seed

def main():
    cfg = load_config()
    set_seed(int(cfg["SEED"]))

    raw_dir = Path(cfg["RAW_DATA_DIR"])
    proc_dir = Path(cfg["PROCESSED_DATA_DIR"])

    print("RAW_DATA_DIR:", raw_dir)
    print("Exists:", raw_dir.exists())

    proc_dir.mkdir(parents=True, exist_ok=True)
    print("PROCESSED_DATA_DIR:", proc_dir.resolve())

    # Basic import sanity
    import numpy as np
    import pandas as pd
    import pydicom
    import torch
    print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())

    # Quick file listing (won't fail if dataset not downloaded yet)
    if raw_dir.exists():
        files = list(raw_dir.rglob("*"))
        print("Files found under RAW_DATA_DIR:", len(files))
        print("Example:", files[0] if files else "No files")

if __name__ == "__main__":
    main()
