from pathlib import Path
import yaml

def load_config(config_path: str = "configs/base.yaml") -> dict:
    """
    Load YAML config and return it as a dict.
    Supports relative paths (relative to repo root).
    """
    repo_root = Path(__file__).resolve().parents[2]  # .../Project setup/
    cfg_path = repo_root / config_path

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Add repo root for convenience
    cfg["REPO_ROOT"] = str(repo_root)

    return cfg
