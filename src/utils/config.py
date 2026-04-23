from pathlib import Path
import yaml


def load_config(config_path: str = "configs/base.yaml") -> dict:
    """
    Load the project YAML config and return it as a plain dictionary.

    The path is resolved relative to the repository root (two levels above this
    file) so the function works regardless of the current working directory.
    A REPO_ROOT key is injected into the returned dict for convenience.
    """
    repo_root = Path(__file__).resolve().parents[2]
    cfg_path  = repo_root / config_path

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["REPO_ROOT"] = str(repo_root)
    return cfg
