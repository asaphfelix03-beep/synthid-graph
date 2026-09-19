from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_path"] = str(path)
    return cfg


def data_dir(cfg: dict) -> Path:
    """Données du PoC, sous le répertoire courant : `data/<config>/vault/` (vérité terrain + PII en clair),
    `bank/` (tables tokenisées), `features/`, `results/`."""
    d = Path.cwd() / "data" / cfg["name"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def report_dir(cfg: dict) -> Path:
    d = Path.cwd() / "reports" / cfg["name"]
    d.mkdir(parents=True, exist_ok=True)
    return d
