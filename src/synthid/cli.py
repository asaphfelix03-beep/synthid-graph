"""Point d'entrée : `synthid run --config configs/small.yaml` (ou `python -m synthid.cli ...`)."""
from __future__ import annotations

import argparse
import warnings

from . import pipeline as pl
from .config import load_config
from .report import build_report

STAGES = ["generate", "features", "privacy", "train", "evaluate", "alerts", "neo4j", "report"]


def main(argv: list[str] | None = None):
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    p = argparse.ArgumentParser(prog="synthid", description="PoC détection d'identités synthétiques par graphes")
    p.add_argument("stage", choices=["run", *STAGES], help="étape à exécuter ('run' = toutes)")
    p.add_argument("--config", default="configs/small.yaml")
    p.add_argument("--skip-neo4j", action="store_true", help="ne pas charger le graphe dans Neo4j")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    todo = STAGES if args.stage == "run" else [args.stage]
    if args.skip_neo4j and "neo4j" in todo and args.stage == "run":
        todo.remove("neo4j")
    actions = {
        "generate": pl.stage_generate, "features": pl.stage_features, "privacy": pl.stage_privacy,
        "train": pl.stage_train, "evaluate": pl.stage_evaluate, "alerts": pl.stage_alerts, "neo4j": pl.stage_neo4j,
        "report": lambda c: pl.log(f"rapport : {build_report(c)}"),
    }
    for stage in todo:
        pl.log(f"=== {stage} ===")
        actions[stage](cfg)


if __name__ == "__main__":
    main()
