"""Orchestration des étapes du PoC (CRISP-DM) : generate → features → privacy → train → alerts → report."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .bank_view import build_bank_view, build_labels
from .config import data_dir, report_dir
from .evaluation import build_rows, community_quality, confounder_rates, ranking_metrics, ring_metrics, top_k_flags
from .features import compute_features, feature_columns, take_snapshot
from .models import lgbm_contributions, rules_score, train_lgbm
from .simulate import simulate

BANK_TABLES = ["persons", "person_attr", "attr_meta", "paid_at", "merchant_meta", "transfers", "authorized_users",
               "similar", "monthly", "fraud_confirmations"]


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def snapshots(cfg: dict) -> list[int]:
    ev = cfg["evaluation"]
    return sorted(set(ev["train_snapshots"]) | set(ev["test_snapshots"]))


# ------------------------------------------------------------------- I/O
def save_tables(tables: dict[str, pd.DataFrame], folder: Path):
    folder.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_parquet(folder / f"{name}.parquet", index=False)


def load_tables(folder: Path, names: list[str] | None = None) -> dict[str, pd.DataFrame]:
    files = [folder / f"{n}.parquet" for n in names] if names else sorted(folder.glob("*.parquet"))
    return {f.stem: pd.read_parquet(f) for f in files}


def load_bank(cfg: dict) -> dict[str, pd.DataFrame]:
    return load_tables(data_dir(cfg) / "bank", BANK_TABLES)


def load_labels(cfg: dict) -> pd.DataFrame:
    return pd.read_parquet(data_dir(cfg) / "vault" / "labels.parquet")


def load_features(cfg: dict) -> pd.DataFrame:
    d = data_dir(cfg) / "features"
    frames = []
    for T in snapshots(cfg):
        f = pd.read_parquet(d / f"features_T{T}.parquet")
        xb = d / f"xbank_T{T}.parquet"
        if xb.exists():
            f = f.merge(pd.read_parquet(xb), on="person_id", how="left")
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------- étapes
def stage_generate(cfg: dict) -> dict:
    t = time.perf_counter()
    sim = simulate(cfg)
    bank = build_bank_view(sim, cfg)
    labels = build_labels(sim)
    d = data_dir(cfg)
    save_tables({**sim, "labels": labels}, d / "vault")
    save_tables(bank, d / "bank")
    stats = {
        "identities": int(len(sim["identities"])),
        "bank_records": int(len(sim["records"])),
        "records_by_bank": sim["records"]["bank"].value_counts().to_dict(),
        "synthetic_records": int(labels["is_synthetic"].sum()),
        "rings": int(len(sim["rings"])),
        "rings_by_level": sim["rings"]["sophistication"].value_counts().sort_index().to_dict(),
        "rings_by_scenario": sim["rings"]["scenario"].value_counts().to_dict(),
        "cross_bank_rings": int(sim["rings"]["cross_bank"].sum()),
        "person_attribute_edges": int(len(bank["person_attr"])),
        "merchant_edges": int(len(bank["paid_at"])),
        "transfer_rows": int(len(bank["transfers"])),
        "monthly_rows": int(len(bank["monthly"])),
        "entity_resolution_edges": int(len(bank["similar"])),
        "fraud_confirmations": int(len(bank["fraud_confirmations"])),
        "seconds": round(time.perf_counter() - t, 1),
    }
    (d / "dataset_stats.json").write_text(json.dumps(stats, indent=2, default=int), encoding="utf-8")
    log(f"generate : {stats['bank_records']} comptes, {stats['synthetic_records']} synthétiques, {stats['rings']} anneaux")
    return stats


def stage_features(cfg: dict):
    bank = load_bank(cfg)
    out = data_dir(cfg) / "features"
    out.mkdir(exist_ok=True)
    for T in snapshots(cfg):
        t = time.perf_counter()
        snap = take_snapshot(bank, T)
        feats, P, _ = compute_features(snap, cfg)
        feats.to_parquet(out / f"features_T{T}.parquet", index=False)
        sp.save_npz(out / f"projection_T{T}.npz", P)
        log(f"features T={T} : {len(feats)} personnes, projection {P.nnz} arêtes ({time.perf_counter() - t:.1f}s)")


def shortcut_audit(rows: pd.DataFrame, cols: list[str], threshold: float = 0.9) -> dict:
    """Jalon de la phase 2 : aucune feature isolée ne doit séparer les classes (AUC > seuil) sur l'entraînement."""
    from sklearn.metrics import roc_auc_score
    tr = rows[rows["split"].isin(["train", "val"])]
    res = {}
    for c in cols:
        x = tr[c]
        if x.nunique(dropna=True) < 2:
            continue
        a = roc_auc_score(tr["y"], x.fillna(x.median()))
        res[c] = round(float(max(a, 1 - a)), 4)
    top = dict(sorted(res.items(), key=lambda kv: -kv[1])[:10])
    return {"threshold": threshold, "max_auc": max(res.values()), "passed": max(res.values()) <= threshold, "top10": top}


def stage_train(cfg: dict) -> dict:
    try:
        from . import gnn as G  # import local : torch est lourd et optionnel (extra « gnn »)
    except ImportError:
        G = None

    labels = load_labels(cfg)
    feats = load_features(cfg)
    rows = build_rows(feats, labels, cfg)
    rows["pos"] = rows.groupby("T").cumcount()
    tab = feature_columns(rows, ("app_", "beh_"))
    graph = tab + feature_columns(rows, ("g_",))
    xb = graph + feature_columns(rows, ("xb_",))
    train, val, test = (rows[rows["split"] == s] for s in ("train", "val", "test"))
    log(f"train : {len(train)} lignes ({int(train['y'].sum())} positifs), val {len(val)} ({int(val['y'].sum())}), "
        f"test {len(test)} ({int(test['y'].sum())})")
    seed = cfg["evaluation"]["seeds"][0]
    scores = test[["person_id", "T", "y", "is_synthetic", "ring_id", "sophistication", "scenario", "bust_out_day",
                   "exposure", "community", "pos"]].copy()
    info: dict = {"rows": {s: {"n": int(len(d)), "positives": int(d["y"].sum())} for s, d in
                           (("train", train), ("val", val), ("test", test))}}
    info["shortcut_audit"] = shortcut_audit(rows, graph)

    scores["rules"] = rules_score(test)
    lgbm_sets = {"lgbm_tab": tab, "lgbm_graph": graph}
    if len(xb) > len(graph):
        lgbm_sets["lgbm_xb"] = xb
    models = {}
    for name, cols in lgbm_sets.items():
        t = time.perf_counter()
        m = train_lgbm(train, val, cols, seed)
        scores[name] = m.predict_proba(test[cols])[:, 1]
        models[name] = m
        info[name] = {"best_iteration": int(m.best_iteration_ or 0), "seconds": round(time.perf_counter() - t, 1)}
        log(f"{name} entraîné ({info[name]['seconds']}s)")
    imp = pd.Series(models["lgbm_graph"].booster_.feature_importance("gain"), index=graph)
    info["lgbm_graph_top_features"] = (imp / imp.sum()).sort_values(ascending=False).head(15).round(4).to_dict()

    if G is None:
        log('PyTorch Geometric absent : modèles GNN ignorés (pip install -e ".[gnn]")')
    else:
        _train_gnns(G, cfg, rows, train, tab, graph, scores, info, seed)

    # explications locales pour les dossiers d'alerte
    best_lgbm = "lgbm_xb" if "lgbm_xb" in models else "lgbm_graph"
    expl_cols = lgbm_sets[best_lgbm]
    scores["explanation"] = [json.dumps(e) for e in lgbm_contributions(models[best_lgbm], test[expl_cols])]

    out = data_dir(cfg) / "results"
    out.mkdir(exist_ok=True)
    scores.to_parquet(out / "scores.parquet")
    (out / "train_info.json").write_text(json.dumps(info, indent=2, default=float), encoding="utf-8")
    return info


def _train_gnns(G, cfg: dict, rows: pd.DataFrame, train: pd.DataFrame, tab: list[str], graph: list[str],
                scores: pd.DataFrame, info: dict, seed: int):
    """GCN et GraphSAGE hétérogène sur tous les snapshots, puis ensembles (rangs moyens) avec LightGBM."""
    bank = load_bank(cfg)
    fdir = data_dir(cfg) / "features"
    scalers = {"tab": G.FeatureScaler().fit(train[tab]), "graph": G.FeatureScaler().fit(train[graph])}
    graphs: dict[tuple, dict] = {}
    for T in snapshots(cfg):
        snap = take_snapshot(bank, T)
        ft = rows[rows["T"] == T]
        assert (ft["person_id"].to_numpy() == snap.persons["person_id"].to_numpy()).all()
        known = snap.persons["known_fraud"].to_numpy(dtype=float)
        x_tab, x_graph = scalers["tab"].transform(ft[tab]), scalers["graph"].transform(ft[graph])
        P = sp.load_npz(fdir / f"projection_T{T}.npz")
        cap = cfg["gnn"]["merchant_degree_cap"]
        built = {"gcn_tab": G.build_homogeneous(P, x_tab, known), "hsage_tab": G.build_hetero(snap, x_tab, cap),
                 "hsage_graph": G.build_hetero(snap, x_graph, cap)}
        for split in ("train", "val", "test"):
            part = ft[ft["split"] == split]
            if len(part):
                for kind, data in built.items():
                    graphs[(kind, T, split)] = {"data": data, "rows": part["pos"].to_numpy(), "y": part["y"].to_numpy(dtype=float),
                                                "index": part.index}
    for kind in ("gcn_tab", "hsage_tab", "hsage_graph"):
        t = time.perf_counter()
        tr_g = [g for (k, _, s), g in graphs.items() if k == kind and s == "train"]
        va_g = [g for (k, _, s), g in graphs.items() if k == kind and s == "val"]
        model, meta = G.train_gnn("gcn" if kind.startswith("gcn") else "hetero", tr_g, va_g, cfg, seed)
        for (k, _, s), g in graphs.items():
            if k == kind and s == "test":
                scores.loc[g["index"], kind] = G.predict(model, g)
        info[kind] = {**meta, "seconds": round(time.perf_counter() - t, 1)}
        log(f"{kind} entraîné : {meta['epochs']} époques, PR-AUC val {meta['best_val_pr_auc']:.3f} ({info[kind]['seconds']}s)")

    ranks = scores.groupby("T")[["lgbm_graph", "hsage_graph"]].rank(pct=True)
    scores["ensemble"] = ranks.mean(axis=1)
    if "lgbm_xb" in scores:
        scores["ensemble_xb"] = pd.concat([scores.groupby("T")["lgbm_xb"].rank(pct=True), ranks["hsage_graph"]], axis=1).mean(axis=1)


def model_columns(scores: pd.DataFrame) -> list[str]:
    base = ["rules", "lgbm_tab", "lgbm_graph", "gcn_tab", "hsage_tab", "hsage_graph", "ensemble", "lgbm_xb", "ensemble_xb"]
    return [c for c in base if c in scores.columns]


def stage_evaluate(cfg: dict) -> dict:
    labels = load_labels(cfg)
    out = data_dir(cfg) / "results"
    scores = pd.read_parquet(out / "scores.parquet")
    T0 = cfg["evaluation"]["test_snapshots"][0]
    res: dict = {"primary_snapshot": T0, "models": {}}
    for m in model_columns(scores):
        s0 = scores[scores["T"] == T0]
        r = ranking_metrics(s0["y"].to_numpy(), s0[m].to_numpy(), cfg)
        per_T = [ranking_metrics(g["y"].to_numpy(), g[m].to_numpy(), cfg)["pr_auc_at_1pct"] for _, g in scores.groupby("T")]
        r["pr_auc_at_1pct_mean_all_test_snapshots"] = float(np.mean(per_T))
        flag = f"flag_{m}"
        scores[flag] = top_k_flags(scores, m, cfg)
        r["recall_at_k_by_level"] = {int(lv): float(g[flag].sum() / max(1, len(g))) for lv, g in
                                     scores[(scores["T"] == T0) & (scores["y"] == 1)].groupby("sophistication")}
        r["rings"] = ring_metrics(scores, flag)
        r["confounders"] = confounder_rates(scores[scores["T"] == T0], labels, flag)
        res["models"][m] = r
    res["community_quality"] = community_quality(scores[scores["T"] == T0])
    scores.to_parquet(out / "scores_flags.parquet")
    (out / "evaluation.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    return res


def stage_privacy(cfg: dict) -> dict:
    """Couche inter-bancaire : OPRF + agrégation Paillier → features xb_* par snapshot, puis scoring chiffré."""
    from .privacy.consortium import run_consortium
    from .privacy.scoring import encrypted_scoring_demo

    vault = load_tables(data_dir(cfg) / "vault", ["records", "identities", "record_attr"])
    t = time.perf_counter()
    feats, metrics = run_consortium(vault, cfg, snapshots(cfg))
    out = data_dir(cfg) / "features"
    for T, df in feats.items():
        df.to_parquet(out / f"xbank_T{T}.parquet", index=False)
    metrics["total_seconds"] = round(time.perf_counter() - t, 1)
    log(f"privacy : {metrics['oprf_evaluations']} évaluations OPRF, "
        f"{sum(r['ciphertexts'] for r in metrics['rounds'].values())} chiffrés Paillier ({metrics['total_seconds']}s)")

    rows = build_rows(load_features(cfg), load_labels(cfg), cfg)
    metrics["encrypted_scoring"] = encrypted_scoring_demo(rows[rows["split"] == "train"], rows[rows["split"] == "test"],
                                                          cfg, cfg["privacy"]["encrypted_scoring_sample"], cfg["seed"])
    log(f"scoring chiffré : erreur max {metrics['encrypted_scoring']['max_abs_error_logit']:.2e}, "
        f"{metrics['encrypted_scoring']['total_ms_per_request']} ms/requête")
    (data_dir(cfg) / "results").mkdir(exist_ok=True)
    (data_dir(cfg) / "results" / "privacy.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    return metrics


def stage_alerts(cfg: dict) -> dict:
    from .alerts import run_alerts

    out = data_dir(cfg) / "results"
    scores = pd.read_parquet(out / "scores_flags.parquet")
    evaluation = json.loads((out / "evaluation.json").read_text(encoding="utf-8"))
    res = run_alerts(scores, evaluation, load_labels(cfg), load_bank(cfg), cfg, report_dir(cfg) / "alertes")
    res.pop("_alerted").to_parquet(out / "alerted.parquet", index=False)
    (out / "alerts.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    log(f"alertes : {res['alerted_communities']} communautés, précision {res['community_level']['precision']:.2f}, "
        f"{len(res['case_files'])} dossiers")
    return res


def stage_neo4j(cfg: dict) -> dict:
    from .alerts import best_model
    from .graph_db import run_neo4j_stage

    out = data_dir(cfg) / "results"
    scores = pd.read_parquet(out / "scores_flags.parquet")
    evaluation = json.loads((out / "evaluation.json").read_text(encoding="utf-8"))
    alerted = pd.read_parquet(out / "alerted.parquet") if (out / "alerted.parquet").exists() else None
    res = run_neo4j_stage(cfg, load_bank(cfg), scores, best_model(evaluation), alerted)
    (out / "neo4j.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    log(f"neo4j : {res['load']['nodes']} nœuds, {res['load']['relationships']} relations ({res['load']['seconds']}s)")
    return res
