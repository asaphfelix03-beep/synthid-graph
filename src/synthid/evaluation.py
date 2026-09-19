"""Protocole d'évaluation : split temporel + disjoint par groupe (anneau / foyer) + scénario hold-out, et métriques.

- Entraînement : snapshots T1 ∈ train_snapshots, labels connus à `label_cutoff` (confirmations observées) ;
  les bust-outs non reconnus (bruit de label) restent des négatifs, comme dans une vraie banque.
- Test : snapshots T2 ∈ test_snapshots (≥ label_cutoff), labels = vérité terrain.
- Un anneau est « d'entraînement » s'il a fait son bust-out avant `label_cutoff` ; les autres, et tous les
  anneaux d'un scénario hold-out, sont réservés au test. Les foyers légitimes sont répartis par tirage.
- Population évaluée à T : comptes ouverts, de moins de `max_tenure_days`, pas encore en bust-out.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, normalized_mutual_info_score, roc_auc_score, roc_curve


def _uniform(key: str, seed: int) -> float:
    h = hashlib.sha256(f"{seed}|{key}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def assign_folds(labels: pd.DataFrame, cfg: dict) -> pd.Series:
    ev, seed = cfg["evaluation"], cfg["seed"]
    holdout = set(cfg["fraud"]["holdout_scenarios"])
    ring_bust = labels[labels["is_synthetic"] == 1].groupby("group")["bust_out_day"].min()
    folds = {}
    for group in labels["group"].unique():
        u = _uniform(group, seed)
        if group.startswith("r"):
            scen = labels.loc[labels["group"] == group, "scenario"].iloc[0]
            if scen in holdout or ring_bust[group] > ev["label_cutoff"]:
                folds[group] = "test"
            else:
                folds[group] = "val" if u < ev["val_group_share"] else "train"
        else:
            if u < ev["test_group_share"]:
                folds[group] = "test"
            else:
                v = (u - ev["test_group_share"]) / (1 - ev["test_group_share"])
                folds[group] = "val" if v < ev["val_group_share"] else "train"
    return labels["group"].map(folds)


def build_rows(features: pd.DataFrame, labels: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Ajoute fold, cible et filtres de population à la table de features empilée (tous snapshots)."""
    ev = cfg["evaluation"]
    lab = labels.set_index("person_id")
    lab = lab.assign(fold=assign_folds(labels, cfg).values)
    df = features.join(lab[["fold", "is_synthetic", "label_observed", "label_day", "bust_out_day", "ring_id",
                            "sophistication", "scenario", "exposure", "group"]], on="person_id")
    df["eligible"] = df["scored"] & (df["app_tenure_days"] <= ev["max_tenure_days"]) & (df["bust_out_day"] > df["T"])
    is_train_T = df["T"].isin(ev["train_snapshots"])
    is_test_T = df["T"].isin(ev["test_snapshots"])
    df["split"] = "none"
    df.loc[df["eligible"] & is_train_T & (df["fold"] == "train"), "split"] = "train"
    df.loc[df["eligible"] & is_train_T & (df["fold"] == "val"), "split"] = "val"
    df.loc[df["eligible"] & is_test_T & (df["fold"] == "test"), "split"] = "test"
    known_at_cutoff = (df["is_synthetic"] == 1) & df["label_observed"] & (df["label_day"] <= ev["label_cutoff"])
    df["y"] = np.where(df["split"] == "test", df["is_synthetic"], known_at_cutoff.astype(int))
    return df


# ------------------------------------------------------------------ métriques
def prevalence_weights(y: np.ndarray, target: float) -> np.ndarray:
    """Pondère les négatifs pour simuler une prévalence cible (les datasets du PoC sont enrichis en fraude)."""
    pos = y.sum()
    neg = len(y) - pos
    if pos == 0 or neg == 0:
        return np.ones(len(y))
    w_neg = pos * (1 - target) / (target * neg)
    return np.where(y == 1, 1.0, w_neg)


def weighted_selection(y: np.ndarray, score: np.ndarray, budget_share: float, target: float) -> np.ndarray:
    """Top-k « à prévalence réaliste » : on alerte les meilleurs scores jusqu'à ce que la population pondérée
    alertée atteigne budget_share × population pondérée (négatifs repondérés à la prévalence cible)."""
    w = prevalence_weights(np.asarray(y).astype(int), target)
    order = np.argsort(-np.asarray(score, dtype=float), kind="stable")
    n_sel = max(1, int(np.searchsorted(np.cumsum(w[order]), budget_share * w.sum(), side="right")))
    sel = np.zeros(len(order), dtype=bool)
    sel[order[:n_sel]] = True
    return sel


def ranking_metrics(y: np.ndarray, score: np.ndarray, cfg: dict) -> dict:
    ev = cfg["evaluation"]
    target = ev["target_prevalence"]
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    w = prevalence_weights(y, target)
    sel = weighted_selection(y, score, ev["alert_budget_share"], target)
    fpr, tpr, _ = roc_curve(y, score)
    return {
        "n": int(len(y)), "positives": int(y.sum()), "prevalence": float(y.mean()),
        "pr_auc": float(average_precision_score(y, score)),
        "pr_auc_at_1pct": float(average_precision_score(y, score, sample_weight=w)),
        "roc_auc": float(roc_auc_score(y, score)),
        "k_weighted": float(ev["alert_budget_share"] * w.sum()),
        "alerted_rows": int(sel.sum()),
        "precision_at_k": float((w * sel * y).sum() / (w * sel).sum()),
        "recall_at_k": float(y[sel].sum() / max(1, y.sum())),
        "recall_at_fpr_1pct": float(np.interp(0.01, fpr, tpr)),
    }


def top_k_flags(df: pd.DataFrame, score_col: str, cfg: dict) -> pd.Series:
    """Alertes par snapshot, au budget d'alertes des analystes (à prévalence réaliste)."""
    ev = cfg["evaluation"]
    flags = pd.Series(False, index=df.index)
    for _, g in df.groupby("T"):
        sel = weighted_selection(g["y"].to_numpy(), g[score_col].to_numpy(), ev["alert_budget_share"], ev["target_prevalence"])
        flags.loc[g.index[sel]] = True
    return flags


def ring_metrics(test: pd.DataFrame, flag_col: str) -> dict:
    """Recall par anneau, couverture, délai d'anticipation et exposition évitée (toutes dates de test)."""
    syn = test[test["is_synthetic"] == 1]
    if syn.empty:
        return {}
    first_alert = syn[syn[flag_col]].groupby("person_id")["T"].min()
    per_person = syn.groupby("person_id").agg(ring_id=("ring_id", "first"), bust=("bust_out_day", "first"),
                                              exposure=("exposure", "first"), level=("sophistication", "first"),
                                              scenario=("scenario", "first"))
    per_person["first_alert"] = first_alert
    per_person["detected"] = per_person["first_alert"].notna()
    per_person["lead"] = per_person["bust"] - per_person["first_alert"]
    rings = per_person.groupby("ring_id").agg(detected=("detected", "max"), coverage=("detected", "mean"),
                                              level=("level", "first"), scenario=("scenario", "first"))
    det = per_person[per_person["detected"]]
    out = {
        "rings": int(len(rings)),
        "ring_recall": float(rings["detected"].mean()),
        "ring_coverage_when_detected": float(rings.loc[rings["detected"], "coverage"].mean()) if rings["detected"].any() else 0.0,
        "person_recall_any_snapshot": float(per_person["detected"].mean()),
        "median_lead_time_days": float(det["lead"].median()) if len(det) else None,
        "share_detected_30d_before_bust": float((per_person["lead"] >= 30).mean()),
        "exposure_avoided_share": float(det["exposure"].sum() / max(1.0, per_person["exposure"].sum())),
        "ring_recall_by_level": {int(k): float(v) for k, v in rings.groupby("level")["detected"].mean().items()},
        "ring_recall_by_scenario": {str(k): float(v) for k, v in rings.groupby("scenario")["detected"].mean().items()},
    }
    return out


CONFOUNDERS = {
    "foyer_3plus": "household_size >= 3",
    "colocation": "is_colocation",
    "residence_etudiante": "is_dorm",
    "personne_agee_aidee": "is_elderly_assisted",
    "defaut_credit_legitime": "is_distressed",
    "primo_arrivant": "is_newcomer",
    "vendeur_tradeline": "is_tradeline_seller",
    "mule": "is_mule",
}


def confounder_rates(test: pd.DataFrame, labels: pd.DataFrame, flag_col: str) -> dict:
    """Taux d'alerte des faux positifs structurels comparé au taux d'alerte moyen des légitimes."""
    lab = labels.set_index("person_id")
    hh_size = labels[labels["is_synthetic"] == 0].groupby("household_id")["identity_id"].nunique()
    legit = test[test["is_synthetic"] == 0].copy()
    for c in ["is_colocation", "is_dorm", "is_elderly_assisted", "is_distressed", "is_newcomer",
              "is_tradeline_seller", "is_mule", "household_id"]:
        legit[c] = legit["person_id"].map(lab[c])
    legit["household_size"] = legit["household_id"].map(hh_size)
    base = float(legit[flag_col].mean())
    out = {"legit_alert_rate": base}
    for name, q in CONFOUNDERS.items():
        g = legit.query(q)
        if len(g):
            out[name] = {"n": int(len(g)), "alert_rate": float(g[flag_col].mean()),
                         "ratio_vs_legit": float(g[flag_col].mean() / base) if base > 0 else None}
    return out


def community_quality(test_T: pd.DataFrame) -> dict:
    syn = test_T[test_T["is_synthetic"] == 1]
    if syn["ring_id"].nunique() < 2:
        return {}
    return {"nmi_communities_vs_rings": float(normalized_mutual_info_score(syn["ring_id"], syn["community"]))}
