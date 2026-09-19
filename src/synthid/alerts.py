"""Phase 6 : du score individuel au score d'anneau, et dossiers d'alerte pour les analystes.

Score de communauté (Louvain sur la projection pondérée) = moyenne des 3 meilleurs scores de ses membres.
Les communautés sont alertées par ordre de score jusqu'à épuisement du budget d'alertes (en personnes).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .evaluation import prevalence_weights, weighted_selection
from .features import take_snapshot

FEATURE_LABELS = {
    "g_known_fraud_1hop": "voisin direct d'une fraude confirmée",
    "g_share_device": "appareil partagé avec d'autres clients",
    "g_share_phone": "téléphone partagé",
    "g_nid_identity_mismatch": "n° d'identité porté par plusieurs identités",
    "app_email_age_at_onboard": "email récent à l'ouverture",
    "app_phone_age_at_onboard": "téléphone récent à l'ouverture",
    "app_file_age": "dossier de crédit jeune",
    "beh_lir_rate": "demandes de hausse de plafond fréquentes",
    "beh_pay_full_share": "remboursements systématiquement intégraux",
    "xb_mismatch_total": "attributs réutilisés sous une autre identité dans une autre banque",
    "xb_nid_mismatch": "n° d'identité utilisé sous une autre identité dans une autre banque",
    "xb_phone_mismatch": "téléphone utilisé sous une autre identité dans une autre banque",
    "xb_device_mismatch": "appareil utilisé sous une autre identité dans une autre banque",
    "xb_email_mismatch": "email utilisé sous une autre identité dans une autre banque",
    "xb_identity_other_banks": "identité présente dans d'autres banques",
    "app_income": "revenu déclaré atypique",
    "app_has_employer": "employeur déclaré",
    "app_phone_voip": "téléphone VoIP",
    "app_tenure_days": "ancienneté du compte",
    "app_file_age_at_onboard": "dossier de crédit jeune à l'ouverture",
    "g_au_host_max_dependents": "utilisateur autorisé sur un compte hôte très sollicité (tradeline)",
    "g_comm_known_share": "communauté contenant des fraudes confirmées",
    "g_known_fraud_2hop_paths": "fraudes confirmées à 2 sauts",
    "g_share_employer": "employeur partagé avec d'autres clients suspects",
    "g_max_strong_share": "attribut fort très partagé",
    "g_share_address": "adresse partagée",
    "g_share_email": "email partagé",
    "g_share_nid": "n° d'identité partagé",
    "g_similar_count": "quasi-doublon nom / date de naissance",
    "g_transfer_out_cp": "virements vers d'autres clients",
    "g_small_merchant_covisitors": "petits marchands fréquentés par les mêmes clients",
    "beh_limit_growth": "croissance rapide du plafond",
    "beh_limit_last": "plafond actuel",
    "beh_pay_mean6": "taux de remboursement",
    "beh_pay_std6": "régularité des remboursements",
    "beh_txn_mean6": "nombre de transactions",
}


def community_alerts(scores: pd.DataFrame, score_col: str, T: int, cfg: dict, max_size: int = 40) -> pd.DataFrame:
    """Communautés alertées par score décroissant jusqu'au budget (population pondérée à prévalence réaliste)."""
    ev = cfg["evaluation"]
    s = scores[scores["T"] == T].copy()
    s["w"] = prevalence_weights(s["y"].to_numpy(), ev["target_prevalence"])
    budget = ev["alert_budget_share"] * s["w"].sum()
    grp = s.groupby("community")
    comm = pd.DataFrame({"size": grp.size(), "weight": grp["w"].sum(),
                         "score": grp[score_col].apply(lambda x: x.nlargest(3).mean())})
    # un anneau compte au plus quelques dizaines de membres : une « communauté » plus grande n'est pas un anneau
    comm = comm[(comm["size"] >= 2) & (comm["size"] <= max_size)].sort_values("score", ascending=False)
    chosen, total = [], 0.0
    for c, row in comm.iterrows():
        if total + row["weight"] > budget:
            continue  # trop grosse pour le budget restant : on essaie la suivante
        chosen.append(c)
        total += row["weight"]
    out = s[s["community"].isin(chosen)].copy()
    out["community_score"] = out["community"].map(comm["score"])
    return out


def alert_metrics(scores: pd.DataFrame, alerted: pd.DataFrame, score_col: str, T: int, cfg: dict) -> dict:
    ev = cfg["evaluation"]
    s = scores[scores["T"] == T].copy()
    s["w"] = prevalence_weights(s["y"].to_numpy(), ev["target_prevalence"])
    top = s[weighted_selection(s["y"].to_numpy(), s[score_col].to_numpy(), ev["alert_budget_share"], ev["target_prevalence"])]
    rings = s[s["y"] == 1]["ring_id"].nunique()

    def summary(df):
        return {"alerted_rows": int(len(df)), "alerted_weighted": float(df["w"].sum()),
                "precision": float((df["w"] * df["y"]).sum() / max(1e-9, df["w"].sum())),
                "recall": float(df["y"].sum() / max(1, s["y"].sum())),
                "ring_recall": float(df[df["y"] == 1]["ring_id"].nunique() / max(1, rings))}
    return {"budget_weighted": float(ev["alert_budget_share"] * s["w"].sum()), "person_level_top_k": summary(top),
            "community_level": summary(alerted), "alerted_communities": int(alerted["community"].nunique())}


def write_case_files(alerted: pd.DataFrame, labels: pd.DataFrame, bank: dict[str, pd.DataFrame], T: int,
                     score_col: str, folder: Path, n_cases: int = 10) -> list[str]:
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("alerte_*.md"):
        old.unlink()
    snap = take_snapshot(bank, T)
    pa = snap.person_attr
    lab = labels.set_index("person_id")
    persons = snap.persons.set_index("person_id")
    files = []
    ranked = (alerted.groupby("community")["community_score"].first().sort_values(ascending=False).head(n_cases))
    for rank, (c, cscore) in enumerate(ranked.items(), start=1):
        members = alerted[alerted["community"] == c].sort_values(score_col, ascending=False)
        ids = set(members["person_id"])
        shared = pa[pa["person_id"].isin(ids)].groupby(["attr_type", "token"])["person_id"].nunique()
        shared = shared[shared >= 2].reset_index().rename(columns={"person_id": "members"})
        tr = snap.transfers[snap.transfers["src"].isin(ids) & snap.transfers["dst"].isin(ids)]
        au_hosts = snap.authorized_users[snap.authorized_users["person_id"].isin(ids)]["host_person_id"].value_counts()
        lines = [f"# Dossier d'alerte n°{rank} — communauté {c}", "",
                 f"- Date d'analyse : J+{T}  ·  score de communauté : **{cscore:.3f}**  ·  membres : {len(members)}",
                 f"- Banque(s) : {', '.join(sorted({p.split('-')[0] for p in ids}))}", "",
                 "## Membres", "", "| Client | Score | Ancienneté (j) | Principaux facteurs |", "|---|---|---|---|"]
        for _, m in members.iterrows():
            expl = json.loads(m["explanation"]) if isinstance(m.get("explanation"), str) else []
            why = ", ".join(FEATURE_LABELS.get(f, f) for f, _ in expl[:3])
            tenure = T - int(persons.at[m["person_id"], "onboard_day"])
            lines.append(f"| {m['person_id']} | {m[score_col]:.3f} | {tenure} | {why} |")
        lines += ["", "## Points de contact partagés (jetons tronqués)", ""]
        if len(shared):
            lines += ["| Type | Jeton | Membres concernés |", "|---|---|---|"]
            lines += [f"| {r.attr_type} | `{r.token[:10]}…` | {r.members} |" for r in shared.itertuples()]
        else:
            lines.append("Aucun attribut partagé direct : lien par comportement, marchand ou virement.")
        if len(tr):
            lines += ["", f"## Virements internes : {len(tr)} mois-paires, {tr['amount'].sum():,.0f} € cumulés"]
        if len(au_hosts):
            lines += ["", "## Utilisateurs autorisés sur des comptes tiers (piggybacking)", ""]
            lines += [f"- hôte {h} : {n} membre(s) de la communauté" for h, n in au_hosts.items()]
        truth = lab.loc[list(ids)]
        lines += ["", "---", "", "_Vérité terrain (évaluation du PoC uniquement, invisible pour l'analyste)_ : "
                  f"{int(truth['is_synthetic'].sum())}/{len(ids)} identités synthétiques ; anneaux "
                  f"{sorted(int(x) for x in truth.loc[truth['is_synthetic'] == 1, 'ring_id'].unique())}."]
        path = folder / f"alerte_{rank:02d}_communaute_{c}.md"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        files.append(path.name)
    return files


def best_model(evaluation: dict) -> str:
    models = evaluation["models"]
    return max(models, key=lambda m: models[m]["pr_auc"])


def run_alerts(scores: pd.DataFrame, evaluation: dict, labels: pd.DataFrame, bank: dict, cfg: dict, folder: Path) -> dict:
    col = best_model(evaluation)
    T = cfg["evaluation"]["test_snapshots"][0]
    alerted = community_alerts(scores, col, T, cfg)
    res = {"model": col, "snapshot": T, **alert_metrics(scores, alerted, col, T, cfg)}
    res["case_files"] = write_case_files(alerted, labels, bank, T, col, folder)
    res["_alerted"] = alerted[["person_id", "community"]]
    return res
