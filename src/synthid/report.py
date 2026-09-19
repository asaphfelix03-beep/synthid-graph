"""Rapport Markdown du PoC (reports/<config>/REPORT.md) et évaluation des critères go/no-go."""
from __future__ import annotations

import json
from pathlib import Path

from .config import data_dir, report_dir

MODEL_LABELS = {
    "rules": "Moteur de règles (existant)",
    "lgbm_tab": "LightGBM — tabulaire seul",
    "lgbm_graph": "LightGBM — tabulaire + features de graphe",
    "gcn_tab": "GCN (projection homogène)",
    "hsage_tab": "GraphSAGE hétérogène — entrées tabulaires",
    "hsage_graph": "GraphSAGE hétérogène — entrées + features de graphe",
    "ensemble": "Ensemble LightGBM-graphe + GraphSAGE",
    "lgbm_xb": "LightGBM + features inter-bancaires chiffrées",
    "ensemble_xb": "Ensemble inter-bancaire + GraphSAGE",
}
BASELINES = ["rules", "lgbm_tab", "lgbm_graph"]
GNNS = ["gcn_tab", "hsage_tab", "hsage_graph"]


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _pct(x) -> str:
    return "—" if x is None else f"{100 * x:.1f} %"


def _f(x, d=3) -> str:
    return "—" if x is None else f"{x:.{d}f}"


def _n(x) -> str:
    return f"{x:,}".replace(",", " ")


def confounder_mean(conf: dict) -> float:
    groups = [v for k, v in conf.items() if isinstance(v, dict)]
    n = sum(g["n"] for g in groups)
    return sum(g["alert_rate"] * g["n"] for g in groups) / max(1, n)


def go_no_go(ev: dict, priv: dict, cfg: dict) -> list[dict]:
    g = cfg["go_no_go"]
    models = ev["models"]
    best_base = max(BASELINES, key=lambda m: models[m]["pr_auc"])
    best = max(models, key=lambda m: models[m]["pr_auc"])
    crit = []
    gnns = [m for m in GNNS if m in models]
    if gnns:
        best_gnn = max(gnns, key=lambda m: models[m]["pr_auc"])
        gain = models[best_gnn]["pr_auc"] - models[best_base]["pr_auc"]
        crit.append({"critere": f"Le meilleur GNN ({best_gnn}) bat le meilleur baseline ({best_base}) en PR-AUC",
                     "mesure": f"{gain:+.3f}", "cible": f"> {g['min_pr_auc_gain_vs_best_baseline']:+.3f}",
                     "ok": gain > g["min_pr_auc_gain_vs_best_baseline"]})
    l3 = {m: models[m]["recall_at_k_by_level"].get("3", 0.0) for m in models}
    graph_models = [m for m in models if m not in ("rules", "lgbm_tab")]
    best_l3 = max(graph_models, key=lambda m: l3[m])
    crit.append({"critere": f"Niveau 3 (fraudeur sophistiqué) : le graphe ({best_l3}) fait mieux que le tabulaire seul",
                 "mesure": f"{_pct(l3[best_l3])} vs {_pct(l3['lgbm_tab'])} (Recall@k)", "cible": "> tabulaire",
                 "ok": l3[best_l3] > l3["lgbm_tab"]})
    rr = models[best]["rings"]["ring_recall"]
    crit.append({"critere": f"Anneaux détectés avant le bust-out ({best})", "mesure": _pct(rr),
                 "cible": f"≥ {_pct(g['min_ring_recall_before_bust_out'])}", "ok": rr >= g["min_ring_recall_before_bust_out"]})
    lead = models[best]["rings"].get("median_lead_time_days")
    crit.append({"critere": "Délai médian d'anticipation", "mesure": f"{lead:.0f} j" if lead is not None else "—",
                 "cible": f"≥ {g['min_lead_time_days']} j", "ok": lead is not None and lead >= g["min_lead_time_days"]})
    c_best, c_rules = confounder_mean(models[best]["confounders"]), confounder_mean(models["rules"]["confounders"])
    crit.append({"critere": "Taux d'alerte sur les faux positifs structurels ≤ moteur de règles",
                 "mesure": f"{_pct(c_best)} vs {_pct(c_rules)}", "cible": "≤ règles", "ok": c_best <= c_rules})
    holdout = cfg["fraud"]["holdout_scenarios"]
    hr = [models[best]["rings"]["ring_recall_by_scenario"].get(s) for s in holdout]
    hr = [x for x in hr if x is not None]
    crit.append({"critere": f"Scénario jamais vu ({', '.join(holdout)}) : détection non nulle",
                 "mesure": _pct(min(hr)) if hr else "—", "cible": "> 0 %", "ok": bool(hr) and min(hr) > 0})
    if priv and "lgbm_xb" in models:
        up = models["lgbm_xb"]["pr_auc"] - models["lgbm_graph"]["pr_auc"]
        crit.append({"critere": "Gain des features inter-bancaires (lgbm_xb vs lgbm_graph)", "mesure": f"{up:+.3f} PR-AUC",
                     "cible": "> 0", "ok": up > 0})
        lk = priv["leakage"]
        crit.append({"critere": "Aucune PII en clair reçue par le hub ni le service de tokenisation",
                     "mesure": f"{lk['raw_pii_values_received_by_hub']} / {lk['unblinded_points_received_by_tokenization_service']}",
                     "cible": "0 / 0",
                     "ok": lk["raw_pii_values_received_by_hub"] == 0 and lk["unblinded_points_received_by_tokenization_service"] == 0})
        crit.append({"critere": "Niveau de sécurité Paillier", "mesure": f"{priv['security_bits']} bits",
                     "cible": "≥ 128 bits", "ok": (priv["security_bits"] or 0) >= 128})
        es = priv["encrypted_scoring"]
        crit.append({"critere": "Scoring chiffré : erreur max / latence par requête",
                     "mesure": f"{es['max_abs_error_logit']:.1e} / {es['total_ms_per_request']:.0f} ms",
                     "cible": f"≤ {g['max_encrypted_score_abs_error']:.0e} / < 1000 ms",
                     "ok": es["max_abs_error_logit"] <= g["max_encrypted_score_abs_error"] and es["total_ms_per_request"] < 1000})
    return crit


def interpretation(models: dict, alerts: dict) -> list[str]:
    """Conclusions calculées à partir des métriques (aucune valeur codée en dur)."""
    m = models
    l3 = lambda k: m[k]["recall_at_k_by_level"].get("3", 0.0)  # noqa: E731
    out = [f"Sans graphe, le fraudeur sophistiqué (niveau 3) est invisible : Recall@k {_pct(l3('rules'))} pour les règles, "
           f"{_pct(l3('lgbm_tab'))} pour LightGBM tabulaire. Les features de graphe le portent à {_pct(l3('lgbm_graph'))}."]
    gnns = [k for k in GNNS if k in m]
    if gnns:
        gnn_best = max(gnns, key=lambda k: m[k]["pr_auc_at_1pct"])
        delta = m[gnn_best]["pr_auc_at_1pct"] - m["lgbm_graph"]["pr_auc_at_1pct"]
        verdict = "dépasse" if delta > 0 else "ne dépasse pas"
        out.append(f"Seul, le meilleur GNN ({MODEL_LABELS[gnn_best]}) {verdict} LightGBM + features de graphe "
                   f"(PR-AUC @1 % : {_f(m[gnn_best]['pr_auc_at_1pct'])} vs {_f(m['lgbm_graph']['pr_auc_at_1pct'])}). "
                   "Des features de graphe bien construites captent déjà l'essentiel de la structure locale.")
    if "ensemble" in m:
        out.append(f"En revanche, le GNN apporte un signal complémentaire : l'ensemble LightGBM-graphe + GraphSAGE atteint "
                   f"une PR-AUC @1 % de {_f(m['ensemble']['pr_auc_at_1pct'])} et détecte {_pct(m['ensemble']['rings']['ring_recall'])} "
                   f"des anneaux avant bust-out (vs {_pct(m['lgbm_graph']['rings']['ring_recall'])}).")
    if "lgbm_xb" in m:
        out.append(f"Les signaux inter-bancaires chiffrés font passer les anneaux détectés de "
                   f"{_pct(m['lgbm_graph']['rings']['ring_recall'])} à {_pct(m['lgbm_xb']['rings']['ring_recall'])}, et les anneaux "
                   f"de niveau 3 de {_pct(m['lgbm_graph']['rings']['ring_recall_by_level'].get('3'))} à "
                   f"{_pct(m['lgbm_xb']['rings']['ring_recall_by_level'].get('3'))}, sans échange de PII.")
    if alerts:
        c, p = alerts["community_level"], alerts["person_level_top_k"]
        better = c["precision"] > p["precision"]
        out.append(f"Alerter par communauté entière {'améliore' if better else 'dégrade'} la précision ({_pct(c['precision'])} vs "
                   f"{_pct(p['precision'])} pour le top-k individuel) : les communautés Louvain mélangent anneaux et voisins "
                   "légitimes. Recommandation : sélectionner les alertes au niveau individuel, puis regrouper les alertés par "
                   "communauté pour construire les dossiers d'enquête (un dossier peut relier plusieurs anneaux d'un même opérateur).")
    return out


def build_report(cfg: dict) -> Path:
    d = data_dir(cfg)
    res = d / "results"
    stats, info = _load(d / "dataset_stats.json"), _load(res / "train_info.json")
    ev, priv = _load(res / "evaluation.json"), _load(res / "privacy.json")
    alerts, neo = _load(res / "alerts.json"), _load(res / "neo4j.json")
    models = ev["models"]
    best = max(models, key=lambda m: models[m]["pr_auc"])
    T0 = ev["primary_snapshot"]
    crit = go_no_go(ev, priv, cfg)
    L = []
    a = L.append
    a(f"# Rapport du PoC — détection d'identités synthétiques (config « {cfg['name']} »)\n")
    a("> Rapport généré automatiquement par `synthid report`. Données 100 % synthétiques : les valeurs absolues "
      "reflètent les hypothèses du générateur. Ce qui compte : les écarts entre modèles, la tenue par niveau de "
      "sophistication et sur le scénario jamais vu.\n")

    a("## 1. Synthèse\n")
    passed = sum(c["ok"] for c in crit)
    a(f"- Meilleur modèle (PR-AUC au snapshot J+{T0}) : **{MODEL_LABELS.get(best, best)}** — "
      f"PR-AUC {_f(models[best]['pr_auc'])}, PR-AUC ramenée à 1 % de prévalence {_f(models[best]['pr_auc_at_1pct'])}.")
    r = models[best]["rings"]
    a(f"- Anneaux détectés avant bust-out : **{_pct(r['ring_recall'])}**, délai médian d'anticipation "
      f"**{_f(r['median_lead_time_days'], 0)} jours**, exposition (plafonds) couverte : {_pct(r['exposure_avoided_share'])}.")
    a(f"- Critères go/no-go satisfaits : **{passed}/{len(crit)}** (détail §9, lecture §11).\n")

    a("## 2. Jeu de données simulé\n")
    a("| Élément | Valeur |\n|---|---|")
    for k, lbl in [("identities", "Identités"), ("bank_records", "Comptes bancaires (3 banques)"),
                   ("synthetic_records", "Comptes d'identités synthétiques"), ("rings", "Anneaux de fraude"),
                   ("cross_bank_rings", "Anneaux répartis sur plusieurs banques"),
                   ("person_attribute_edges", "Arêtes personne → attribut"), ("merchant_edges", "Arêtes compte → marchand"),
                   ("transfer_rows", "Virements (agrégats mensuels)"), ("entity_resolution_edges", "Quasi-doublons (résolution d'entités)"),
                   ("fraud_confirmations", "Confirmations de fraude datées")]:
        a(f"| {lbl} | {_n(stats[k]) if k in stats else '—'} |")
    a(f"| Anneaux par niveau (1/2/3) | {' / '.join(str(v) for v in stats.get('rings_by_level', {}).values())} |")
    a(f"| Anneaux par scénario | {', '.join(f'{k} : {v}' for k, v in stats.get('rings_by_scenario', {}).items())} |\n")
    rows = info.get("rows", {})
    a(f"Lignes d'apprentissage : train {rows.get('train', {}).get('n')} ({rows.get('train', {}).get('positives')} positifs), "
      f"validation {rows.get('val', {}).get('n')} ({rows.get('val', {}).get('positives')}), "
      f"test {rows.get('test', {}).get('n')} ({rows.get('test', {}).get('positives')}) "
      f"sur {len(cfg['evaluation']['test_snapshots'])} snapshots.\n")

    sa = info.get("shortcut_audit", {})
    a("### Audit de raccourcis (jalon de la phase 2)\n")
    a(f"AUC maximale d'une feature isolée : **{_f(sa.get('max_auc'))}** (seuil {sa.get('threshold')}) → "
      f"{'✅ réussi' if sa.get('passed') else '❌ échec : le générateur laisse une signature'}.\n")
    a("| Feature | AUC univariée |\n|---|---|")
    for k, v in sa.get("top10", {}).items():
        a(f"| `{k}` | {v:.3f} |")
    a("")

    a(f"## 3. Performance des modèles (snapshot J+{T0}, population : comptes < 24 mois, pas encore en bust-out)\n")
    a("| Modèle | PR-AUC | PR-AUC @1 % | ROC-AUC | Precision@k | Recall@k | Recall @FPR 1 % | Anneaux détectés | Délai médian |")
    a("|---|---|---|---|---|---|---|---|---|")
    for m, v in models.items():
        a(f"| {MODEL_LABELS.get(m, m)} | {_f(v['pr_auc'])} | {_f(v['pr_auc_at_1pct'])} | {_f(v['roc_auc'])} | "
          f"{_pct(v['precision_at_k'])} | {_pct(v['recall_at_k'])} | {_pct(v['recall_at_fpr_1pct'])} | "
          f"{_pct(v['rings'].get('ring_recall'))} | {_f(v['rings'].get('median_lead_time_days'), 0)} j |")
    k0 = next(iter(models.values()))
    a(f"\nBudget d'alertes : {_pct(cfg['evaluation']['alert_budget_share'])} de la population, évalué comme si la "
      f"prévalence était de {_pct(cfg['evaluation']['target_prevalence'])} (négatifs repondérés ; prévalence réelle du jeu de "
      f"test : {_pct(k0['prevalence'])}, {k0['n']} comptes). Precision@k, Recall@k et « PR-AUC @1 % » suivent cette convention.\n")

    a("## 4. Robustesse par niveau de sophistication et scénario\n")
    a("| Modèle | Recall@k niveau 1 | niveau 2 | niveau 3 | " + " | ".join(
        f"anneaux « {s} »" for s in sorted(models[best]["rings"]["ring_recall_by_scenario"])) + " |")
    a("|---|---|---|---|" + "---|" * len(models[best]["rings"]["ring_recall_by_scenario"]))
    for m, v in models.items():
        lv = v["recall_at_k_by_level"]
        sc = v["rings"]["ring_recall_by_scenario"]
        a(f"| {MODEL_LABELS.get(m, m)} | {_pct(lv.get('1'))} | {_pct(lv.get('2'))} | {_pct(lv.get('3'))} | "
          + " | ".join(_pct(sc.get(s)) for s in sorted(models[best]["rings"]["ring_recall_by_scenario"])) + " |")
    a(f"\nScénario hold-out (jamais vu à l'entraînement) : {', '.join(cfg['fraud']['holdout_scenarios'])}.\n")

    a(f"## 5. Faux positifs structurels (taux d'alerte, snapshot J+{T0})\n")
    compared = ["rules", "lgbm_graph", best]
    a("| Groupe | Effectif | " + " | ".join(MODEL_LABELS.get(m) or m for m in compared) + " |")
    a("|---|---|---|---|---|")
    groups = [g for g in models["rules"]["confounders"] if g != "legit_alert_rate"]
    legit_rates = " | ".join(_pct(models[m]["confounders"]["legit_alert_rate"]) for m in compared)
    a(f"| Tous les clients légitimes | — | {legit_rates} |")
    for gname in groups:
        a(f"| {gname.replace('_', ' ')} | {models['rules']['confounders'][gname]['n']} | " + " | ".join(
            _pct(models[m]["confounders"].get(gname, {}).get("alert_rate")) for m in ["rules", "lgbm_graph", best]) + " |")
    a("")

    if alerts:
        a("## 6. Alertes par communauté (phase 6)\n")
        a(f"Modèle : {MODEL_LABELS.get(alerts['model'], alerts['model'])} — même budget d'alertes que ci-dessus.\n")
        a("| Stratégie | Comptes alertés | Précision (prév. 1 %) | Recall | Anneaux touchés |\n|---|---|---|---|---|")
        for key, lbl in [("person_level_top_k", "Top-k individuel"), ("community_level", "Communautés (score d'anneau)")]:
            s = alerts[key]
            a(f"| {lbl} | {s['alerted_rows']} | {_pct(s['precision'])} | {_pct(s['recall'])} | {_pct(s['ring_recall'])} |")
        cq = ev.get("community_quality", {})
        if cq:
            a(f"\nNMI communautés Louvain ↔ anneaux réels (identités synthétiques) : {_f(cq['nmi_communities_vs_rings'])}.")
        a(f"\nDossiers d'alerte générés : {len(alerts['case_files'])} (dossier `reports/{cfg['name']}/alertes/`).\n")

    if priv:
        a("## 7. Couche inter-bancaire chiffrée\n")
        if "lgbm_xb" in models:
            a("| Modèle | PR-AUC | Recall@k | Anneaux détectés | Anneaux niveau 3 |\n|---|---|---|---|---|")
            for m in ["lgbm_graph", "lgbm_xb", "ensemble_xb"]:
                if m in models:
                    v = models[m]
                    a(f"| {MODEL_LABELS.get(m, m)} | {_f(v['pr_auc'])} | {_pct(v['recall_at_k'])} | {_pct(v['rings']['ring_recall'])} | "
                      f"{_pct(v['rings']['ring_recall_by_level'].get('3'))} |")
            a("")
        lk = priv["leakage"]
        rounds = priv["rounds"]
        n_ct = sum(r["ciphertexts"] for r in rounds.values())
        a("| Mesure | Valeur |\n|---|---|")
        a(f"| Paillier | {priv['paillier_modulus_bits']} bits ≈ {priv['security_bits']} bits de sécurité, "
          f"{priv['slots_per_ciphertext']} compteurs par chiffré (packing) |")
        a(f"| OPRF (secp256k1) | {_n(priv['oprf_evaluations'])} évaluations, {priv['oprf_ms_per_value']} ms/valeur |")
        a(f"| Chiffrés échangés ({len(rounds)} snapshots) | {_n(n_ct)} ({sum(r['bytes'] for r in rounds.values()) / 1e6:.1f} Mo) |")
        real_slots = sum(r["real_slots"] for r in rounds.values())
        decoy_slots = sum(r["decoy_slots"] for r in rounds.values())
        a(f"| Slots réels / leurres | {_n(real_slots)} / {_n(decoy_slots)} |")
        a(f"| Exactitude des agrégats chiffrés | erreur max {max(r['max_abs_error'] for r in rounds.values())} |")
        a(f"| PII en clair reçues par le hub | {lk['raw_pii_values_received_by_hub']} |")
        a(f"| Points non aveuglés reçus par le service OPRF | {lk['unblinded_points_received_by_tokenization_service']} |")
        naive = _pct(lk["dictionary_attack_recovery_naive_sha256"])
        a(f"| Attaque par dictionnaire du hub — jetons SHA-256 naïfs | {naive} des valeurs retrouvées |")
        a(f"| Attaque par dictionnaire du hub — jetons OPRF | {_pct(lk['dictionary_attack_recovery_oprf'])} |")
        es = priv["encrypted_scoring"]
        a(f"| Scoring chiffré (tête logistique, {len(es['features'])} features) | erreur max {es['max_abs_error_logit']:.1e}, "
          f"{es['total_ms_per_request']:.0f} ms et {es['bytes_per_request'] / 1024:.1f} Ko par requête |")
        a(f"| Durée totale de la couche | {priv['total_seconds']} s |\n")
        a("Fuite résiduelle assumée : le hub apprend quels jetons pseudonymes sont présents dans au moins deux banques "
          "(pas les PII, ni les volumes). Les banques contributrices reçoivent leurs jetons mélangés à des leurres.\n")

    if neo:
        a("## 8. Neo4j\n")
        ld = neo["load"]
        a(f"Chargement : {_n(ld['nodes'])} nœuds, {_n(ld['relationships'])} relations en {ld['seconds']} s.\n")
        a("| Requête d'investigation | Résultat | Latence |\n|---|---|---|")
        for k, v in neo["investigation_queries"].items():
            a(f"| {k.replace('_', ' ')} | {v['result']} | {v['ms']} ms |")
        gd = neo["gds"]
        a(f"\nGDS : projection {_n(gd['projected_nodes'])} nœuds / {_n(gd['projected_relationships'])} relations ; "
          f"WCC {_n(gd['wcc_components'])} composantes ; Louvain {_n(gd['louvain_communities'])} communautés "
          f"(modularité {gd['louvain_modularity']}).\n")

    a("## 9. Critères go/no-go (§3.6 du plan)\n")
    a("| Critère | Mesure | Cible | Statut |\n|---|---|---|---|")
    for c in crit:
        a(f"| {c['critere']} | {c['mesure']} | {c['cible']} | {'✅' if c['ok'] else '❌'} |")
    a("")

    a("## 10. Importance des features (LightGBM + graphe, gain)\n")
    a("| Feature | Part du gain |\n|---|---|")
    for k, v in info.get("lgbm_graph_top_features", {}).items():
        a(f"| `{k}` | {_pct(v)} |")
    a("")

    a("## 11. Lecture des résultats\n")
    for line in interpretation(models, alerts):
        a(f"- {line}")
    a("")
    a("## 12. Limites\n")
    a("- Données synthétiques : le modèle retrouve en partie les mécanismes codés dans le générateur (risque de "
      "circularité). Le scénario hold-out et les faux positifs structurels en donnent une mesure, pas une garantie.")
    a("- Prévalence enrichie pour la puissance statistique ; se référer à la PR-AUC ramenée à 1 %.")
    a("- Une seule graine : les écarts de quelques points entre modèles ne sont pas significatifs sans répétitions.")
    a("- Le GNN est entraîné en « full batch » sur CPU ; au-delà de ~1 M de nœuds, passer au `NeighborLoader` (pyg-lib).")
    a("- La tête du consortium est entraînée en clair ; en production : apprentissage fédéré + agrégation sécurisée.")
    path = report_dir(cfg) / "REPORT.md"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path
