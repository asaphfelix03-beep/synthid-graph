> Rapport généré par `python -m synthid run --config configs/small.yaml` (graine 42, données 100 % synthétiques).
> Une nouvelle exécution le régénère dans `reports/small/REPORT.md` (à de légères variations près pour les GNN, dont l'entraînement sur CPU n'est pas strictement déterministe).

# Rapport du PoC — détection d'identités synthétiques (config « small »)

> Rapport généré automatiquement par `synthid report`. Données 100 % synthétiques : les valeurs absolues reflètent les hypothèses du générateur. Ce qui compte : les écarts entre modèles, la tenue par niveau de sophistication et sur le scénario jamais vu.

## 1. Synthèse

- Meilleur modèle (PR-AUC au snapshot J+720) : **Ensemble inter-bancaire + GraphSAGE** — PR-AUC 0.899, PR-AUC ramenée à 1 % de prévalence 0.765.
- Anneaux détectés avant bust-out : **94.7 %**, délai médian d'anticipation **212 jours**, exposition (plafonds) couverte : 80.0 %.
- Critères go/no-go satisfaits : **9/10** (détail §9, lecture §11).

## 2. Jeu de données simulé

| Élément | Valeur |
|---|---|
| Identités | 20 684 |
| Comptes bancaires (3 banques) | 25 891 |
| Comptes d'identités synthétiques | 825 |
| Anneaux de fraude | 110 |
| Anneaux répartis sur plusieurs banques | 66 |
| Arêtes personne → attribut | 213 458 |
| Arêtes compte → marchand | 418 601 |
| Virements (agrégats mensuels) | 227 215 |
| Quasi-doublons (résolution d'entités) | 70 |
| Confirmations de fraude datées | 655 |
| Anneaux par niveau (1/2/3) | 30 / 48 / 32 |
| Anneaux par scénario | partial : 48, overt : 30, circular_transfers : 11, complicit_merchant : 8, shared_employer : 7, tradeline : 6 |

Lignes d'apprentissage : train 7901 (306 positifs), validation 2536 (60), test 8116 (965) sur 3 snapshots.

### Audit de raccourcis (jalon de la phase 2)

AUC maximale d'une feature isolée : **0.817** (seuil 0.9) → ✅ réussi.

| Feature | AUC univariée |
|---|---|
| `app_phone_age_at_onboard` | 0.817 |
| `app_phone_voip` | 0.800 |
| `g_share_device` | 0.791 |
| `app_income` | 0.790 |
| `app_email_age_at_onboard` | 0.789 |
| `beh_pay_std6` | 0.772 |
| `beh_pay_full_share` | 0.768 |
| `beh_pay_mean6` | 0.757 |
| `app_file_age` | 0.755 |
| `g_share_phone` | 0.749 |

## 3. Performance des modèles (snapshot J+720, population : comptes < 24 mois, pas encore en bust-out)

| Modèle | PR-AUC | PR-AUC @1 % | ROC-AUC | Precision@k | Recall@k | Recall @FPR 1 % | Anneaux détectés | Délai médian |
|---|---|---|---|---|---|---|---|---|
| Moteur de règles (existant) | 0.722 | 0.623 | 0.836 | 32.4 % | 64.4 % | 64.0 % | 70.2 % | 210 j |
| LightGBM — tabulaire seul | 0.782 | 0.594 | 0.884 | 32.8 % | 65.4 % | 63.1 % | 73.7 % | 207 j |
| LightGBM — tabulaire + features de graphe | 0.881 | 0.700 | 0.946 | 38.4 % | 75.8 % | 74.7 % | 86.0 % | 212 j |
| GCN (projection homogène) | 0.817 | 0.585 | 0.913 | 34.5 % | 68.4 % | 67.7 % | 77.2 % | 210 j |
| GraphSAGE hétérogène — entrées tabulaires | 0.806 | 0.679 | 0.885 | 34.3 % | 67.9 % | 67.2 % | 84.2 % | 204 j |
| GraphSAGE hétérogène — entrées + features de graphe | 0.861 | 0.695 | 0.935 | 37.0 % | 73.7 % | 72.7 % | 91.2 % | 201 j |
| Ensemble LightGBM-graphe + GraphSAGE | 0.893 | 0.765 | 0.950 | 38.9 % | 77.3 % | 76.3 % | 94.7 % | 211 j |
| LightGBM + features inter-bancaires chiffrées | 0.894 | 0.721 | 0.956 | 38.4 % | 75.8 % | 74.7 % | 96.5 % | 214 j |
| Ensemble inter-bancaire + GraphSAGE | 0.899 | 0.765 | 0.958 | 38.4 % | 75.8 % | 75.5 % | 94.7 % | 212 j |

Budget d'alertes : 2.0 % de la population, évalué comme si la prévalence était de 1.0 % (négatifs repondérés ; prévalence réelle du jeu de test : 14.4 %, 2759 comptes). Precision@k, Recall@k et « PR-AUC @1 % » suivent cette convention.

## 4. Robustesse par niveau de sophistication et scénario

| Modèle | Recall@k niveau 1 | niveau 2 | niveau 3 | anneaux « circular_transfers » | anneaux « complicit_merchant » | anneaux « overt » | anneaux « partial » | anneaux « shared_employer » | anneaux « tradeline » |
|---|---|---|---|---|---|---|---|---|---|
| Moteur de règles (existant) | 100.0 % | 92.6 % | 2.2 % | 0.0 % | 33.3 % | 100.0 % | 100.0 % | 0.0 % | 25.0 % |
| LightGBM — tabulaire seul | 96.4 % | 97.5 % | 4.4 % | 0.0 % | 50.0 % | 100.0 % | 100.0 % | 33.3 % | 25.0 % |
| LightGBM — tabulaire + features de graphe | 98.6 % | 98.3 % | 31.9 % | 28.6 % | 50.0 % | 100.0 % | 100.0 % | 100.0 % | 100.0 % |
| GCN (projection homogène) | 97.9 % | 95.0 % | 14.1 % | 14.3 % | 0.0 % | 100.0 % | 100.0 % | 66.7 % | 100.0 % |
| GraphSAGE hétérogène — entrées tabulaires | 99.3 % | 98.3 % | 8.1 % | 14.3 % | 66.7 % | 100.0 % | 100.0 % | 66.7 % | 100.0 % |
| GraphSAGE hétérogène — entrées + features de graphe | 100.0 % | 90.1 % | 31.9 % | 42.9 % | 83.3 % | 100.0 % | 100.0 % | 100.0 % | 100.0 % |
| Ensemble LightGBM-graphe + GraphSAGE | 100.0 % | 96.7 % | 36.3 % | 71.4 % | 83.3 % | 100.0 % | 100.0 % | 100.0 % | 100.0 % |
| LightGBM + features inter-bancaires chiffrées | 100.0 % | 100.0 % | 28.9 % | 100.0 % | 83.3 % | 100.0 % | 100.0 % | 100.0 % | 75.0 % |
| Ensemble inter-bancaire + GraphSAGE | 100.0 % | 96.7 % | 31.9 % | 71.4 % | 83.3 % | 100.0 % | 100.0 % | 100.0 % | 100.0 % |

Scénario hold-out (jamais vu à l'entraînement) : complicit_merchant.

## 5. Faux positifs structurels (taux d'alerte, snapshot J+720)

| Groupe | Effectif | Moteur de règles (existant) | LightGBM — tabulaire + features de graphe | Ensemble inter-bancaire + GraphSAGE |
|---|---|---|---|---|
| Tous les clients légitimes | — | 1.4 % | 1.2 % | 1.2 % |
| foyer 3plus | 1439 | 1.3 % | 1.0 % | 0.8 % |
| colocation | 200 | 2.0 % | 3.0 % | 3.0 % |
| residence etudiante | 39 | 0.0 % | 0.0 % | 0.0 % |
| personne agee aidee | 41 | 7.3 % | 0.0 % | 0.0 % |
| defaut credit legitime | 40 | 0.0 % | 0.0 % | 0.0 % |
| primo arrivant | 176 | 6.8 % | 4.5 % | 3.4 % |
| mule | 6 | 0.0 % | 16.7 % | 0.0 % |

## 6. Alertes par communauté (phase 6)

Modèle : Ensemble inter-bancaire + GraphSAGE — même budget d'alertes que ci-dessus.

| Stratégie | Comptes alertés | Précision (prév. 1 %) | Recall | Anneaux touchés |
|---|---|---|---|---|
| Top-k individuel | 329 | 38.4 % | 75.8 % | 91.2 % |
| Communautés (score d'anneau) | 126 | 11.1 % | 21.5 % | 42.1 % |

NMI communautés Louvain ↔ anneaux réels (identités synthétiques) : 0.616.

Dossiers d'alerte générés : 10 (dossier `reports/small/alertes/`).

## 7. Couche inter-bancaire chiffrée

| Modèle | PR-AUC | Recall@k | Anneaux détectés | Anneaux niveau 3 |
|---|---|---|---|---|
| LightGBM — tabulaire + features de graphe | 0.881 | 75.8 % | 86.0 % | 60.0 % |
| LightGBM + features inter-bancaires chiffrées | 0.894 | 75.8 % | 96.5 % | 90.0 % |
| Ensemble inter-bancaire + GraphSAGE | 0.899 | 75.8 % | 94.7 % | 85.0 % |

| Mesure | Valeur |
|---|---|
| Paillier | 3072 bits ≈ 128 bits de sécurité, 191 compteurs par chiffré (packing) |
| OPRF (secp256k1) | 240 946 évaluations, 0.72 ms/valeur |
| Chiffrés échangés (7 snapshots) | 8 832 (6.8 Mo) |
| Slots réels / leurres | 558 872 / 282 698 |
| Exactitude des agrégats chiffrés | erreur max 0 |
| PII en clair reçues par le hub | 0 |
| Points non aveuglés reçus par le service OPRF | 0 |
| Attaque par dictionnaire du hub — jetons SHA-256 naïfs | 100.0 % des valeurs retrouvées |
| Attaque par dictionnaire du hub — jetons OPRF | 0.0 % |
| Scoring chiffré (tête logistique, 12 features) | erreur max 1.8e-12, 303 ms et 9.8 Ko par requête |
| Durée totale de la couche | 353.1 s |

Fuite résiduelle assumée : le hub apprend quels jetons pseudonymes sont présents dans au moins deux banques (pas les PII, ni les volumes). Les banques contributrices reçoivent leurs jetons mélangés à des leurres.

## 8. Neo4j

Chargement : 198 313 nœuds, 682 467 relations en 113.4 s.

| Requête d'investigation | Résultat | Latence |
|---|---|---|
| paires partageant 2 attributs forts | 1890 | 3515.5 ms |
| nid portes par plusieurs identites | 141 | 209.3 ms |
| hotes tradeline 3 dependants plus | 12 | 412.2 ms |
| clients ouverts a 2 sauts d une fraude confirmee | 169 | 129.8 ms |

GDS : projection 149 215 nœuds / 268 942 relations ; WCC 16 408 composantes ; Louvain 16 411 communautés (modularité 0.9999).

## 9. Critères go/no-go (§3.6 du plan)

| Critère | Mesure | Cible | Statut |
|---|---|---|---|
| Le meilleur GNN (hsage_graph) bat le meilleur baseline (lgbm_graph) en PR-AUC | -0.020 | > +0.000 | ❌ |
| Niveau 3 (fraudeur sophistiqué) : le graphe (ensemble) fait mieux que le tabulaire seul | 36.3 % vs 4.4 % (Recall@k) | > tabulaire | ✅ |
| Anneaux détectés avant le bust-out (ensemble_xb) | 94.7 % | ≥ 50.0 % | ✅ |
| Délai médian d'anticipation | 212 j | ≥ 30 j | ✅ |
| Taux d'alerte sur les faux positifs structurels ≤ moteur de règles | 1.2 % vs 2.0 % | ≤ règles | ✅ |
| Scénario jamais vu (complicit_merchant) : détection non nulle | 83.3 % | > 0 % | ✅ |
| Gain des features inter-bancaires (lgbm_xb vs lgbm_graph) | +0.013 PR-AUC | > 0 | ✅ |
| Aucune PII en clair reçue par le hub ni le service de tokenisation | 0 / 0 | 0 / 0 | ✅ |
| Niveau de sécurité Paillier | 128 bits | ≥ 128 bits | ✅ |
| Scoring chiffré : erreur max / latence par requête | 1.8e-12 / 303 ms | ≤ 1e-06 / < 1000 ms | ✅ |

## 10. Importance des features (LightGBM + graphe, gain)

| Feature | Part du gain |
|---|---|
| `app_phone_voip` | 13.2 % |
| `app_phone_age_at_onboard` | 12.1 % |
| `g_share_phone` | 7.6 % |
| `app_income` | 5.4 % |
| `g_au_host_max_dependents` | 4.5 % |
| `g_share_employer` | 4.0 % |
| `app_email_age_at_onboard` | 3.6 % |
| `app_file_age` | 3.3 % |
| `beh_pay_mean6` | 3.1 % |
| `g_share_device` | 2.9 % |
| `app_tenure_days` | 2.9 % |
| `app_has_employer` | 2.7 % |
| `app_file_age_at_onboard` | 2.4 % |
| `beh_txn_mean6` | 2.4 % |
| `beh_limit_last` | 2.4 % |

## 11. Lecture des résultats

- Sans graphe, le fraudeur sophistiqué (niveau 3) est invisible : Recall@k 2.2 % pour les règles, 4.4 % pour LightGBM tabulaire. Les features de graphe le portent à 31.9 %.
- Seul, le meilleur GNN (GraphSAGE hétérogène — entrées + features de graphe) ne dépasse pas LightGBM + features de graphe (PR-AUC @1 % : 0.695 vs 0.700). Des features de graphe bien construites captent déjà l'essentiel de la structure locale.
- En revanche, le GNN apporte un signal complémentaire : l'ensemble LightGBM-graphe + GraphSAGE atteint une PR-AUC @1 % de 0.765 et détecte 94.7 % des anneaux avant bust-out (vs 86.0 %).
- Les signaux inter-bancaires chiffrés font passer les anneaux détectés de 86.0 % à 96.5 %, et les anneaux de niveau 3 de 60.0 % à 90.0 %, sans échange de PII.
- Alerter par communauté entière dégrade la précision (11.1 % vs 38.4 % pour le top-k individuel) : les communautés Louvain mélangent anneaux et voisins légitimes. Recommandation : sélectionner les alertes au niveau individuel, puis regrouper les alertés par communauté pour construire les dossiers d'enquête (un dossier peut relier plusieurs anneaux d'un même opérateur).

## 12. Limites

- Données synthétiques : le modèle retrouve en partie les mécanismes codés dans le générateur (risque de circularité). Le scénario hold-out et les faux positifs structurels en donnent une mesure, pas une garantie.
- Prévalence enrichie pour la puissance statistique ; se référer à la PR-AUC ramenée à 1 %.
- Une seule graine : les écarts de quelques points entre modèles ne sont pas significatifs sans répétitions.
- Le GNN est entraîné en « full batch » sur CPU ; au-delà de ~1 M de nœuds, passer au `NeighborLoader` (pyg-lib).
- La tête du consortium est entraînée en clair ; en production : apprentissage fédéré + agrégation sécurisée.
