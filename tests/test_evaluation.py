import numpy as np
import pandas as pd

from synthid.evaluation import assign_folds, prevalence_weights, ranking_metrics, weighted_selection


def test_folds_are_group_disjoint_and_holdout_is_test_only(tiny_world, tiny_cfg):
    _, _, labels = tiny_world
    folds = assign_folds(labels, tiny_cfg)
    per_group = pd.DataFrame({"group": labels["group"], "fold": folds.values}).groupby("group")["fold"].nunique()
    assert (per_group == 1).all(), "un anneau / foyer n'appartient qu'à un seul fold"
    holdout = labels["scenario"].isin(tiny_cfg["fraud"]["holdout_scenarios"])
    assert (folds[holdout.values] == "test").all()
    first_bust = labels.groupby("group")["bust_out_day"].transform("min")
    late = (labels["is_synthetic"] == 1) & (first_bust > tiny_cfg["evaluation"]["label_cutoff"])
    assert (folds[late.values] == "test").all(), "un anneau non encore démasqué ne peut pas servir à l'entraînement"


def test_prevalence_weights():
    y = np.array([1] * 10 + [0] * 90)
    w = prevalence_weights(y, 0.01)
    assert abs(w[y == 1].sum() / w.sum() - 0.01) < 1e-9


def test_ranking_metrics_perfect_ranking(tiny_cfg):
    y = np.array([1, 1] + [0] * 198)
    perfect = ranking_metrics(y, np.linspace(1, 0, 200), tiny_cfg)
    assert perfect["pr_auc"] == 1.0 and perfect["recall_at_k"] == 1.0
    # budget (2 %) = 2 × prévalence (1 %) : même un classement parfait alerte autant de négatifs que de positifs
    assert perfect["alerted_rows"] == 4 and perfect["precision_at_k"] == 0.5


def test_weighted_selection_respects_budget():
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.1).astype(int)
    score = rng.random(5000)
    sel = weighted_selection(y, score, 0.02, 0.01)
    w = prevalence_weights(y, 0.01)
    assert w[sel].sum() <= 0.02 * w.sum() + w.max()
