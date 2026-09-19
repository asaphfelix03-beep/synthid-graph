"""Baselines : moteur de règles (l'existant) et LightGBM (tabulaire seul, puis + features de graphe)."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

RULES = {
    "telephone_partage_>=2": lambda d: d["g_share_phone"] >= 2,
    "appareil_partage_>=2": lambda d: d["g_share_device"] >= 2,
    "adresse_domiciliation": lambda d: (d["app_address_mail_drop"] + d["app_address_commercial"]) > 0,
    "email_cree_<60j": lambda d: d["app_email_age_at_onboard"] < 60,
    "telephone_voip": lambda d: d["app_phone_voip"] == 1,
    "ip_datacenter_vpn": lambda d: (d["app_ip_datacenter"] + d["app_ip_vpn"]) > 0,
    "nid_plusieurs_identites": lambda d: d["g_nid_identity_mismatch"] >= 1,
    "hausses_plafond_>=3": lambda d: d["beh_lir_count"] >= 3,
    "email_jetable": lambda d: d["app_email_disposable"] == 1,
    "emulateur": lambda d: d["app_device_emulator"] == 1,
}


def rules_score(df: pd.DataFrame) -> np.ndarray:
    hits = sum(rule(df).fillna(False).astype(int) for rule in RULES.values())
    tie_break = 0.01 * np.log1p(df["g_share_phone"] + df["g_share_device"])
    return (hits + tie_break).to_numpy(dtype=float)


LGBM_PARAMS = dict(n_estimators=2000, learning_rate=0.03, num_leaves=31, min_child_samples=30, subsample=0.8,
                   subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)


def train_lgbm(train: pd.DataFrame, val: pd.DataFrame, cols: list[str], seed: int) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(random_state=seed, **LGBM_PARAMS)
    model.fit(train[cols], train["y"], eval_set=[(val[cols], val["y"])], eval_metric="average_precision",
              callbacks=[lgb.early_stopping(100, verbose=False)])
    return model


def lgbm_contributions(model: lgb.LGBMClassifier, X: pd.DataFrame, top: int = 5) -> list[list[tuple[str, float]]]:
    """Explications locales (valeurs SHAP natives de LightGBM) : les `top` features qui poussent le score."""
    contrib = model.booster_.predict(X, pred_contrib=True)[:, :-1]
    cols = np.array(X.columns)
    out = []
    for row in contrib:
        order = np.argsort(-row)[:top]
        out.append([(cols[i], float(row[i])) for i in order if row[i] > 0])
    return out
