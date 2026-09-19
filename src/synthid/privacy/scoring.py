"""Scoring chiffré de la « tête » du modèle de consortium (régression logistique) avec Paillier.

La banque chiffre son vecteur de features sous SA clé publique ; le consortium calcule Enc(w·x + b) sans voir x
(Enc(x_i)^w_i, puis produit) ; la banque déchiffre le logit. Le consortium ne voit pas les features, la banque ne
voit pas les poids (seulement le score). Limites : Paillier ne permet que des fonctions linéaires des données
chiffrées (la sigmoïde est appliquée en clair par la banque) ; un modèle profond demanderait CKKS/TFHE.
En production, la tête serait entraînée par apprentissage fédéré ; ici elle l'est en clair sur le fold d'entraînement.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from . import paillier

HEAD_FEATURES = ["xb_nid_mismatch", "xb_phone_mismatch", "xb_email_mismatch", "xb_device_mismatch",
                 "xb_identity_other_banks", "xb_phone_other_banks", "xb_device_other_banks",
                 "app_email_age_at_onboard", "app_file_age", "beh_lir_rate", "beh_pay_full_share", "g_share_device"]


def encrypted_scoring_demo(train: pd.DataFrame, test: pd.DataFrame, cfg: dict, sample: int, seed: int) -> dict:
    cols = [c for c in HEAD_FEATURES if c in train.columns]
    med = train[cols].median()
    scaler = StandardScaler().fit(train[cols].fillna(med))
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(scaler.transform(train[cols].fillna(med)), train["y"])
    Xte = scaler.transform(test[cols].fillna(med))
    clear_logits = clf.decision_function(Xte)

    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(test["y"].to_numpy() == 1)
    neg = np.flatnonzero(test["y"].to_numpy() == 0)
    pick = np.concatenate([rng.choice(pos, min(len(pos), sample // 2), replace=False),
                           rng.choice(neg, sample - min(len(pos), sample // 2), replace=False)])
    pk, sk = paillier.keypair(cfg["privacy"]["paillier_bits"])
    n = pk.n
    w_fixed = [paillier.encode_fixed(w, n) for w in clf.coef_[0]]
    bias = paillier.encode_fixed(float(clf.intercept_[0]) * 2**paillier.FRAC, n)

    t0 = time.perf_counter()
    flat = [paillier.encode_fixed(float(v), n) for i in pick for v in Xte[i]]
    enc = paillier.encrypt_raw(n, flat)                                  # côté banque
    t_enc = time.perf_counter() - t0
    d = len(cols)
    t0 = time.perf_counter()
    enc_bias = paillier.encrypt_raw(n, [bias], workers=1)[0]
    results = []
    for j in range(len(pick)):                                           # côté consortium
        acc = enc_bias
        for i in range(d):
            acc = paillier.add(n, acc, paillier.scalar_mul(n, enc[j * d + i], w_fixed[i]))
        results.append(acc)
    t_eval = time.perf_counter() - t0
    t0 = time.perf_counter()
    enc_logits = np.array([paillier.decode_fixed(paillier.decrypt_raw(sk, c), n, 2 * paillier.FRAC) for c in results])
    t_dec = time.perf_counter() - t0

    err = np.abs(enc_logits - clear_logits[pick])
    n_req = len(pick)
    ct = (2 * n.bit_length() + 7) // 8
    return {
        "features": cols,
        "head_pr_auc_clear_full_test": float(average_precision_score(test["y"], clear_logits)),
        "sample_size": int(n_req),
        "max_abs_error_logit": float(err.max()),
        "ranking_identical_on_sample": bool((np.argsort(enc_logits) == np.argsort(clear_logits[pick])).all()),
        "bank_encrypt_ms_per_request": round(1000 * t_enc / n_req, 1),
        "consortium_eval_ms_per_request": round(1000 * t_eval / n_req, 1),
        "bank_decrypt_ms_per_request": round(1000 * t_dec / n_req, 1),
        "total_ms_per_request": round(1000 * (t_enc + t_eval + t_dec) / n_req, 1),
        "bytes_per_request": int((d + 1) * ct),
        "paillier_modulus_bits": int(n.bit_length()),
    }
