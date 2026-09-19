"""Périmètre bancaire : ce que chaque banque voit et stocke.

- Les PII sont normalisées puis tokenisées par HMAC-SHA256 avec une clé propre à chaque banque
  (PoC : dérivée de la graine ; production : clé en KMS/HSM). Aucune PII en clair n'entre dans le graphe.
- La résolution d'entités (quasi-doublons nom + date de naissance) se fait AVANT tokenisation,
  dans le périmètre de la banque, car un jeton HMAC ne permet pas de comparaison approximative.
- La vérité terrain (anneaux, rôles) est séparée dans `labels` : elle ne sert qu'à l'évaluation.
  La banque ne connaît que ses confirmations de fraude datées (`fraud_confirmations`).
"""
from __future__ import annotations

import hashlib
import hmac

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from .normalize import NORMALIZERS, identity_key, norm_name
from .simulate import FAR


def bank_key(seed: int, bank: str) -> bytes:
    return hashlib.sha256(f"poc-hmac-key|{seed}|{bank}".encode()).digest()


def hmac_token(key: bytes, text: str) -> str:
    return hmac.new(key, text.encode(), hashlib.sha256).hexdigest()[:20]


def build_bank_view(sim: dict[str, pd.DataFrame], cfg: dict) -> dict[str, pd.DataFrame]:
    seed = cfg["seed"]
    ident = sim["identities"].set_index("identity_id")
    rec = sim["records"]
    cat = sim["catalog"]
    keys = {b: bank_key(seed, b) for b in cfg["banks"]}
    bank_of = rec.set_index("person_id")["bank"]

    # --- arêtes personne → attribut, tokenisées
    ra = sim["record_attr"].copy()
    ra["bank"] = ra["person_id"].map(bank_of)
    norm = [NORMALIZERS[t](raw) for t, raw in zip(ra["attr_type"], ra["raw"])]
    ra["token"] = [hmac_token(keys[b], f"{t}:{v}") for b, t, v in zip(ra["bank"], ra["attr_type"], norm)]
    person_attr = ra[["person_id", "attr_type", "token", "first_day", "last_day"]].copy()

    # --- métadonnées des attributs (visibles par la banque : type d'IP, VoIP, email jetable…)
    # (le flag « employeur fictif » du catalogue n'est PAS repris : la banque ne le connaît pas)
    visible = ["attr_type", "value", "subtype", "domain", "disposable", "os", "emulator"]
    attr_meta = (ra[["attr_type", "token", "value"]].drop_duplicates("token")
                 .merge(cat[visible], on=["attr_type", "value"], how="left"))
    attr_meta["email_popular_domain"] = attr_meta["domain"].isin(
        ["gmail.com", "orange.fr", "hotmail.fr", "free.fr", "yahoo.fr", "outlook.fr", "laposte.net", "sfr.fr"])
    attr_meta = attr_meta.drop(columns=["domain", "value"])

    # --- table des personnes (informations de souscription + bureau de crédit)
    p = rec.merge(ident[["first_name", "last_name", "dob", "birth_year", "income", "file_start_day", "email", "phone",
                         "address"]], left_on="identity_id", right_index=True)
    email_meta = cat[cat.attr_type == "email"].set_index("value")
    phone_meta = cat[cat.attr_type == "phone"].set_index("value")
    addr_meta = cat[cat.attr_type == "address"].set_index("value")
    ip_meta = cat[cat.attr_type == "ip"].set_index("value")
    dev_meta = cat[cat.attr_type == "device"].set_index("value")
    persons = pd.DataFrame({
        "person_id": p["person_id"], "bank": p["bank"],
        "onboard_day": p["onboard_day"], "close_day": p["close_day"],
        "birth_year": p["birth_year"], "income": p["income"].round(-2),
        "file_start_day": p["file_start_day"],
        "id_token": [hmac_token(keys[b], "id:" + identity_key(first, last, dob))
                     for b, first, last, dob in zip(p["bank"], p["first_name"], p["last_name"], p["dob"])],
        "email_created_day": p["email"].map(email_meta["created_day"]).astype(float),
        "email_disposable": p["email"].map(email_meta["disposable"]).astype(bool),
        "phone_voip": p["phone"].map(phone_meta["subtype"]).eq("voip"),
        "phone_first_seen": p["phone"].map(phone_meta["first_seen"]).astype(float),
        "address_type": p["address"].map(addr_meta["subtype"]),
        "app_ip_type": p["app_ip"].map(ip_meta["subtype"]),
        "app_device_emulator": p["app_device"].map(dev_meta["emulator"]).astype(bool),
    })
    persons["email_popular_domain"] = p["email"].str.split("@").str[1].isin(
        ["gmail.com", "orange.fr", "hotmail.fr", "free.fr", "yahoo.fr", "outlook.fr", "laposte.net", "sfr.fr"]).values
    has_emp = set(sim["record_attr"].query("attr_type == 'employer'")["person_id"])
    persons["has_employer"] = persons["person_id"].isin(has_emp)

    # --- marchands (identifiants propres à chaque banque : pas de nœud partagé entre banques)
    pa = sim["paid_at"].copy()
    pa["bank"] = pa["person_id"].map(bank_of)
    pa["merchant_token"] = pa["bank"] + ":m" + pa["merchant_id"].astype(str)
    merchants = sim["merchants"].set_index("merchant_id")
    merchant_meta = pa[["merchant_token", "merchant_id"]].drop_duplicates("merchant_token")
    merchant_meta = merchant_meta.assign(mcc=merchant_meta["merchant_id"].map(merchants["mcc"]),
                                         high_risk_mcc=merchant_meta["merchant_id"].map(merchants["high_risk_mcc"]))
    paid_at = pa[["person_id", "merchant_token", "first_day", "last_day"]]

    # --- résolution d'entités intra-banque (blocage sur la date de naissance)
    similar = entity_resolution(p, cfg["graph"]["similarity_threshold"])

    # --- confirmations de fraude connues de la banque (datées) ; les bust-outs non reconnus restent des impayés
    fc = rec[(rec["bust_out_day"] < FAR) & rec["label_observed"]][["person_id", "label_day"]]
    fraud_confirmations = fc.rename(columns={"label_day": "confirmed_day"})

    return dict(persons=persons, person_attr=person_attr, attr_meta=attr_meta, paid_at=paid_at.reset_index(drop=True),
                merchant_meta=merchant_meta.drop(columns=["merchant_id"]).reset_index(drop=True),
                transfers=sim["transfers"], authorized_users=sim["authorized_users"], similar=similar,
                monthly=sim["monthly"], fraud_confirmations=fraud_confirmations.reset_index(drop=True))


def entity_resolution(p: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    p = p.assign(full=[f"{norm_name(first)} {norm_name(last)}" for first, last in zip(p["first_name"], p["last_name"])])
    for (_, _), g in p.groupby(["bank", "dob"]):
        if len(g) < 2:
            continue
        items = list(zip(g["person_id"], g["full"], g["onboard_day"]))
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                score = fuzz.ratio(items[i][1], items[j][1])
                if score >= threshold:
                    rows.append((items[i][0], items[j][0], max(items[i][2], items[j][2]), float(score)))
    return pd.DataFrame(rows, columns=["a", "b", "first_day", "score"])


def build_labels(sim: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Vérité terrain par enregistrement bancaire (jamais utilisée comme feature)."""
    rec = sim["records"]
    ident = sim["identities"].set_index("identity_id")
    lab = rec[["person_id", "bank", "identity_id", "onboard_day", "close_day", "bust_out_day", "bust_out_end",
               "label_day", "label_observed", "is_distressed"]].copy()
    for c in ["kind", "ring_id", "operator_id", "sophistication", "scenario", "household_id", "is_dorm",
              "is_colocation", "is_elderly_assisted", "is_newcomer", "is_tradeline_seller", "is_mule", "stolen_nid",
              "variant", "file_start_day"]:
        lab[c] = lab["identity_id"].map(ident[c])
    lab["is_synthetic"] = (lab["kind"] == "synthetic").astype(int)
    lab["group"] = np.where(lab["is_synthetic"] == 1, "r" + lab["ring_id"].astype(str),
                            "h" + lab["household_id"].astype(str))
    max_limit = sim["monthly"].groupby("person_id")["credit_limit"].max()
    lab["exposure"] = lab["person_id"].map(max_limit).fillna(0.0)
    return lab
