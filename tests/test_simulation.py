import numpy as np

from synthid.normalize import NORMALIZERS, norm_address, norm_email, norm_phone
from synthid.simulate import FAR


def test_phone_formats_normalize_to_e164():
    assert norm_phone("06 12 34 56 78") == norm_phone("+33612345678") == norm_phone("0612345678") == "+33612345678"


def test_email_and_address_normalization():
    assert norm_email("  Jean.Dupont@Gmail.com ") == "jean.dupont@gmail.com"
    assert norm_address("12, Rue de l'Église 75001 PARIS") == norm_address("12 rue de l eglise 75001 paris")


def test_raw_values_normalize_to_canonical(tiny_world):
    sim, _, _ = tiny_world
    ra = sim["record_attr"]
    normed = [NORMALIZERS[t](raw) for t, raw in zip(ra["attr_type"], ra["raw"])]
    assert (np.array(normed) == ra["value"].to_numpy()).all()


def test_synthetic_lifecycle_is_consistent(tiny_world):
    _, _, labels = tiny_world
    syn = labels[labels["is_synthetic"] == 1]
    assert len(syn) > 0
    assert (syn["onboard_day"] + 180 <= syn["bust_out_day"]).all(), "culture d'au moins 6 mois avant bust-out"
    assert (syn["bust_out_end"] >= syn["bust_out_day"]).all()
    assert (syn["label_day"] > syn["bust_out_end"]).all(), "la fraude n'est connue qu'après le bust-out"
    legit = labels[labels["is_synthetic"] == 0]
    assert (legit["bust_out_day"] == FAR).all()


def test_no_clear_pii_in_bank_tables(tiny_world):
    sim, bank, _ = tiny_world
    ident = sim["identities"]
    pii = set(ident["nid"]) | set(ident["phone"]) | set(ident["email"]) | set(ident["address"]) \
        | set(sim["record_attr"]["raw"]) | set(ident["first_name"] + " " + ident["last_name"])
    for name, df in bank.items():
        for col in df.columns:
            if df[col].dtype == object:
                leaked = pii & set(df[col].dropna().astype(str))
                assert not leaked, f"PII en clair dans bank/{name}.{col} : {list(leaked)[:3]}"


def test_hmac_tokens_are_bank_specific(tiny_world):
    sim, bank, _ = tiny_world
    ra = sim["record_attr"].assign(bank=sim["record_attr"]["person_id"].str[0], token=bank["person_attr"]["token"])
    multi = ra[ra["attr_type"] == "nid"].groupby("value")["bank"].nunique()
    shared_value = multi[multi > 1].index
    if len(shared_value):
        sub = ra[(ra["attr_type"] == "nid") & ra["value"].isin(shared_value)]
        assert (sub.groupby("value")["token"].nunique() == sub.groupby("value")["bank"].nunique()).all()
