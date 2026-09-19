"""Correction point-in-time : supprimer tout ce qui se passe après T ne doit changer aucune feature calculée à T."""
import numpy as np
import pandas as pd
import pytest

from synthid.features import compute_features, take_snapshot
from synthid.simulate import MONTH


def truncate(bank: dict[str, pd.DataFrame], T: int) -> dict[str, pd.DataFrame]:
    b = {k: v.copy() for k, v in bank.items()}
    p = b["persons"]
    b["persons"] = p[p["onboard_day"] <= T].assign(close_day=p["close_day"].where(p["close_day"] <= T, 10**9))
    pa = b["person_attr"]
    b["person_attr"] = pa[pa["first_day"] <= T].assign(last_day=pa["last_day"].clip(upper=T))
    b["paid_at"] = b["paid_at"][b["paid_at"]["first_day"] <= T]
    b["transfers"] = b["transfers"][(b["transfers"]["month"] + 1) * MONTH <= T]
    b["authorized_users"] = b["authorized_users"][b["authorized_users"]["first_day"] <= T]
    b["similar"] = b["similar"][b["similar"]["first_day"] <= T]
    b["monthly"] = b["monthly"][(b["monthly"]["month"] + 1) * MONTH <= T]
    b["fraud_confirmations"] = b["fraud_confirmations"][b["fraud_confirmations"]["confirmed_day"] <= T]
    return b


@pytest.mark.parametrize("T", [360, 720])
def test_features_do_not_use_the_future(tiny_world, tiny_cfg, T):
    _, bank, _ = tiny_world
    full, _, _ = compute_features(take_snapshot(bank, T), tiny_cfg)
    cut, _, _ = compute_features(take_snapshot(truncate(bank, T), T), tiny_cfg)
    cols = [c for c in full.columns if c.startswith(("app_", "beh_", "g_"))]
    pd.testing.assert_frame_equal(full[["person_id"] + cols].reset_index(drop=True),
                                  cut[["person_id"] + cols].reset_index(drop=True), check_exact=False, rtol=1e-9)


def test_scored_population_excludes_closed_accounts(tiny_world):
    _, bank, _ = tiny_world
    s = take_snapshot(bank, 720)
    assert not (s.persons["scored"] & (s.persons["close_day"] <= 720)).any()
    assert not (s.persons["known_fraud"] & s.persons["scored"]).any(), "une fraude confirmée est déjà clôturée"
    assert np.isfinite(s.persons["onboard_day"]).all()
