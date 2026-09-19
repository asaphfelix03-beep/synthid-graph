import copy
from pathlib import Path

import pytest

from synthid.bank_view import build_bank_view, build_labels
from synthid.config import load_config
from synthid.simulate import simulate


@pytest.fixture(scope="session")
def tiny_cfg():
    cfg = copy.deepcopy(load_config(Path(__file__).resolve().parent.parent / "configs" / "small.yaml"))
    cfg["name"] = "pytest"
    cfg["population"].update(identities=1500, n_employers=80, n_merchants=300, cgnat_ips=20, public_wifi_ips=5,
                             dorm_buildings=1, tradeline_sellers_per_bank=2, mules_per_bank=4)
    cfg["fraud"]["rings"] = 14
    cfg["privacy"]["paillier_bits"] = 1024  # tests rapides ; la config réelle utilise 3072 bits
    return cfg


@pytest.fixture(scope="session")
def tiny_world(tiny_cfg):
    sim = simulate(tiny_cfg)
    return sim, build_bank_view(sim, tiny_cfg), build_labels(sim)
