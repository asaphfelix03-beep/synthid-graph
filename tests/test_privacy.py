import numpy as np
import pytest

from synthid.privacy import oprf, paillier
from synthid.privacy.consortium import run_consortium


def test_oprf_tokens_are_consistent_and_blinded():
    service = oprf.TokenizationService()
    a = oprf.tokenize(["phone|+33612345678", "nid|1850175123456"], service)
    b = oprf.tokenize(["phone|+33612345678"], service)
    assert a["phone|+33612345678"] == b["phone|+33612345678"], "même jeton pour toutes les banques"
    h1 = oprf.hash_to_curve(b"phone|+33612345678").format()
    assert h1 not in service.received, "le service ne voit jamais H1(x) sans aveuglement"
    other = oprf.tokenize(["phone|+33612345678"], oprf.TokenizationService())
    assert other["phone|+33612345678"] != a["phone|+33612345678"], "le jeton dépend de la clé du service"


def test_oprf_rate_limit():
    service = oprf.TokenizationService(max_evaluations=2)
    with pytest.raises(PermissionError):
        oprf.tokenize(["a", "b", "c"], service)


def test_packed_paillier_sums_are_exact():
    pk, sk = paillier.keypair(1024)
    n = pk.n
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, 500, 300).tolist(), rng.integers(0, 500, 300).tolist()
    ca, cb = paillier.encrypt_raw(n, paillier.pack(a, n, 24)), paillier.encrypt_raw(n, paillier.pack(b, n, 24))
    total = [paillier.add(n, x, y) for x, y in zip(ca, cb)]
    out = paillier.unpack([paillier.decrypt_raw(sk, c) for c in total], 300, 24, n)
    assert out == [x + y for x, y in zip(a, b)]
    with pytest.raises(OverflowError):
        paillier.pack([2**24], n, 24)


def test_fixed_point_linear_score():
    pk, sk = paillier.keypair(1024)
    n = pk.n
    x, w, b = [0.5, -1.25, 2.0], [0.3, 0.8, -0.1], 0.05
    enc = paillier.encrypt_raw(n, [paillier.encode_fixed(v, n) for v in x], workers=1)
    acc = paillier.encrypt_raw(n, [paillier.encode_fixed(b * 2**paillier.FRAC, n)], workers=1)[0]
    for c, wi in zip(enc, w):
        acc = paillier.add(n, acc, paillier.scalar_mul(n, c, paillier.encode_fixed(wi, n)))
    got = paillier.decode_fixed(paillier.decrypt_raw(sk, acc), n, 2 * paillier.FRAC)
    assert abs(got - (np.dot(x, w) + b)) < 1e-9


def test_consortium_round_is_exact_and_leak_free(tiny_world, tiny_cfg):
    sim, _, labels = tiny_world
    feats, metrics = run_consortium(sim, tiny_cfg, [720])
    assert all(r["max_abs_error"] == 0 for r in metrics["rounds"].values())
    lk = metrics["leakage"]
    assert lk["raw_pii_values_received_by_hub"] == 0
    assert lk["hub_strings_not_tokens"] == 0
    assert lk["unblinded_points_received_by_tokenization_service"] == 0
    assert lk["dictionary_attack_recovery_oprf"] == 0
    assert lk["dictionary_attack_recovery_naive_sha256"] == 1.0
    xb = feats[720].set_index("person_id")
    syn = labels.set_index("person_id")["is_synthetic"].reindex(xb.index)
    # les identités synthétiques réutilisent davantage leurs attributs sous d'autres identités ailleurs
    assert xb.loc[syn == 1, "xb_mismatch_total"].mean() > xb.loc[syn == 0, "xb_mismatch_total"].mean()
