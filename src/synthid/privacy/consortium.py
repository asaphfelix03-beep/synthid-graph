"""Protocole inter-bancaire : réutilisation d'un téléphone / ID / email / appareil sous une AUTRE identité ailleurs.

Parties (modèle « honnête mais curieux », pas de collusion hub ↔ service de tokenisation) :
- Service de tokenisation (OPRF) : détient k, ne voit que des points aveuglés.
- Hub du consortium : voit des jetons pseudonymes et des chiffrés ; ne voit ni PII, ni compteurs.
- Banque Q (requérante) : possède sa paire de clés Paillier ; n'apprend que des totaux agrégés sur SES jetons.
- Banque B (contributrice) : reçoit une liste de ses jetons à renseigner, mélangée à des leurres ; elle ne peut
  donc pas savoir lesquels de ses clients sont aussi clients de Q.

Déroulé par snapshot T (point-in-time) :
 1. chaque banque tokenise ses clés (OPRF, fait une seule fois) : « type|valeur », « type|valeur|identité », « id|identité » ;
 2. elle envoie au hub l'ensemble de ses jetons présents à T (sans compteur) ;
 3. pour chaque Q, le hub construit une disposition de slots = jetons de Q présents dans ≥ 1 autre banque (+ leurres) ;
 4. chaque B chiffre ses compteurs dans cette disposition sous la clé publique de Q (Paillier « packé ») ;
 5. le hub multiplie les chiffrés (= somme des compteurs) et renvoie le résultat à Q, qui déchiffre.
Fuite résiduelle, assumée et documentée : le hub apprend quels jetons pseudonymes sont présents dans ≥ 2 banques.
"""
from __future__ import annotations

import hashlib
import re
import secrets
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..normalize import NORMALIZERS, identity_key
from . import oprf, paillier

XB_TYPES = ["nid", "phone", "email", "device"]


def ciphertext_bytes(n: int) -> int:
    return (2 * n.bit_length() + 7) // 8


def bank_pii(vault: dict[str, pd.DataFrame], bank: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """PII en clair d'UNE banque, telles qu'elle les détient dans son propre périmètre."""
    rec = vault["records"]
    rec = rec[rec["bank"] == bank][["person_id", "identity_id", "onboard_day"]]
    ident = vault["identities"].set_index("identity_id")
    rec = rec.assign(idkey=[identity_key(ident.at[i, "first_name"], ident.at[i, "last_name"], ident.at[i, "dob"])
                            for i in rec["identity_id"]])
    ra = vault["record_attr"]
    ra = ra[ra["attr_type"].isin(XB_TYPES) & ra["person_id"].isin(rec["person_id"])]
    ra = ra.merge(rec[["person_id", "onboard_day", "idkey"]], on="person_id")
    norm = [NORMALIZERS[t](v) for t, v in zip(ra["attr_type"], ra["raw"])]
    ra = ra.assign(attr_key=[f"{t}|{v}" for t, v in zip(ra["attr_type"], norm)])
    ra["pair_key"] = ra["attr_key"] + "|" + ra["idkey"]
    return ra[["person_id", "attr_type", "attr_key", "pair_key", "idkey", "first_day", "onboard_day"]], rec


@dataclass
class BankNode:
    name: str
    pii: pd.DataFrame
    records: pd.DataFrame
    bits: int
    tokens: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        self.pk, self.sk = paillier.keypair(self.bits)

    def all_keys(self) -> list[str]:
        ids = ("id|" + self.records["idkey"]).tolist()
        return sorted(set(self.pii["attr_key"]) | set(self.pii["pair_key"]) | set(ids))

    def live(self, T: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        pii = self.pii[(self.pii["first_day"] <= T) & (self.pii["onboard_day"] <= T)]
        recs = self.records[self.records["onboard_day"] <= T]
        return pii, recs

    def counts(self, T: int) -> dict[str, int]:
        pii, recs = self.live(T)
        c = pd.concat([pii.groupby("attr_key")["person_id"].nunique(), pii.groupby("pair_key")["person_id"].nunique(),
                       ("id|" + recs["idkey"]).value_counts()])
        return {self.tokens[k]: int(v) for k, v in c.items()}


class Hub:
    def __init__(self):
        self.inbox: list[tuple] = []  # journal de tout ce que le hub reçoit (tests de fuite)

    def receive(self, kind: str, sender: str, payload):
        self.inbox.append((kind, sender, payload))


def run_round(banks: list[BankNode], hub: Hub, T: int, cfg: dict, rng: np.random.Generator) -> tuple[dict, dict]:
    pc = cfg["privacy"]
    slot_bits, ratio = pc["slot_bits"], pc["decoy_ratio"]
    counts = {b.name: b.counts(T) for b in banks}
    sets = {name: set(c) for name, c in counts.items()}
    for b in banks:
        hub.receive("token_set", b.name, sorted(sets[b.name]))
    totals, stats = {}, {"ciphertexts": 0, "bytes": 0, "encrypt_s": 0.0, "decrypt_s": 0.0, "real_slots": 0, "decoy_slots": 0}
    for Q in banks:
        others = [B for B in banks if B is not Q]
        elsewhere = set().union(*(sets[B.name] for B in others))
        real = sorted(sets[Q.name] & elsewhere)
        slot_of = {tok: i for i, tok in enumerate(real)}
        n_slots, requests = len(real), {}
        real_set = set(real)
        for B in others:
            mine = [tok for tok in real if tok in sets[B.name]]
            pool = sorted(sets[B.name] - real_set)
            n_decoy = min(len(pool), int(ratio * len(mine)))
            decoys = [pool[i] for i in rng.choice(len(pool), size=n_decoy, replace=False)] if n_decoy else []
            assign = [(tok, slot_of[tok]) for tok in mine] + [(tok, n_slots + j) for j, tok in enumerate(decoys)]
            n_slots += n_decoy
            rng.shuffle(assign)
            requests[B.name] = assign
        n = Q.pk.n
        aggregate = None
        for B in others:
            vec = [0] * n_slots
            for tok, slot in requests[B.name]:
                vec[slot] = counts[B.name][tok]
            t = time.perf_counter()
            cts = paillier.encrypt_raw(n, paillier.pack(vec, n, slot_bits))
            stats["encrypt_s"] += time.perf_counter() - t
            hub.receive("ciphertexts", B.name, {"for": Q.name, "count": len(cts), "sample": cts[:1]})
            stats["ciphertexts"] += len(cts)
            stats["bytes"] += len(cts) * ciphertext_bytes(n)
            aggregate = cts if aggregate is None else [paillier.add(n, a, c) for a, c in zip(aggregate, cts)]
        t = time.perf_counter()
        plain = [paillier.decrypt_raw(Q.sk, c) for c in aggregate] if aggregate else []
        stats["decrypt_s"] += time.perf_counter() - t
        values = paillier.unpack(plain, len(real), slot_bits, n)
        totals[Q.name] = dict(zip(real, values))
        stats["real_slots"] += len(real)
        stats["decoy_slots"] += n_slots - len(real)
        # contrôle d'exactitude (possible uniquement dans le simulateur, qui voit tout)
        expected = {tok: sum(counts[B.name].get(tok, 0) for B in others) for tok in real}
        stats.setdefault("max_abs_error", 0)
        stats["max_abs_error"] = max(stats["max_abs_error"], max((abs(expected[t_] - totals[Q.name][t_]) for t_ in real), default=0))
    return totals, stats


def xb_features(bank: BankNode, totals: dict[str, int], T: int) -> pd.DataFrame:
    pii, recs = bank.live(T)
    tok = bank.tokens
    df = pii.assign(other=[totals.get(tok[k], 0) for k in pii["attr_key"]],
                    same_identity=[totals.get(tok[k], 0) for k in pii["pair_key"]])
    df["mismatch"] = df["other"] - df["same_identity"]
    out = pd.DataFrame({"person_id": recs["person_id"].values})
    for t in XB_TYPES:
        g = df[df["attr_type"] == t].groupby("person_id")[["other", "mismatch"]].max()
        out[f"xb_{t}_other_banks"] = out["person_id"].map(g["other"]).fillna(0).values
        out[f"xb_{t}_mismatch"] = out["person_id"].map(g["mismatch"]).fillna(0).values
    out["xb_identity_other_banks"] = [totals.get(tok["id|" + k], 0) for k in recs["idkey"]]
    out["xb_mismatch_total"] = out[[f"xb_{t}_mismatch" for t in XB_TYPES]].sum(axis=1)
    return out


def _log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}]   {msg}", flush=True)


def run_consortium(vault: dict[str, pd.DataFrame], cfg: dict, snapshots: list[int]) -> tuple[dict[int, pd.DataFrame], dict]:
    pc = cfg["privacy"]
    rng = np.random.default_rng(cfg["seed"] + 7)
    banks = []
    t = time.perf_counter()
    for name in cfg["banks"]:
        pii, recs = bank_pii(vault, name)
        banks.append(BankNode(name, pii, recs, pc["paillier_bits"]))
    keygen_s = time.perf_counter() - t
    _log(f"clés Paillier {pc['paillier_bits']} bits générées pour {len(banks)} banques ({keygen_s:.1f}s)")

    service, hub = oprf.TokenizationService(), Hub()
    t = time.perf_counter()
    for b in banks:
        b.tokens = oprf.tokenize(b.all_keys(), service)
        _log(f"OPRF banque {b.name} : {len(b.tokens)} jetons")
    oprf_s = time.perf_counter() - t

    features, rounds = {}, {}
    for T in snapshots:
        totals, st = run_round(banks, hub, T, cfg, rng)
        rounds[T] = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in st.items()}
        features[T] = pd.concat([xb_features(b, totals[b.name], T) for b in banks], ignore_index=True)
        _log(f"tour T={T} : {st['ciphertexts']} chiffrés, {st['real_slots']} slots réels, {st['encrypt_s']:.1f}s de chiffrement")

    n_bits = banks[0].pk.n.bit_length()
    metrics = {
        "paillier_modulus_bits": n_bits,
        "security_bits": paillier.SECURITY_BITS.get(pc["paillier_bits"]),
        "slots_per_ciphertext": paillier.slots_per_ciphertext(banks[0].pk.n, pc["slot_bits"]),
        "ciphertext_bytes": ciphertext_bytes(banks[0].pk.n),
        "keygen_seconds": round(keygen_s, 2),
        "oprf_evaluations": service.evaluations,
        "oprf_seconds": round(oprf_s, 1),
        "oprf_ms_per_value": round(1000 * oprf_s / max(1, service.evaluations), 3),
        "rounds": rounds,
        "leakage": leakage_checks(banks, hub, service),
    }
    return features, metrics


def leakage_checks(banks: list[BankNode], hub: Hub, service: oprf.TokenizationService, sample: int = 3000) -> dict:
    """Vérifications automatiques : ce que reçoivent le hub et le service, et attaque par dictionnaire du hub."""
    raw_values = set()
    for b in banks:
        raw_values |= {k.split("|", 1)[1] for k in b.pii["attr_key"]} | set(b.records["idkey"])
    hub_strings = [s for kind, _, p in hub.inbox if kind == "token_set" for s in p]
    non_token = sum(1 for s in hub_strings if not re.fullmatch(r"[0-9a-f]{32}", s))
    raw_in_hub = len(set(hub_strings) & raw_values)

    # le service ne doit recevoir que des points aveuglés, jamais H1(x)
    keys = [k for b in banks for k in b.all_keys()[:sample]]
    h1 = {oprf.hash_to_curve(k.encode()).format() for k in keys}
    unblinded_seen = len(h1 & set(service.received))

    # Attaque par dictionnaire : le hub connaît TOUTES les valeurs candidates (pire cas).
    candidates = sorted({k for b in banks for k in b.all_keys()})
    sha = lambda k: hashlib.sha256(k.encode()).hexdigest()[:32]  # noqa: E731
    naive_hub_view = {sha(k) for k in candidates}          # ce que verrait le hub avec des jetons SHA-256
    attacker_dict = {sha(c): c for c in candidates}
    naive_recovered = len(naive_hub_view & attacker_dict.keys()) / len(naive_hub_view)
    guessed_key = (secrets.randbelow(oprf.ORDER - 1) + 1).to_bytes(32, "big")  # sans k, la meilleure tentative
    guesses = {hashlib.sha256(b"synthid-oprf|" + c.encode() + b"|" +
                              oprf.hash_to_curve(c.encode()).multiply(guessed_key).format()).hexdigest()[:32]
               for c in candidates[:sample]}
    oprf_recovered = len(guesses & set(hub_strings)) / min(sample, len(candidates))
    return {
        "hub_strings_not_tokens": int(non_token),
        "raw_pii_values_received_by_hub": int(raw_in_hub),
        "unblinded_points_received_by_tokenization_service": int(unblinded_seen),
        "dictionary_attack_recovery_naive_sha256": round(naive_recovered, 4),
        "dictionary_attack_recovery_oprf": round(oprf_recovered, 4),
        "hub_messages": len(hub.inbox),
    }
