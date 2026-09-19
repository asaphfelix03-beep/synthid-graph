"""Snapshots « as-of T » et features.

Règle d'or (correction point-in-time) : une feature calculée à la date T n'utilise que des événements
datés ≤ T — arêtes dont `first_day ≤ T`, mois entièrement écoulés, confirmations de fraude datées ≤ T.
`tests/test_leakage.py` vérifie que tronquer toutes les données après T ne change aucune feature.

Familles de features :
- `app_*` : souscription et bureau de crédit (âge des contacts, type d'IP, dossier de crédit…)
- `beh_*` : comportement du compte (utilisation, remboursements, demandes de hausse de plafond…)
- `g_*`   : graphe local de la banque (attributs partagés, projection pondérée Adamic-Adar, communautés…)
- `xb_*`  : signaux inter-bancaires issus de la couche chiffrée (ajoutés par `privacy.consortium`)
"""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from .simulate import DAY0, MONTH

ATTR_TYPES = ["nid", "phone", "email", "address", "device", "ip", "employer"]
IP_TYPES = ["residential", "cgnat", "public_wifi", "office", "datacenter", "vpn"]
ADDRESS_TYPES = ["residential", "mail_drop", "commercial"]


@dataclass
class Snapshot:
    T: int
    persons: pd.DataFrame
    person_attr: pd.DataFrame
    attr_meta: pd.DataFrame
    paid_at: pd.DataFrame
    merchant_meta: pd.DataFrame
    transfers: pd.DataFrame
    authorized_users: pd.DataFrame
    similar: pd.DataFrame
    monthly: pd.DataFrame

    @property
    def index(self) -> pd.Series:
        return pd.Series(np.arange(len(self.persons)), index=self.persons["person_id"].values)


def take_snapshot(bank: dict[str, pd.DataFrame], T: int) -> Snapshot:
    P = bank["persons"]
    P = P[P["onboard_day"] <= T].reset_index(drop=True).copy()
    P["closed"] = P["close_day"] <= T
    fc = bank["fraud_confirmations"]
    P["known_fraud"] = P["person_id"].isin(fc.loc[fc["confirmed_day"] <= T, "person_id"])
    P["scored"] = ~P["closed"]
    ids = set(P["person_id"])

    pa = bank["person_attr"]
    pa = pa[(pa["first_day"] <= T) & pa["person_id"].isin(ids)]
    paid = bank["paid_at"]
    paid = paid[(paid["first_day"] <= T) & paid["person_id"].isin(ids)]
    tr = bank["transfers"]
    tr = tr[((tr["month"] + 1) * MONTH <= T) & tr["src"].isin(ids) & tr["dst"].isin(ids)]
    au = bank["authorized_users"]
    au = au[(au["first_day"] <= T) & au["person_id"].isin(ids) & au["host_person_id"].isin(ids)]
    sim = bank["similar"]
    sim = sim[(sim["first_day"] <= T) & sim["a"].isin(ids) & sim["b"].isin(ids)]
    mo = bank["monthly"]
    mo = mo[((mo["month"] + 1) * MONTH <= T) & mo["person_id"].isin(ids)]
    return Snapshot(T, P, pa, bank["attr_meta"], paid, bank["merchant_meta"], tr, au, sim, mo)


# --------------------------------------------------------------------------- app_
def application_features(s: Snapshot) -> pd.DataFrame:
    P, T = s.persons, s.T
    f = pd.DataFrame(index=P.index)
    f["app_age"] = DAY0.year + T / 365.25 - P["birth_year"]
    f["app_tenure_days"] = T - P["onboard_day"]
    f["app_file_age_at_onboard"] = P["onboard_day"] - P["file_start_day"]
    f["app_file_age"] = T - P["file_start_day"]
    f["app_income"] = P["income"]
    f["app_email_age_at_onboard"] = P["onboard_day"] - P["email_created_day"]
    f["app_email_disposable"] = P["email_disposable"].astype(int)
    f["app_email_popular_domain"] = P["email_popular_domain"].astype(int)
    f["app_phone_voip"] = P["phone_voip"].astype(int)
    f["app_phone_age_at_onboard"] = P["onboard_day"] - P["phone_first_seen"]
    for t in ADDRESS_TYPES:
        f[f"app_address_{t}"] = (P["address_type"] == t).astype(int)
    for t in IP_TYPES:
        f[f"app_ip_{t}"] = (P["app_ip_type"] == t).astype(int)
    f["app_device_emulator"] = P["app_device_emulator"].astype(int)
    f["app_has_employer"] = P["has_employer"].astype(int)
    return f


# --------------------------------------------------------------------------- beh_
def behavior_features(s: Snapshot) -> pd.DataFrame:
    P, m = s.persons, s.monthly
    idx = s.index
    out = pd.DataFrame(index=P.index)
    if m.empty:
        return out
    m = m.assign(pidx=m["person_id"].map(idx).values, pay_full=(m["pay_ratio"] >= 0.99).astype(float))
    m = m.sort_values(["pidx", "month"])
    g = m.groupby("pidx")
    life = pd.DataFrame({
        "beh_months": g.size(),
        "beh_lir_count": g["limit_increase_request"].sum(),
        "beh_lir_granted": g["limit_increase_granted"].sum(),
        "beh_late_count": g["late"].sum(),
        "beh_returned_count": g["returned_payment"].sum(),
        "beh_pay_full_share": g["pay_full"].mean(),
        "beh_limit_first": g["credit_limit"].first(),
        "beh_limit_last": g["credit_limit"].last(),
    })
    life["beh_lir_rate"] = life["beh_lir_count"] / life["beh_months"]
    life["beh_limit_growth"] = life["beh_limit_last"] / life["beh_limit_first"]
    recent = m[m["month"] >= s.T // MONTH - 6]
    g6 = recent.groupby("pidx")
    six = pd.DataFrame({
        "beh_util_mean6": g6["utilization"].mean(),
        "beh_util_std6": g6["utilization"].std(),
        "beh_util_max6": g6["utilization"].max(),
        "beh_pay_mean6": g6["pay_ratio"].mean(),
        "beh_pay_std6": g6["pay_ratio"].std(),
        "beh_txn_mean6": g6["n_txn"].mean(),
        "beh_mcc_mean6": g6["n_mcc"].mean(),
        "beh_cash_share6": g6["cash_advance"].sum() / g6["spend"].sum().replace(0, np.nan),
    })
    out = out.join(life.drop(columns=["beh_limit_first"])).join(six)
    return out


# ---------------------------------------------------------------------------- g_
def _incidence(pidx: np.ndarray, keys: pd.Series, n: int) -> tuple[sp.csr_matrix, np.ndarray]:
    codes, uniques = pd.factorize(keys)
    B = sp.csr_matrix((np.ones(len(pidx)), (pidx, codes)), shape=(n, len(uniques)))
    B.data[:] = 1.0
    deg = np.asarray(B.sum(axis=0)).ravel()
    return B, deg


def projection(s: Snapshot, cfg: dict) -> sp.csr_matrix:
    """Graphe personne–personne : attributs partagés (poids Adamic-Adar 1/log(deg), super-nœuds exclus),
    petits marchands en commun, virements, utilisateurs autorisés et quasi-doublons."""
    n = len(s.persons)
    idx = s.index
    cap, mcap = cfg["graph"]["projection_degree_cap"], cfg["graph"]["merchant_degree_cap"]
    P = sp.csr_matrix((n, n))
    pa = s.person_attr.drop_duplicates(["person_id", "token"])
    for frame, key, limit in ((pa, "token", cap), (s.paid_at.drop_duplicates(["person_id", "merchant_token"]), "merchant_token", mcap)):
        if frame.empty:
            continue
        B, deg = _incidence(frame["person_id"].map(idx).values, frame[key], n)
        keep = np.flatnonzero((deg >= 2) & (deg <= limit))
        Bk = B[:, keep]
        P = P + (Bk @ sp.diags(1.0 / np.log(deg[keep])) @ Bk.T).tocsr()
    pairs = [s.transfers[["src", "dst"]].drop_duplicates().to_numpy(),
             s.authorized_users[["person_id", "host_person_id"]].to_numpy(),
             s.similar[["a", "b"]].to_numpy()]
    for arr in pairs:
        if len(arr):
            a, b = idx.loc[arr[:, 0]].values, idx.loc[arr[:, 1]].values
            E = sp.csr_matrix((np.ones(len(a)), (a, b)), shape=(n, n))
            E.data[:] = 1.0
            P = P + E + E.T
    P = P.tolil()
    P.setdiag(0)
    P = P.tocsr()
    P.eliminate_zeros()
    return P


def pagerank(W: sp.csr_matrix, d: float = 0.85, iters: int = 50) -> np.ndarray:
    n = W.shape[0]
    out_w = np.asarray(W.sum(axis=1)).ravel()
    inv = np.divide(1.0, out_w, out=np.zeros_like(out_w), where=out_w > 0)
    M = (sp.diags(inv) @ W).T.tocsr()
    pr = np.full(n, 1.0 / n)
    dangling = out_w == 0
    for _ in range(iters):
        pr = (1 - d) / n + d * (M @ pr + pr[dangling].sum() / n)
    return pr * n


def louvain(P: sp.csr_matrix, seed: int) -> np.ndarray:
    G = nx.from_scipy_sparse_array(P, edge_attribute="weight")
    comms = nx.community.louvain_communities(G, weight="weight", seed=seed)
    label = np.empty(P.shape[0], dtype=np.int64)
    for c, members in enumerate(comms):
        label[list(members)] = c
    return label


def graph_features(s: Snapshot, cfg: dict) -> tuple[pd.DataFrame, sp.csr_matrix, np.ndarray]:
    Pn = s.persons
    n = len(Pn)
    idx = s.index
    f = pd.DataFrame(index=Pn.index)
    known = Pn["known_fraud"].to_numpy(dtype=float)

    pa = s.person_attr.drop_duplicates(["person_id", "token"]).copy()
    pa["pidx"] = pa["person_id"].map(idx).values
    pa["deg"] = pa.groupby("token")["pidx"].transform("size")
    pa["known_self"] = known[pa["pidx"].values]
    pa["known_on_attr"] = pa.groupby("token")["known_self"].transform("sum")
    for t in ATTR_TYPES:
        sub = pa[pa["attr_type"] == t]
        f[f"g_share_{t}"] = (sub["deg"] - 1).groupby(sub["pidx"]).sum().reindex(f.index, fill_value=0)
    strong = pa[pa["attr_type"].isin(["nid", "phone", "email", "address", "device"])]
    f["g_max_strong_share"] = (strong["deg"] - 1).groupby(strong["pidx"]).max().reindex(f.index, fill_value=0)
    others_known = pa["known_on_attr"] - pa["known_self"]
    f["g_attr_known_links"] = (others_known > 0).groupby(pa["pidx"]).sum().reindex(f.index, fill_value=0)

    nid = pa[pa["attr_type"] == "nid"].copy()
    nid["id_token"] = Pn["id_token"].values[nid["pidx"].values]
    n_ids = nid.groupby("token")["id_token"].transform("nunique")
    f["g_nid_identity_mismatch"] = (n_ids - 1).groupby(nid["pidx"]).max().reindex(f.index, fill_value=0)

    P = projection(s, cfg)
    A = P.copy()
    A.data[:] = 1.0
    f["g_proj_degree"] = np.asarray(A.sum(axis=1)).ravel()
    f["g_proj_wdegree"] = np.asarray(P.sum(axis=1)).ravel()
    kf1 = A @ known
    f["g_known_fraud_1hop"] = kf1
    f["g_known_fraud_2hop_paths"] = A @ kf1
    _, wcc = connected_components(A, directed=False)
    f["g_wcc_size"] = np.bincount(wcc)[wcc]
    comm = louvain(P, cfg["seed"])
    csize = np.bincount(comm)
    cknown = np.bincount(comm, weights=known)
    f["g_comm_size"] = csize[comm]
    f["g_comm_known_share"] = np.divide(cknown[comm] - known, csize[comm] - 1,
                                        out=np.zeros(n), where=csize[comm] > 1)
    f["g_pagerank"] = pagerank(P)

    tr = s.transfers
    if len(tr):
        out_cp = tr.groupby("src")["dst"].nunique()
        in_cp = tr.groupby("dst")["src"].nunique()
        f["g_transfer_out_cp"] = Pn["person_id"].map(out_cp).fillna(0).values
        f["g_transfer_in_cp"] = Pn["person_id"].map(in_cp).fillna(0).values
    else:
        f["g_transfer_out_cp"] = f["g_transfer_in_cp"] = 0.0
    au = s.authorized_users
    deps = au.groupby("host_person_id")["person_id"].nunique()
    f["g_au_hosts"] = Pn["person_id"].map(au.groupby("person_id")["host_person_id"].nunique()).fillna(0).values
    f["g_au_dependents"] = Pn["person_id"].map(deps).fillna(0).values
    host_deps = au.assign(d=au["host_person_id"].map(deps)).groupby("person_id")["d"].max()
    f["g_au_host_max_dependents"] = Pn["person_id"].map(host_deps).fillna(0).values
    sim_count = pd.concat([s.similar["a"], s.similar["b"]]).value_counts()
    f["g_similar_count"] = Pn["person_id"].map(sim_count).fillna(0).values

    pm = s.paid_at.drop_duplicates(["person_id", "merchant_token"])
    if len(pm):
        B, deg = _incidence(pm["person_id"].map(idx).values, pm["merchant_token"], n)
        small = sp.diags(((deg >= 2) & (deg <= cfg["graph"]["merchant_degree_cap"])).astype(float))
        C = (B @ small @ B.T).tocsr()
        C.data[:] = 1.0
        f["g_small_merchant_covisitors"] = np.asarray(C.sum(axis=1)).ravel() - np.asarray((B @ small).sum(axis=1)).ravel().clip(max=1)
    else:
        f["g_small_merchant_covisitors"] = 0.0
    return f, P, comm


def compute_features(s: Snapshot, cfg: dict) -> tuple[pd.DataFrame, sp.csr_matrix, np.ndarray]:
    app = application_features(s)
    beh = behavior_features(s)
    g, P, comm = graph_features(s, cfg)
    feats = pd.concat([app, beh, g], axis=1)
    feats.insert(0, "person_id", s.persons["person_id"].values)
    feats.insert(1, "T", s.T)
    feats["scored"] = s.persons["scored"].values
    feats["known_fraud"] = s.persons["known_fraud"].values
    feats["community"] = comm
    return feats, P, comm


def feature_columns(df: pd.DataFrame, prefixes: tuple[str, ...]) -> list[str]:
    return [c for c in df.columns if c.startswith(prefixes)]
