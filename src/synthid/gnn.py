"""Graph Neural Networks (PyTorch Geometric).

- `GCN` : Kipf & Welling sur la projection homogène personne–personne pondérée. Baseline GNN.
- `HeteroSAGE` : GraphSAGE converti en hétérogène par `to_hetero` (GCNConv ne gère pas le message passing
  biparti et ne peut donc pas être converti). Inductif : entraîné sur les snapshots T1, appliqué aux T2.
  Un « skip » garde les features propres du nœud (utile en hétérophilie : les fraudeurs se camouflent).

Le graphe est l'union disjointe des graphes des 3 banques (jetons propres à chaque banque) :
aucun message ne traverse une frontière bancaire ; le signal inter-bancaire passe par la couche chiffrée.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import torch_geometric.transforms as Tr
from sklearn.metrics import average_precision_score
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import GCNConv, Linear, SAGEConv, to_hetero

from .features import ATTR_TYPES, IP_TYPES, Snapshot

PERSON_PP_EDGES = ["transfer", "authorized_user", "similar"]


class FeatureScaler:
    """log signé puis standardisation, statistiques apprises sur les lignes d'entraînement."""

    def fit(self, X: pd.DataFrame) -> FeatureScaler:
        Z = self._log(X)
        self.cols = list(X.columns)
        self.median = Z.median()
        Z = Z.fillna(self.median)
        self.mean, self.std = Z.mean(), Z.std().replace(0, 1).fillna(1)
        return self

    @staticmethod
    def _log(X: pd.DataFrame) -> pd.DataFrame:
        X = X.astype(float)
        return np.sign(X) * np.log1p(np.abs(X))

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        Z = self._log(X[self.cols])
        missing = Z.isna().any(axis=1).astype(float).to_numpy()[:, None]
        Z = ((Z.fillna(self.median) - self.mean) / self.std).clip(-6, 6).to_numpy(dtype=np.float32)
        return np.hstack([Z, missing.astype(np.float32)])


def _attr_features(t: str, meta: pd.DataFrame) -> np.ndarray:
    cols = []
    if t == "phone":
        cols = [meta["subtype"].eq("voip")]
    elif t == "email":
        cols = [meta["disposable"].fillna(False).astype(bool), meta["email_popular_domain"].fillna(False).astype(bool)]
    elif t == "address":
        cols = [meta["subtype"].eq(x) for x in ["residential", "mail_drop", "commercial"]]
    elif t == "device":
        cols = [meta["emulator"].fillna(False).astype(bool), meta["os"].eq("ios")]
    elif t == "ip":
        cols = [meta["subtype"].eq(x) for x in IP_TYPES]
    if not cols:
        return np.zeros((len(meta), 0), dtype=np.float32)
    return np.column_stack([c.to_numpy(dtype=float) for c in cols]).astype(np.float32)


def build_hetero(s: Snapshot, x_person: np.ndarray, merchant_cap: int = 200) -> HeteroData:
    n = len(s.persons)
    idx = s.index
    known = s.persons["known_fraud"].to_numpy(dtype=float)
    data = HeteroData()
    data["person"].x = torch.tensor(np.hstack([x_person, known[:, None]]), dtype=torch.float32)
    meta_all = s.attr_meta.set_index("token")

    def add_bipartite(name: str, rel: str, pidx: np.ndarray, keys: pd.Series, extra):
        codes, uniques = pd.factorize(keys)
        deg = np.bincount(codes, minlength=len(uniques)).astype(float)
        kn = np.bincount(codes, weights=known[pidx], minlength=len(uniques))
        base = np.column_stack([np.log1p(deg), np.log1p(kn)]).astype(np.float32)
        x = np.hstack([base, extra(uniques)]) if extra else base
        data[name].x = torch.tensor(x)
        data["person", rel, name].edge_index = torch.tensor(np.vstack([pidx, codes]), dtype=torch.long)

    pa = s.person_attr.drop_duplicates(["person_id", "token"])
    for t in ATTR_TYPES:
        sub = pa[pa["attr_type"] == t]
        add_bipartite(t, f"has_{t}", sub["person_id"].map(idx).to_numpy(), sub["token"],
                      lambda u, t=t: _attr_features(t, meta_all.reindex(u)))
    pm = s.paid_at.drop_duplicates(["person_id", "merchant_token"])
    mdeg = pm.groupby("merchant_token")["person_id"].transform("size")
    pm = pm[mdeg <= merchant_cap]  # les super-marchands (grandes enseignes) n'apportent pas d'information
    mmeta = s.merchant_meta.set_index("merchant_token")
    add_bipartite("merchant", "paid_at", pm["person_id"].map(idx).to_numpy(), pm["merchant_token"],
                  lambda u: mmeta.reindex(u)[["high_risk_mcc"]].fillna(False).to_numpy(dtype=np.float32))

    pairs = {"transfer": s.transfers[["src", "dst"]].drop_duplicates().to_numpy(),
             "authorized_user": s.authorized_users[["person_id", "host_person_id"]].to_numpy(),
             "similar": s.similar[["a", "b"]].to_numpy()}
    for rel, arr in pairs.items():
        if len(arr):
            ei = np.vstack([idx.loc[arr[:, 0]].to_numpy(), idx.loc[arr[:, 1]].to_numpy()])
        else:
            ei = np.zeros((2, 0), dtype=np.int64)
        data["person", rel, "person"].edge_index = torch.tensor(ei, dtype=torch.long)
    data = Tr.ToUndirected()(data)
    data["person"].num_nodes = n
    return data


def build_homogeneous(P: sp.csr_matrix, x_person: np.ndarray, known: np.ndarray) -> Data:
    coo = P.tocoo()
    return Data(x=torch.tensor(np.hstack([x_person, known[:, None]]), dtype=torch.float32),
                edge_index=torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long),
                edge_weight=torch.tensor(coo.data, dtype=torch.float32))


class GCN(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.head = torch.nn.Linear(hidden, 1)
        self.drop = torch.nn.Dropout(dropout)

    def forward(self, data: Data) -> torch.Tensor:
        h = self.drop(self.conv1(data.x, data.edge_index, data.edge_weight).relu())
        h = self.conv2(h, data.edge_index, data.edge_weight).relu()
        return self.head(h).squeeze(-1)


class _SAGE(torch.nn.Module):
    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden)
        self.conv2 = SAGEConv((-1, -1), hidden)
        self.norm1 = torch.nn.LayerNorm(hidden)
        self.norm2 = torch.nn.LayerNorm(hidden)
        self.skip = Linear(-1, hidden)
        self.head = Linear(-1, 1)
        self.drop = torch.nn.Dropout(dropout)

    def forward(self, x, edge_index):
        h = self.drop(self.norm1(self.conv1(x, edge_index)).relu())
        h = self.norm2(self.conv2(h, edge_index)).relu()
        z = torch.cat([h, self.skip(x).relu()], dim=-1)
        return self.head(self.drop(z))


class HeteroSAGE(torch.nn.Module):
    """Agrégation « mean » entre types de relations : sans elle, la somme sur 11 relations déstabilise l'apprentissage."""

    def __init__(self, metadata, hidden: int, dropout: float):
        super().__init__()
        self.net = to_hetero(_SAGE(hidden, dropout), metadata, aggr="mean")

    def forward(self, data: HeteroData) -> torch.Tensor:
        return self.net(data.x_dict, data.edge_index_dict)["person"].squeeze(-1)


def train_gnn(kind: str, train_graphs: list[dict], val_graphs: list[dict], cfg: dict, seed: int):
    """Chaque élément : {'data', 'rows' (index des personnes), 'y'}. Arrêt anticipé sur la PR-AUC de validation."""
    gc = cfg["gnn"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    first = train_graphs[0]["data"]
    if kind == "gcn":
        model = GCN(first.x.shape[1], gc["hidden"], gc["dropout"])
    else:
        model = HeteroSAGE(first.metadata(), gc["hidden"], gc["dropout"])
        with torch.no_grad():
            model(first)  # initialise les couches paresseuses
    opt = torch.optim.Adam(model.parameters(), lr=gc["lr"], weight_decay=gc["weight_decay"])
    ys = np.concatenate([g["y"] for g in train_graphs])
    pos_weight = torch.tensor(min(20.0, (len(ys) - ys.sum()) / max(1, ys.sum())))
    best, best_state, wait, history = -1.0, None, 0, []
    for epoch in range(gc["max_epochs"]):
        model.train()
        for g in train_graphs:
            opt.zero_grad()
            out = model(g["data"])[g["rows"]]
            loss = F.binary_cross_entropy_with_logits(out, torch.tensor(g["y"], dtype=torch.float32), pos_weight=pos_weight)
            loss.backward()
            opt.step()
        if epoch % 2:  # validation une époque sur deux (le coût d'une passe de validation ≈ une passe d'entraînement)
            continue
        score = average_precision_score(np.concatenate([g["y"] for g in val_graphs]),
                                        np.concatenate([predict(model, g) for g in val_graphs]))
        history.append(score)
        if epoch % 10 == 0:
            print(f"    {kind} époque {epoch} : PR-AUC val {score:.3f} (meilleure {max(best, score):.3f})", flush=True)
        if score > best + 1e-4:
            best, best_state, wait = score, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= gc["patience"]:
                break
    model.load_state_dict(best_state)
    return model, {"best_val_pr_auc": best, "epochs": epoch + 1}


@torch.no_grad()
def predict(model, g: dict) -> np.ndarray:
    model.eval()
    return model(g["data"])[g["rows"]].numpy()
