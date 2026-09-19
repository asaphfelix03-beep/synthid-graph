"""Chargement du graphe tokenisé dans Neo4j, write-back des scores, démo Graph Data Science.

Modèle : (Person)-[:OWNS]->(Account) ; (Person)-[:HAS_PHONE|HAS_EMAIL|LIVES_AT|HAS_NATIONAL_ID|USED_DEVICE|USED_IP|WORKS_AT]->(attribut) ;
(Account)-[:PAID_AT]->(Merchant) ; (Account)-[:TRANSFERRED_TO]->(Account) ; (Person)-[:AUTHORIZED_USER_ON]->(Account) ;
(Person)-[:SIMILAR_TO]->(Person). Toutes les relations sont datées (first_day / last_day) pour les vues « as-of T ».
"""
from __future__ import annotations

import os
import time
from importlib.resources import files
from pathlib import Path

import pandas as pd
from neo4j import GraphDatabase

ATTR = {"nid": ("NationalID", "HAS_NATIONAL_ID"), "phone": ("Phone", "HAS_PHONE"), "email": ("Email", "HAS_EMAIL"),
        "address": ("Address", "LIVES_AT"), "device": ("Device", "USED_DEVICE"), "ip": ("IP", "USED_IP"),
        "employer": ("Employer", "WORKS_AT")}


def driver_from_env():
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    password = os.environ.get("NEO4J_PASSWORD", "synthid-poc-local")
    return GraphDatabase.driver(uri, auth=(os.environ.get("NEO4J_USER", "neo4j"), password))


def _records(df: pd.DataFrame) -> list[dict]:
    return df.astype(object).where(df.notna(), None).to_dict("records")


def _batched(session, query: str, rows: list[dict], batch: int = 5000):
    for i in range(0, len(rows), batch):
        session.run(query, rows=rows[i:i + batch]).consume()


def wait_until_ready(driver, timeout: int = 180):
    t = time.time()
    while True:
        try:
            driver.verify_connectivity()
            return
        except Exception:
            if time.time() - t > timeout:
                raise
            time.sleep(3)


def load_graph(driver, bank: dict[str, pd.DataFrame]) -> dict:
    t = time.perf_counter()
    with driver.session() as s:
        s.run("MATCH (n) CALL (n) { DETACH DELETE n } IN TRANSACTIONS OF 10000 ROWS").consume()
        for stmt in files("synthid").joinpath("schema.cypher").read_text(encoding="utf-8").split(";"):
            lines = [ln for ln in stmt.splitlines() if ln.strip() and not ln.strip().startswith("//")]
            if lines:
                s.run("\n".join(lines)).consume()

        P = bank["persons"].copy()
        fc = bank["fraud_confirmations"].set_index("person_id")["confirmed_day"]
        P["known_fraud"] = P["person_id"].isin(fc.index)
        P["confirmed_day"] = P["person_id"].map(fc)
        cols = ["person_id", "bank", "onboard_day", "close_day", "id_token", "address_type", "app_ip_type",
                "known_fraud", "confirmed_day"]
        _batched(s, """UNWIND $rows AS r
            CREATE (p:Person) SET p = r
            CREATE (p)-[:OWNS]->(:Account {account_id: r.person_id + ':acct', bank: r.bank, open_day: r.onboard_day})""",
                 _records(P[cols]))

        meta = bank["attr_meta"]
        for t_, (label, rel) in ATTR.items():
            m = meta[meta["attr_type"] == t_].drop(columns=["attr_type"])
            _batched(s, f"UNWIND $rows AS r CREATE (n:{label}) SET n = r", _records(m))
            e = bank["person_attr"][bank["person_attr"]["attr_type"] == t_][["person_id", "token", "first_day", "last_day"]]
            _batched(s, f"""UNWIND $rows AS r
                MATCH (p:Person {{person_id: r.person_id}}), (n:{label} {{token: r.token}})
                CREATE (p)-[:{rel} {{first_day: r.first_day, last_day: r.last_day}}]->(n)""", _records(e))

        _batched(s, "UNWIND $rows AS r CREATE (m:Merchant) SET m = r",
                 _records(bank["merchant_meta"].rename(columns={"merchant_token": "token"})))
        _batched(s, """UNWIND $rows AS r
            MATCH (a:Account {account_id: r.person_id + ':acct'}), (m:Merchant {token: r.merchant_token})
            CREATE (a)-[:PAID_AT {first_day: r.first_day, last_day: r.last_day}]->(m)""", _records(bank["paid_at"]))

        tr = (bank["transfers"].groupby(["src", "dst"])
              .agg(n=("n", "sum"), amount=("amount", "sum"), first_month=("month", "min"), last_month=("month", "max"))
              .reset_index())
        tr["amount"] = tr["amount"].round(2)
        _batched(s, """UNWIND $rows AS r
            MATCH (a:Account {account_id: r.src + ':acct'}), (b:Account {account_id: r.dst + ':acct'})
            CREATE (a)-[:TRANSFERRED_TO {n: r.n, amount: r.amount, first_day: r.first_month * 30,
                                         last_day: r.last_month * 30 + 29}]->(b)""", _records(tr))
        _batched(s, """UNWIND $rows AS r
            MATCH (p:Person {person_id: r.person_id}), (a:Account {account_id: r.host_person_id + ':acct'})
            CREATE (p)-[:AUTHORIZED_USER_ON {first_day: r.first_day}]->(a)""", _records(bank["authorized_users"]))
        _batched(s, """UNWIND $rows AS r
            MATCH (a:Person {person_id: r.a}), (b:Person {person_id: r.b})
            CREATE (a)-[:SIMILAR_TO {score: r.score, first_day: r.first_day}]->(b)""", _records(bank["similar"]))
        counts = s.run("MATCH (n) RETURN count(n) AS nodes").single()["nodes"], \
            s.run("MATCH ()-[r]->() RETURN count(r) AS rels").single()["rels"]
    return {"nodes": counts[0], "relationships": counts[1], "seconds": round(time.perf_counter() - t, 1)}


def write_back(driver, scores: pd.DataFrame, score_col: str, T: int, alerted: pd.DataFrame | None = None):
    rows = scores[scores["T"] == T][["person_id", score_col]].rename(columns={score_col: "score"})
    with driver.session() as s:
        _batched(s, "UNWIND $rows AS r MATCH (p:Person {person_id: r.person_id}) SET p.risk_score = r.score, p.scored_at = $T"
                 .replace("$T", str(T)), _records(rows))
        if alerted is not None and len(alerted):
            _batched(s, "UNWIND $rows AS r MATCH (p:Person {person_id: r.person_id}) SET p.community_alert = r.community",
                     _records(alerted[["person_id", "community"]]))


def investigation_report(driver) -> dict:
    queries = {
        "paires_partageant_2_attributs_forts": """
            MATCH (p1:Person)-[:HAS_NATIONAL_ID|HAS_PHONE|HAS_EMAIL|LIVES_AT|USED_DEVICE]->(a)
                  <-[:HAS_NATIONAL_ID|HAS_PHONE|HAS_EMAIL|LIVES_AT|USED_DEVICE]-(p2:Person)
            WHERE p1.person_id < p2.person_id AND COUNT { (a)<--(:Person) } <= 20
            WITH p1, p2, collect(DISTINCT labels(a)[0]) AS shared WHERE size(shared) >= 2
            RETURN count(*) AS n""",
        "nid_portes_par_plusieurs_identites": """
            MATCH (n:NationalID)<-[:HAS_NATIONAL_ID]-(p:Person)
            WITH n, count(DISTINCT p.id_token) AS ids WHERE ids >= 2 RETURN count(n) AS n""",
        "hotes_tradeline_3_dependants_plus": """
            MATCH (p:Person)-[:AUTHORIZED_USER_ON]->(a:Account)<-[:OWNS]-(h:Person)
            WITH h, count(DISTINCT p) AS d WHERE d >= 3 RETURN count(h) AS n""",
        "clients_ouverts_a_2_sauts_d_une_fraude_confirmee": """
            MATCH (f:Person {known_fraud: true})-[:HAS_PHONE|HAS_EMAIL|LIVES_AT|USED_DEVICE|HAS_NATIONAL_ID]->(a)<--(p:Person)
            WHERE p.known_fraud = false AND COUNT { (a)<--(:Person) } <= 20
            RETURN count(DISTINCT p) AS n""",
    }
    out = {}
    with driver.session() as s:
        for name, q in queries.items():
            t = time.perf_counter()
            out[name] = {"result": s.run(q).single()["n"], "ms": round(1000 * (time.perf_counter() - t), 1)}
    return out


def gds_demo(driver) -> dict:
    with driver.session() as s:
        s.run("CALL gds.graph.drop('identity', false) YIELD graphName RETURN graphName").consume()
        proj = s.run("""CALL gds.graph.project('identity',
                ['Person', 'Phone', 'Email', 'Address', 'Device', 'NationalID'],
                {HAS_PHONE: {orientation: 'UNDIRECTED'}, HAS_EMAIL: {orientation: 'UNDIRECTED'},
                 LIVES_AT: {orientation: 'UNDIRECTED'}, USED_DEVICE: {orientation: 'UNDIRECTED'},
                 HAS_NATIONAL_ID: {orientation: 'UNDIRECTED'}})
            YIELD nodeCount, relationshipCount RETURN nodeCount, relationshipCount""").single()
        wcc = s.run("""CALL gds.wcc.stats('identity') YIELD componentCount, componentDistribution
                       RETURN componentCount, componentDistribution.max AS largest""").single()
        louv = s.run("""CALL gds.louvain.stats('identity') YIELD communityCount, modularity
                        RETURN communityCount, modularity""").single()
        s.run("CALL gds.graph.drop('identity') YIELD graphName RETURN graphName").consume()
    return {"projected_nodes": proj["nodeCount"], "projected_relationships": proj["relationshipCount"],
            "wcc_components": wcc["componentCount"], "wcc_largest": wcc["largest"],
            "louvain_communities": louv["communityCount"], "louvain_modularity": round(louv["modularity"], 4)}


def run_neo4j_stage(cfg: dict, bank: dict[str, pd.DataFrame], scores: pd.DataFrame | None, score_col: str | None,
                    alerted: pd.DataFrame | None) -> dict:
    driver = driver_from_env()
    try:
        wait_until_ready(driver)
        res = {"load": load_graph(driver, bank)}
        if scores is not None and score_col:
            write_back(driver, scores, score_col, cfg["evaluation"]["test_snapshots"][0], alerted)
        res["investigation_queries"] = investigation_report(driver)
        res["gds"] = gds_demo(driver)
        return res
    finally:
        driver.close()


if __name__ == "__main__":  # pragma: no cover
    print(Path(__file__).name)
