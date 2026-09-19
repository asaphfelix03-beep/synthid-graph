"""Simulateur multi-agents temporel.

Agents : foyers légitimes (et leurs confondants : colocations, résidences étudiantes, CGNAT, Wi-Fi public,
personnes âgées aidées, défauts de crédit), opérateurs de fraude qui lancent des anneaux d'identités
synthétiques par vagues successives (création → culture → bust-out → exfiltration vers des mules).

Le temps est exprimé en jours depuis J0 (négatif = avant le début de la simulation) ; un mois = 30 jours.
Les sorties constituent le « coffre » du simulateur : PII en clair + vérité terrain. Les banques ne voient
ensuite que des tables tokenisées (voir `bank_view.py`).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from faker import Faker

from .normalize import norm_address, norm_name

MONTH = 30
DAY0 = pd.Timestamp("2023-01-01")
FAR = 10**6  # « jamais » : compte toujours ouvert, pas de bust-out, etc.

REGULAR_MCC = ["grocery", "restaurant", "fuel", "pharmacy", "clothing", "transport", "utilities", "telecom",
               "entertainment", "travel", "home", "health", "online_retail", "sports", "books", "beauty"]
HIGH_RISK_MCC = ["electronics", "jewelry", "gift_cards", "money_transfer"]
POPULAR_DOMAINS = ["gmail.com", "orange.fr", "hotmail.fr", "free.fr", "yahoo.fr", "outlook.fr", "laposte.net", "sfr.fr"]
POPULAR_DOMAIN_W = [0.38, 0.16, 0.12, 0.10, 0.08, 0.07, 0.05, 0.04]
DISPOSABLE_DOMAINS = ["yopmail.com", "mailinator.com", "trashmail.fr", "tempmail.dev", "guerrillamail.com"]
IP_PREFIX = {"residential": 90, "cgnat": 37, "public_wifi": 212, "office": 193, "datacenter": 51, "vpn": 185}
COMPANY_SUFFIX = ["SAS", "SARL", "SA", "EURL"]
COMPANY_WORDS = ["Conseil", "Services", "Logistique", "Batiment", "Distribution", "Industrie", "Transports",
                 "Solutions", "Digital", "Nettoyage", "Securite", "Sante", "Energie", "Commerce", "Invest"]


class Simulator:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.pop = cfg["population"]
        self.fr = cfg["fraud"]
        self.H = int(cfg["horizon_days"])
        self.M = self.H // MONTH
        self.rng = np.random.default_rng(cfg["seed"])
        self.banks = list(cfg["banks"])
        w = np.array(list(cfg["banks"].values()), dtype=float)
        self.bank_w = w / w.sum()

        fake = Faker("fr_FR")
        fake.seed_instance(cfg["seed"])
        self.first_m = sorted({fake.first_name_male() for _ in range(3000)})
        self.first_f = sorted({fake.first_name_female() for _ in range(3000)})
        self.last_names = sorted({fake.last_name() for _ in range(6000)})
        self.streets = sorted({fake.street_name() for _ in range(4000)})
        self.cities = sorted({(fake.postcode(), fake.city()) for _ in range(800)})
        self.company_names = sorted({fake.last_name() for _ in range(3000)})

        self.used: dict[str, set] = {k: set() for k in ["nid", "phone", "email", "address", "device", "ip", "employer"]}
        self.catalog: dict[str, dict[str, dict]] = {k: {} for k in self.used}
        self.identities: list[dict] = []
        self.records: list[dict] = []
        self.record_attr: list[tuple] = []  # person_id, attr_type, raw, canonical, first_day, last_day
        self.paid_at: list[tuple] = []  # person_id, merchant_id, first_day, last_day
        self.transfers: list[tuple] = []  # src, dst, month, n, amount
        self.au: list[tuple] = []  # person_id, host_person_id, first_day
        self.merchants: list[dict] = []
        self._rec_counter = {b: 0 for b in self.banks}

    # ------------------------------------------------------------------ utilitaires
    def _choice(self, seq, size=None, replace=True, p=None):
        idx = self.rng.choice(len(seq), size=size, replace=replace, p=p)
        if size is None:
            return seq[int(idx)]
        return [seq[int(i)] for i in np.atleast_1d(idx)]

    def _unique(self, kind: str, make) -> str:
        for _ in range(10_000):
            v = make()
            if v not in self.used[kind]:
                self.used[kind].add(v)
                return v
        raise RuntimeError(f"espace de valeurs épuisé pour {kind}")

    def _u(self, lo, hi) -> int:
        return int(self.rng.integers(int(lo), int(hi) + 1))

    # ------------------------------------------------------------------ attributs
    def new_nid(self, sex: int, birth_year: int) -> str:
        r = self.rng
        return self._unique("nid", lambda: f"{sex}{birth_year % 100:02d}{r.integers(1, 13):02d}"
                                           f"{r.integers(1, 96):02d}{r.integers(1, 1000):03d}{r.integers(1, 1000):03d}")

    def new_phone(self, subtype: str, first_seen: int) -> str:
        prefix = "9" if subtype == "voip" else self._choice(["6", "7"])
        v = self._unique("phone", lambda: f"+33{prefix}{self.rng.integers(0, 10**8):08d}")
        self.catalog["phone"][v] = {"subtype": subtype, "first_seen": first_seen}
        return v

    def touch_phone(self, v: str, day: int):
        meta = self.catalog["phone"][v]
        meta["first_seen"] = min(meta["first_seen"], day)

    def new_email(self, first: str, last: str, created_day: int, disposable: bool) -> str:
        domain = self._choice(DISPOSABLE_DOMAINS) if disposable else self._choice(POPULAR_DOMAINS, p=POPULAR_DOMAIN_W)
        base = f"{norm_name(first).replace(' ', '')}.{norm_name(last).replace(' ', '')}"
        v = self._unique("email", lambda: f"{base}{self.rng.integers(0, 1000) if self.rng.random() < 0.7 else ''}@{domain}")
        self.catalog["email"][v] = {"domain": domain, "disposable": disposable, "created_day": created_day}
        return v

    def new_address(self, subtype: str) -> str:
        def make():
            pc, city = self._choice(self.cities)
            street = self._choice(self.streets)
            if subtype in ("mail_drop", "commercial"):
                return norm_address(f"{self.rng.integers(1, 200)} {street} bp {self.rng.integers(1, 9999)} {pc} {city}")
            return norm_address(f"{self.rng.integers(1, 200)} {street} {pc} {city}")
        v = self._unique("address", make)
        self.catalog["address"][v] = {"subtype": subtype}
        return v

    def new_device(self, emulator: bool = False) -> str:
        v = self._unique("device", lambda: f"fp_{self.rng.integers(0, 2**62):016x}")
        self.catalog["device"][v] = {"os": self._choice(["ios", "android"]), "emulator": emulator}
        return v

    def new_ip(self, subtype: str) -> str:
        a = IP_PREFIX[subtype]
        v = self._unique("ip", lambda: f"{a}.{self.rng.integers(0, 256)}.{self.rng.integers(0, 256)}.{self.rng.integers(1, 255)}")
        self.catalog["ip"][v] = {"subtype": subtype}
        return v

    def new_employer(self, fictitious: bool) -> str:
        v = self._unique("employer", lambda: norm_address(
            f"{self._choice(self.company_names)} {self._choice(COMPANY_WORDS)} {self._choice(COMPANY_SUFFIX)}"))
        self.catalog["employer"][v] = {"fictitious": fictitious}
        return v

    def new_merchant(self, high_risk: bool, complicit: bool) -> int:
        mid = len(self.merchants)
        mcc = self._choice(HIGH_RISK_MCC) if high_risk else self._choice(REGULAR_MCC)
        self.merchants.append({"merchant_id": mid, "mcc": mcc, "high_risk_mcc": mcc in HIGH_RISK_MCC, "complicit": complicit})
        return mid

    # -------------------------------------------------------------- formats bruts
    def fmt_phone(self, v: str) -> str:
        local = "0" + v[3:]
        k = self.rng.integers(3)
        if k == 0:
            return v
        if k == 1:
            return local
        return " ".join(local[i:i + 2] for i in range(0, 10, 2))

    def fmt_email(self, v: str) -> str:
        return v.title() if self.rng.random() < 0.2 else v

    def fmt_address(self, v: str) -> str:
        k = self.rng.integers(3)
        if k == 0:
            return v
        if k == 1:
            return v.upper()
        parts = v.split(" ", 1)
        return f"{parts[0]}, {parts[1].title()}" if len(parts) == 2 else v

    # ------------------------------------------------------------------ identités
    def _new_identity(self, kind: str, age: float, last_name=None, first_name=None, sex=None,
                      birth_day=None, household_id=-1, **flags) -> dict:
        sex = sex or int(self.rng.integers(1, 3))
        first = first_name or self._choice(self.first_m if sex == 1 else self.first_f)
        last = last_name or self._choice(self.last_names)
        if birth_day is None:
            birth_day = int(-age * 365.25 - self.rng.integers(0, 365))
        dob = (DAY0 + pd.Timedelta(days=birth_day)).date()
        ident = dict(identity_id=len(self.identities), kind=kind, sex=sex, first_name=first, last_name=last,
                     birth_day=birth_day, dob=dob.isoformat(), birth_year=dob.year, household_id=household_id,
                     ring_id=-1, operator_id=-1, sophistication=0, scenario="", is_dorm=False, is_colocation=False,
                     is_elderly_assisted=False, is_newcomer=False, is_tradeline_seller=False, is_mule=False,
                     stolen_nid=False, variant=False, employer=None, devices=[], ips=[])
        ident.update(flags)
        self.identities.append(ident)
        return ident

    def _age_at(self, ident: dict, day: int) -> float:
        return (day - ident["birth_day"]) / 365.25

    def _new_record(self, ident: dict, bank: str, onboard: int, **kw) -> dict:
        self._rec_counter[bank] += 1
        rec = dict(person_id=f"{bank}-{self._rec_counter[bank]:06d}", bank=bank, identity_id=ident["identity_id"],
                   onboard_day=int(onboard), close_day=FAR, bust_out_day=FAR, bust_out_end=FAR, label_day=FAR,
                   label_observed=False, is_distressed=False, default_day=FAR, app_device=None, app_ip=None)
        rec.update(kw)
        self.records.append(rec)
        return rec

    def _static_edges(self, rec: dict, ident: dict):
        end = min(rec["close_day"], self.H)
        start = rec["onboard_day"]
        pid = rec["person_id"]
        self.record_attr.append((pid, "nid", ident["nid"], ident["nid"], start, end))
        self.record_attr.append((pid, "phone", self.fmt_phone(ident["phone"]), ident["phone"], start, end))
        self.record_attr.append((pid, "email", self.fmt_email(ident["email"]), ident["email"], start, end))
        self.record_attr.append((pid, "address", self.fmt_address(ident["address"]), ident["address"], start, end))
        if ident["employer"]:
            self.record_attr.append((pid, "employer", ident["employer"], ident["employer"], start, end))

    def _session_edges(self, rec: dict, devices: list, ips: list, extra_delay: int = 400):
        """Le 1er appareil / la 1re IP servent à la demande d'ouverture ; les autres apparaissent ensuite."""
        end = min(rec["close_day"], self.H)
        start = rec["onboard_day"]
        pid = rec["person_id"]
        for kind, vals in (("device", devices), ("ip", ips)):
            for i, v in enumerate(vals):
                first = start if i == 0 else min(end, start + self._u(0, extra_delay))
                self.record_attr.append((pid, kind, v, v, first, end))
        rec["app_device"], rec["app_ip"] = devices[0], ips[0]

    # ------------------------------------------------------------------ légitimes
    def generate_legit(self):
        p, r = self.pop, self.rng
        employers = [self.new_employer(False) for _ in range(p["n_employers"])]
        ranks = np.arange(1, len(employers) + 1, dtype=float)
        w = ranks ** -p["employer_zipf_a"]
        self.employers, self.employer_w = employers, w / w.sum()
        self.office_ip = {e: self.new_ip("office") for e in employers[: len(employers) // 5]}
        self.cgnat_pool = [self.new_ip("cgnat") for _ in range(p["cgnat_ips"])]
        self.wifi_pool = [self.new_ip("public_wifi") for _ in range(p["public_wifi_ips"])]
        self.home_ips: list[str] = []

        hh = 0
        for _ in range(p["dorm_buildings"]):
            addr, wifi = self.new_address("residential"), self.new_ip("residential")
            for _ in range(self._u(*p["dorm_size"])):
                ident = self._new_identity("legit", r.uniform(18, 25), household_id=hh, is_dorm=True)
                ident.update(address=addr, home_ip=wifi)
                hh += 1

        sizes = np.arange(1, len(p["household_size_weights"]) + 1)
        hw = np.array(p["household_size_weights"], float)
        while len(self.identities) < p["identities"]:
            size = int(min(r.choice(sizes, p=hw / hw.sum()), p["identities"] - len(self.identities)))
            head_age = r.uniform(19, 88)
            coloc = size >= 2 and head_age < 32 and r.random() < 0.5
            last = self._choice(self.last_names)
            addr, home_ip = self.new_address("residential"), self.new_ip("residential")
            self.home_ips.append(home_ip)
            members = []
            for i in range(size):
                if i == 0:
                    age = head_age
                elif coloc:
                    age = float(np.clip(head_age + r.normal(0, 3), 18, 35))
                elif i == 1:
                    age = float(np.clip(head_age + r.normal(0, 4), 18, 95))
                else:
                    age = r.uniform(18, max(19.0, head_age - 20)) if head_age > 42 else r.uniform(18, 30)
                ident = self._new_identity("legit", age, last_name=None if coloc else last, household_id=hh,
                                           is_colocation=coloc)
                ident.update(address=addr, home_ip=home_ip)
                members.append(ident)
            if size >= 2 and r.random() < p["shared_tablet_share"]:
                tablet = self.new_device()
                for m in self._choice(members, size=2, replace=False):
                    m["devices"].append(tablet)
            hh += 1
        self.n_households = hh

        for ident in self.identities:
            self._legit_attributes(ident)
        self._elderly_assisted()
        for ident in self.identities:
            self._legit_records(ident)
        self._pick_roles()
        self.legit_ages = [i["age0"] for i in self.identities if 20 <= i["age0"] <= 65]

    def _legit_attributes(self, ident: dict):
        p, r = self.pop, self.rng
        age = ident["age0"] if "age0" in ident else self._age_at(ident, 0)
        ident["age0"] = age
        ident["nid"] = self.new_nid(ident["sex"], ident["birth_year"])
        ident["phone"] = self.new_phone("voip" if r.random() < p["voip_share"] else "mobile", first_seen=0)
        works = self._works(age)
        if works:
            ident["employer"] = self._choice(self.employers, p=self.employer_w)
        ident["income"] = self._legit_income(age, works)
        ident["devices"] = [self.new_device(emulator=r.random() < 0.002)] + ident["devices"]
        ips = [ident["home_ip"]] + self._choice(self.cgnat_pool, size=int(r.choice([0, 1, 2], p=[0.25, 0.45, 0.3])), replace=False)
        if r.random() < p["public_wifi_share"]:
            ips.append(self._choice(self.wifi_pool))
        if ident["employer"] in self.office_ip and r.random() < p["office_ip_share"]:
            ips.append(self.office_ip[ident["employer"]])
        ident["ips"] = ips

    def _works(self, age: float) -> bool:
        r = self.rng
        return bool((22 <= age <= 64 and r.random() < 0.82) or (age < 22 and r.random() < 0.3))

    def _legit_income(self, age: float, works: bool) -> float:
        base = self.rng.lognormal(math.log(28000), 0.45)
        return float(base * (0.3 if age < 23 else 0.8 if age > 65 else 1.0 if works else 0.5))

    def _elderly_assisted(self):
        r = self.rng
        helpers = [i for i in self.identities if 30 <= i["age0"] <= 60]
        for ident in self.identities:
            if ident["age0"] >= 75 and r.random() < self.pop["elderly_assisted_share"]:
                helper = self._choice(helpers)
                if helper["household_id"] != ident["household_id"]:
                    ident["devices"].append(helper["devices"][0])
                    ident["is_elderly_assisted"] = True

    def _legit_records(self, ident: dict):
        p, r = self.pop, self.rng
        banks = [self._choice(self.banks, p=self.bank_w)]
        if r.random() < p["multi_bank_share"]:
            others = [b for b in self.banks if b != banks[0]]
            ow = np.array([self.cfg["banks"][b] for b in others], float)
            banks.append(self._choice(others, p=ow / ow.sum()))
        adult_day = ident["birth_day"] + int(18 * 365.25)
        lo = max(p["onboarding_window_start"], adult_day)
        onboards = [self._u(lo, max(lo, self.H - 120)) for _ in banks]
        first = min(onboards)

        newcomer = r.random() < p["newcomer_share"]
        ident["is_newcomer"] = newcomer
        if newcomer:
            ident["file_start_day"] = first - self._u(0, 400)
        else:
            ident["file_start_day"] = min(first - 30, adult_day + self._u(0, 8 * 365))
        new_email = newcomer or r.random() < p["new_email_share"]
        email_created = first - (self._u(0, 90) if new_email else self._u(365, 6000))
        email_created = max(email_created, ident["birth_day"] + 12 * 365)
        ident["email"] = self.new_email(ident["first_name"], ident["last_name"], email_created,
                                        disposable=r.random() < p["disposable_email_share"])
        new_phone = newcomer or r.random() < p["new_phone_share"]
        self.catalog["phone"][ident["phone"]]["first_seen"] = first - (self._u(0, 90) if new_phone else self._u(200, 5000))

        for bank, onboard in zip(banks, onboards):
            rec = self._new_record(ident, bank, onboard)
            if r.random() < p["distressed_share"]:
                start = max(onboard, 0) + self._u(90, 400)
                if start < self.H - 60:
                    rec.update(is_distressed=True, default_day=start, close_day=start + self._u(90, 150))
            self._legit_behavior(rec, ident)
            self._static_edges(rec, ident)
            devices = ident["devices"]
            ips = list(ident["ips"])
            if len(ips) > 1 and r.random() < 0.4:  # ouverture depuis le réseau mobile plutôt que la box
                ips[0], ips[1] = ips[1], ips[0]
            self._session_edges(rec, devices, ips)

    def _legit_behavior(self, rec: dict, ident: dict):
        p, r = self.pop, self.rng
        file_age = rec["onboard_day"] - ident["file_start_day"]
        transactor = r.random() < p["transactor_share"]
        rec.update(
            limit0=float(np.clip(round(ident["income"] * r.uniform(0.04, 0.12), -2), 300, 12000)),
            lir_p=r.uniform(0.02, 0.07) if file_age < 730 else r.uniform(0.005, 0.03),
            grant_p=0.5, lir_mult=1.25,
            util_mu=float(r.beta(2, 5)), util_sd=r.uniform(0.02, 0.12),
            pay_mu=1.0 if transactor else r.uniform(0.05, 0.6), pay_sd=0.01 if transactor else r.uniform(0.05, 0.2),
            late_p=float(r.beta(1, 80) if transactor else r.beta(1.5, 25)),
            txn_lam=r.uniform(4, 50), n_mcc_base=r.uniform(3, 12),
            cash_p=r.uniform(0.05, 0.2) if r.random() < 0.1 else 0.0,
        )

    def _pick_roles(self):
        """Vendeurs de « tradelines » (utilisateurs autorisés payants) et mules, choisis parmi les légitimes."""
        p, r = self.pop, self.rng
        ids = self.identities
        self.sellers, self.mules = {}, {}
        for b in self.banks:
            recs = [x for x in self.records if x["bank"] == b and not x["is_distressed"]]
            sellers = [x for x in recs if 40 <= ids[x["identity_id"]]["age0"] <= 70 and x["onboard_day"] <= 0]
            mules = [x for x in recs if ids[x["identity_id"]]["age0"] <= 30]
            self.sellers[b] = self._choice(sellers, size=p["tradeline_sellers_per_bank"], replace=False)
            self.mules[b] = self._choice(mules, size=p["mules_per_bank"], replace=False)
            for x in self.sellers[b]:
                ids[x["identity_id"]]["is_tradeline_seller"] = True
                x["limit0"] = float(r.uniform(15000, 25000))
            for x in self.mules[b]:
                ids[x["identity_id"]]["is_mule"] = True

    # --------------------------------------------------------------------- fraude
    def generate_fraud(self):
        fr, r = self.fr, self.rng
        n_rings = fr["rings"]
        n_ops = math.ceil(n_rings / fr["rings_per_operator"])
        self.operators = []
        for o in range(n_ops):
            self.operators.append(dict(
                operator_id=o,
                devices=[self.new_device(emulator=r.random() < 0.4) for _ in range(self._u(4, 10))],
                dc_ips=[self.new_ip(self._choice(["datacenter", "vpn"])) for _ in range(self._u(4, 8))],
                voip=[self.new_phone("voip", first_seen=FAR) for _ in range(self._u(4, 12))],
                mail_drops=[self.new_address(self._choice(["mail_drop", "commercial"])) for _ in range(self._u(2, 6))],
                fake_employers=[self.new_employer(True) for _ in range(self._u(1, 3))],
                complicit=[self.new_merchant(high_risk=True, complicit=True) for _ in range(self._u(1, 2))],
                mules={b: self._choice(self.mules[b], size=2, replace=False) for b in self.banks},
                sellers={b: self._choice(self.sellers[b], size=self._u(1, 2), replace=False) for b in self.banks},
            ))
        mix = fr["sophistication_mix"]
        levels = r.choice([1, 2, 3], size=n_rings, p=np.array([mix[1], mix[2], mix[3]], float))
        self.rings = []
        for ring_id in range(n_rings):
            self._generate_ring(ring_id, int(levels[ring_id]))

    def _ring_size(self) -> int:
        s = self.fr["ring_size"]
        return int(np.clip(round(self.rng.lognormal(math.log(s["median"]), s["sigma"])), s["min"], s["max"]))

    def _mutate(self, s: str) -> str:
        chars, r = list(s), self.rng
        i = int(r.integers(1, max(2, len(chars))))
        i = min(i, len(chars) - 1)
        op = int(r.integers(4))
        if op == 0 and len(chars) > 4:
            del chars[i]
        elif op == 1:
            chars.insert(i, chars[i])
        elif op == 2 and i < len(chars) - 1:
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
        else:
            chars[i] = self._choice(list("aeioulmnrst"))
        out = "".join(chars)
        return out[0].upper() + out[1:]

    def _generate_ring(self, ring_id: int, level: int):
        fr, r = self.fr, self.rng
        op = self._choice(self.operators)
        scenario = {1: "overt", 2: "partial"}.get(level) or self._choice(fr["subtle_scenarios"])
        size = self._ring_size()
        start = self._u(*fr["ring_start_days"])
        spread = fr["onboarding_spread_days"]
        bust_start = start + spread + self._u(*fr["cultivation_days"])
        duration = self._u(*fr["bust_out_days"])
        cross = r.random() < fr["cross_bank_share"]
        ring_bank = self._choice(self.banks, p=self.bank_w)
        ring = dict(ring_id=ring_id, operator_id=op["operator_id"], sophistication=level, scenario=scenario,
                    size=size, start_day=start, bust_out_day=bust_start, cross_bank=cross)
        self.rings.append(ring)

        # ressources de l'anneau (puisées en partie dans le pool de l'opérateur → liens entre vagues)
        if level == 1:
            res = dict(phones=self._choice(op["voip"], size=min(2, len(op["voip"])), replace=False),
                       devices=[self._choice(op["devices"])],
                       ips=self._choice(op["dc_ips"], size=self._u(1, 2), replace=False),
                       addrs=self._choice(op["mail_drops"], size=min(self._u(1, 2), len(op["mail_drops"])), replace=False),
                       employer=self._choice(op["fake_employers"]))
        elif level == 2:
            k = max(1, size // 2)
            phones = [self.new_phone("voip" if r.random() < 0.5 else "mobile", FAR) for _ in range(k)]
            res = dict(phones=phones + [self._choice(op["voip"])],
                       devices=[self.new_device(emulator=r.random() < 0.2) for _ in range(k)],
                       op_device=self._choice(op["devices"]),
                       proxies=self._choice(self.home_ips, size=k + 1, replace=False),
                       dc_ip=self._choice(op["dc_ips"]),
                       addrs=[self.new_address("residential") for _ in range(max(1, size // 3))] + [self._choice(op["mail_drops"])],
                       employer=self._choice(op["fake_employers"]) if r.random() < 0.5 else None)
        else:
            res = dict(employer=self._choice(op["fake_employers"]) if scenario == "shared_employer" else None)

        base_sex = int(r.integers(1, 3))
        base_first = self._choice(self.first_m if base_sex == 1 else self.first_f)
        base_last = self._choice(self.last_names)
        base_birth = int(-r.uniform(22, 50) * 365.25)
        variant_p = {1: 0.5, 2: 0.2, 3: 0.0}[level]
        members: list[dict] = []
        ring_records: list[dict] = []
        for i in range(size):
            onboard = start + self._u(0, spread)
            if i > 0 and r.random() < variant_p:
                first = base_first if r.random() < 0.3 else self._mutate(base_first)
                ident = self._new_identity("synthetic", 0, first_name=first, last_name=self._mutate(base_last),
                                           sex=base_sex, birth_day=base_birth, variant=True)
            else:  # niveau 3 : âge tiré de la population légitime (pas de signature démographique)
                ident = self._new_identity("synthetic", self._choice(self.legit_ages) if level == 3 else r.uniform(22, 50))
            ident.update(ring_id=ring_id, operator_id=op["operator_id"], sophistication=level, scenario=scenario,
                         age0=self._age_at(ident, 0))
            self._synthetic_nid(ident, members, level)
            self._synthetic_contacts(ident, res, level, onboard, op)
            members.append(ident)

            banks = [self._choice(self.banks, p=self.bank_w) if cross else ring_bank]
            if cross and r.random() < fr["spray_share"]:
                banks.append(self._choice([b for b in self.banks if b != banks[0]]))
            bust = bust_start + self._u(0, 7)
            for j, bank in enumerate(banks):
                ob = onboard + (0 if j == 0 else self._u(0, 45))
                end = bust + duration
                rec = self._new_record(ident, bank, ob, close_day=end, bust_out_day=bust, bust_out_end=end,
                                       label_day=end + self._u(*fr["label_delay_days"]),
                                       label_observed=bool(r.random() >= fr["label_noise"]))
                self._synthetic_behavior(rec, level, ident)
                self._static_edges(rec, ident)
                ips = list(ident["ips"])
                if level == 3 and len(ips) > 1 and r.random() < 0.4:  # comme un client : ouverture depuis le mobile
                    ips[0], ips[1] = ips[1], ips[0]
                self._session_edges(rec, ident["devices"], ips, extra_delay=120)
                self._synthetic_links(rec, level, scenario, op)
                ring_records.append(rec)
        self._ring_transfers(ring_records, level, scenario, op)

    def _synthetic_nid(self, ident: dict, members: list, level: int):
        r = self.rng
        reuse_p = {1: 0.5, 2: 0.15, 3: 0.0}[level]
        if members and r.random() < reuse_p:
            ident["nid"] = self._choice(members)["nid"]
        elif r.random() < self.fr["stolen_nid_share"][level]:
            victim = self.identities[int(r.integers(0, self.pop["identities"]))]
            ident["nid"], ident["stolen_nid"] = victim["nid"], True
        else:
            ident["nid"] = self.new_nid(ident["sex"], ident["birth_year"])

    def _synthetic_contacts(self, ident: dict, res: dict, level: int, onboard: int, op: dict):
        r = self.rng
        if level == 1:
            ident["phone"] = self._choice(res["phones"])
            self.touch_phone(ident["phone"], onboard - self._u(0, 30))
            ident["address"] = self._choice(res["addrs"])
            ident["employer"] = res["employer"]
            ident["devices"] = list(res["devices"])
            ident["ips"] = list(res["ips"]) + (self._choice(self.cgnat_pool, size=1) if r.random() < 0.3 else [])
            email_age, disposable = self._u(1, 45), r.random() < 0.5
            ident["file_start_day"] = onboard - self._u(30, 300)
        elif level == 2:
            ident["phone"] = self._choice(res["phones"])
            self.touch_phone(ident["phone"], onboard - self._u(10, 300))
            ident["address"] = self._choice(res["addrs"])
            ident["employer"] = res["employer"] or self._choice(self.employers, p=self.employer_w)
            ident["devices"] = [self._choice(res["devices"])] + ([res["op_device"]] if r.random() < 0.3 else [])
            ident["ips"] = [self._choice(res["proxies"])] + ([res["dc_ip"]] if r.random() < 0.5 else [])
            if r.random() < 0.5:
                ident["ips"].append(self._choice(self.cgnat_pool))
            email_age, disposable = self._u(60, 900), r.random() < 0.1
            ident["file_start_day"] = onboard - self._u(200, 900)
        else:
            # Niveau 3 : profil de souscription tiré des MÊMES distributions que les clients légitimes (âge, emploi,
            # revenu, ancienneté du dossier de crédit « acheté », des contacts). Seuls le réseau et les signaux
            # inter-bancaires peuvent le trahir.
            p = self.pop
            new_phone = r.random() < p["new_phone_share"]
            ident["phone"] = self.new_phone("voip" if r.random() < p["voip_share"] else "mobile",
                                            onboard - (self._u(0, 90) if new_phone else self._u(200, 5000)))
            ident["address"] = self.new_address("residential")
            works = self._works(ident["age0"])
            ident["employer"] = res["employer"] or (self._choice(self.employers, p=self.employer_w) if works else None)
            ident["devices"] = [self.new_device(emulator=r.random() < 0.002)]
            ident["ips"] = [self._choice(self.home_ips)] + self._choice(
                self.cgnat_pool, size=int(r.choice([0, 1, 2], p=[0.25, 0.45, 0.3])), replace=False)
            if r.random() < p["public_wifi_share"]:
                ident["ips"].append(self._choice(self.wifi_pool))
            email_age = self._u(0, 90) if r.random() < p["new_email_share"] else self._u(365, 6000)
            email_age = min(email_age, onboard - (ident["birth_day"] + 12 * 365))
            disposable = r.random() < p["disposable_email_share"]
            adult_day = ident["birth_day"] + int(18 * 365.25)
            ident["file_start_day"] = min(onboard - 30, adult_day + self._u(0, 8 * 365))
            ident["income"] = self._legit_income(ident["age0"], works or res["employer"] is not None)
        ident["email"] = self.new_email(ident["first_name"], ident["last_name"], onboard - email_age, disposable)
        if level < 3:
            ident["income"] = float(r.lognormal(math.log(34000), 0.35))  # revenu déclaré gonflé

    def _synthetic_behavior(self, rec: dict, level: int, ident: dict):
        """Niveau 1 : culture « scolaire » (remboursements parfaits, hausses de plafond en rafale).
        Niveau 2 : imite un bon client « transactor ». Niveau 3 : indiscernable d'un client légitime sur le
        comportement seul — seuls le réseau et les signaux inter-bancaires peuvent le trahir."""
        r = self.rng
        rec.update(limit0=float(round(r.uniform(300, 1500), -2)), grant_p=0.75, lir_mult=1.3, cash_p=0.0)
        if level == 1:
            rec.update(lir_p=r.uniform(0.2, 0.3), util_mu=r.uniform(0.1, 0.3), util_sd=0.02, pay_mu=1.0, pay_sd=0.005,
                       late_p=0.0, txn_lam=r.uniform(6, 14), n_mcc_base=r.uniform(2, 4))
        elif level == 2:
            rec.update(lir_p=r.uniform(0.06, 0.15), util_mu=r.uniform(0.1, 0.35), util_sd=r.uniform(0.03, 0.08),
                       pay_mu=1.0, pay_sd=0.01, late_p=float(r.beta(1, 80)), txn_lam=r.uniform(6, 30),
                       n_mcc_base=r.uniform(3, 9))
        else:
            # plafond initial et politique de hausse identiques à ceux d'un client légitime de même revenu
            transactor = r.random() < self.pop["transactor_share"]
            rec.update(limit0=float(np.clip(round(ident["income"] * r.uniform(0.04, 0.12), -2), 300, 12000)),
                       grant_p=0.5, lir_mult=1.25)
            rec.update(lir_p=r.uniform(0.02, 0.07), util_mu=float(r.beta(2, 5)), util_sd=r.uniform(0.02, 0.12),
                       pay_mu=1.0 if transactor else r.uniform(0.05, 0.6), pay_sd=0.01 if transactor else r.uniform(0.05, 0.2),
                       late_p=float(r.beta(1, 80) if transactor else r.beta(1.5, 25)), txn_lam=r.uniform(4, 50),
                       n_mcc_base=r.uniform(3, 12), cash_p=r.uniform(0.05, 0.2) if r.random() < 0.1 else 0.0)

    def _synthetic_links(self, rec: dict, level: int, scenario: str, op: dict):
        r = self.rng
        bank, pid = rec["bank"], rec["person_id"]
        au = (level == 1 and r.random() < 0.2) or (level == 2 and r.random() < 0.4) or scenario == "tradeline"
        if au:
            for host in op["sellers"][bank]:
                self.au.append((pid, host["person_id"], rec["onboard_day"] + self._u(20, 120)))
        if scenario == "complicit_merchant":
            for m in op["complicit"]:
                self.paid_at.append((pid, m, rec["onboard_day"] + self._u(10, 90), rec["close_day"]))

    def _ring_transfers(self, recs: list, level: int, scenario: str, op: dict):
        r = self.rng
        uses = (level == 1 and r.random() < 0.6) or (level == 2 and r.random() < 0.4) or scenario == "circular_transfers"
        by_bank: dict[str, list] = {}
        for x in recs:
            by_bank.setdefault(x["bank"], []).append(x)
        for bank, group in by_bank.items():
            if uses and len(group) >= 2:
                for a in group:
                    for b in self._choice(group, size=min(len(group), self._u(1, 2)), replace=False):
                        if b is a:
                            continue
                        m0 = max(a["onboard_day"], b["onboard_day"]) // MONTH + 1
                        m1 = min(a["bust_out_day"], b["bust_out_day"]) // MONTH - 1
                        for m in range(max(m0, 0), min(m1, self.M - 1) + 1):
                            if r.random() < 0.5:
                                self.transfers.append((a["person_id"], b["person_id"], m, self._u(1, 3), float(r.uniform(20, 300))))
            for a in group:  # exfiltration vers les mules pendant le bust-out
                m = a["bust_out_day"] // MONTH
                if 0 <= m < self.M:
                    for mule in self._choice(op["mules"][bank], size=self._u(1, 2), replace=False):
                        self.transfers.append((a["person_id"], mule["person_id"], m, self._u(2, 5), float(r.uniform(2000, 9000))))

    # ------------------------------------------------- transactions et comportements
    def generate_links(self):
        """Marchands, virements et utilisateurs autorisés des clients légitimes (+ camouflage des marchands complices)."""
        p, r = self.pop, self.rng
        n_regular = p["n_merchants"]
        for _ in range(n_regular):
            self.new_merchant(high_risk=r.random() < 0.08, complicit=False)
        regular = [m["merchant_id"] for m in self.merchants if not m["complicit"]]
        pop_w = np.arange(1, len(regular) + 1, dtype=float) ** -1.05
        pop_w /= pop_w.sum()
        ids = self.identities
        for rec in self.records:
            ident = ids[rec["identity_id"]]
            if ident["kind"] == "legit":
                k = self._u(8, 25)
            else:
                k = {1: self._u(3, 6), 2: self._u(4, 8), 3: self._u(8, 20)}[ident["sophistication"]]
            for m in self._choice(regular, size=k, replace=False, p=pop_w):
                first = min(rec["onboard_day"] + self._u(0, 300), rec["close_day"] - 1)
                self.paid_at.append((rec["person_id"], m, first, min(rec["close_day"], self.H)))
        legit_recs = [x for x in self.records if ids[x["identity_id"]]["kind"] == "legit"]
        for m in (m for m in self.merchants if m["complicit"]):
            for rec in self._choice(legit_recs, size=self._u(0, 4), replace=False):
                first = min(rec["onboard_day"] + self._u(0, 600), rec["close_day"] - 1)
                self.paid_at.append((rec["person_id"], m["merchant_id"], first, min(rec["close_day"], self.H)))

        by_house_bank: dict[tuple, list] = {}
        for rec in legit_recs:
            by_house_bank.setdefault((ids[rec["identity_id"]]["household_id"], rec["bank"]), []).append(rec)
        for group in by_house_bank.values():
            for a in group:
                for b in group:
                    if a is not b and r.random() < 0.5:
                        self._monthly_transfers(a, b, prob=0.6, amount=(50, 800))
            if len(group) >= 2 and r.random() < 0.15:
                first_day = max(group[0]["onboard_day"], group[1]["onboard_day"]) + self._u(0, 200)
                self.au.append((group[1]["person_id"], group[0]["person_id"], first_day))
        by_bank: dict[str, list] = {}
        for rec in legit_recs:
            by_bank.setdefault(rec["bank"], []).append(rec)
        for rec in legit_recs:
            if r.random() < 0.3:
                for other in self._choice(by_bank[rec["bank"]], size=self._u(1, 2), replace=False):
                    if other is not rec:
                        self._monthly_transfers(rec, other, prob=0.15, amount=(10, 300))

    def _monthly_transfers(self, a: dict, b: dict, prob: float, amount: tuple):
        m0 = max(a["onboard_day"], b["onboard_day"], 0) // MONTH + 1
        m1 = min(a["close_day"], b["close_day"], self.H) // MONTH - 1
        r = self.rng
        months = np.arange(m0, min(m1, self.M - 1) + 1)
        for m in months[r.random(len(months)) < prob]:
            self.transfers.append((a["person_id"], b["person_id"], int(m), self._u(1, 2), float(r.uniform(*amount))))

    def generate_monthly(self) -> pd.DataFrame:
        r = self.rng
        cols = {k: [] for k in ["person_id", "month", "credit_limit", "utilization", "spend", "n_txn", "n_mcc",
                                "pay_ratio", "late", "cash_advance", "limit_increase_request", "limit_increase_granted",
                                "returned_payment"]}
        for rec in self.records:
            m0 = max(0, rec["onboard_day"] // MONTH)
            m1 = min(self.M - 1, -(-min(rec["close_day"], self.H) // MONTH) - 1)
            if m1 < m0:
                continue
            months = np.arange(m0, m1 + 1)
            L = len(months)
            late = r.random(L) < rec["late_p"]
            lir = r.random(L) < rec["lir_p"]
            granted = lir & ~late & (r.random(L) < rec["grant_p"])
            mult = np.where(granted, rec["lir_mult"], 1.0)
            limit = np.minimum(rec["limit0"] * np.concatenate([[1.0], np.cumprod(mult)[:-1]]), 30000.0)
            util = np.clip(rec["util_mu"] + r.normal(0, rec["util_sd"], L), 0.0, 1.0)
            pay = np.clip(rec["pay_mu"] + r.normal(0, rec["pay_sd"], L), 0.0, 1.0)
            txn = r.poisson(rec["txn_lam"], L).astype(float)
            mcc = np.minimum(txn, np.maximum(1, np.round(r.normal(rec["n_mcc_base"], 1, L))))
            cash = np.where(r.random(L) < rec["cash_p"], util * limit * r.uniform(0.05, 0.2, L), 0.0)
            returned = np.zeros(L)

            if rec["is_distressed"]:
                k = np.maximum(0, (months * MONTH + MONTH - rec["default_day"]) / MONTH)
                hit = k > 0
                util = np.where(hit, np.minimum(1.0, util + 0.15 * k), util)
                pay = np.where(hit, np.maximum(0.0, pay - 0.25 * k), pay)
                late = late | (hit & (r.random(L) < np.minimum(1.0, 0.3 * k)))
                lir &= ~hit
                granted &= ~hit
            if rec["bust_out_day"] < FAR:
                bust = (months * MONTH < rec["bust_out_end"]) & (months * MONTH + MONTH > rec["bust_out_day"])
                util = np.where(bust, r.uniform(0.97, 1.0, L), util)
                cash = np.where(bust, util * limit * r.uniform(0.3, 0.7, L), cash)
                returned = np.where(bust, r.integers(1, 4, L), returned)
                pay = np.where(bust, r.uniform(0.8, 1.0, L), pay)
                late = late | bust
                txn = np.where(bust, txn * 2.5, txn)
                lir &= ~bust
                granted &= ~bust

            cols["person_id"].extend([rec["person_id"]] * L)
            cols["month"].append(months)
            cols["credit_limit"].append(limit)
            cols["utilization"].append(util)
            cols["spend"].append(util * limit)
            cols["n_txn"].append(txn)
            cols["n_mcc"].append(np.minimum(mcc, np.maximum(txn, 1)))
            cols["pay_ratio"].append(pay)
            cols["late"].append(late.astype(int))
            cols["cash_advance"].append(cash)
            cols["limit_increase_request"].append(lir.astype(int))
            cols["limit_increase_granted"].append(granted.astype(int))
            cols["returned_payment"].append(returned)
        out = {k: (v if k == "person_id" else np.concatenate(v)) for k, v in cols.items()}
        return pd.DataFrame(out)

    # -------------------------------------------------------------------- sortie
    def run(self) -> dict[str, pd.DataFrame]:
        self.generate_legit()
        self.generate_fraud()
        self.generate_links()
        monthly = self.generate_monthly()

        ident_cols = ["identity_id", "kind", "sex", "first_name", "last_name", "dob", "birth_day", "birth_year",
                      "household_id", "ring_id", "operator_id", "sophistication", "scenario", "is_dorm",
                      "is_colocation", "is_elderly_assisted", "is_newcomer", "is_tradeline_seller", "is_mule",
                      "stolen_nid", "variant", "nid", "phone", "email", "address", "employer", "income",
                      "file_start_day"]
        identities = pd.DataFrame([{k: i.get(k) for k in ident_cols} for i in self.identities])
        records = pd.DataFrame(self.records)
        catalog = pd.DataFrame([dict(attr_type=t, value=v, **meta) for t, d in self.catalog.items() for v, meta in d.items()])
        return dict(
            identities=identities,
            records=records,
            catalog=catalog,
            record_attr=pd.DataFrame(self.record_attr, columns=["person_id", "attr_type", "raw", "value", "first_day", "last_day"]),
            paid_at=pd.DataFrame(self.paid_at, columns=["person_id", "merchant_id", "first_day", "last_day"]),
            merchants=pd.DataFrame(self.merchants),
            transfers=pd.DataFrame(self.transfers, columns=["src", "dst", "month", "n", "amount"]),
            authorized_users=pd.DataFrame(self.au, columns=["person_id", "host_person_id", "first_day"]),
            monthly=monthly,
            rings=pd.DataFrame(self.rings),
        )


def simulate(cfg: dict) -> dict[str, pd.DataFrame]:
    return Simulator(cfg).run()
