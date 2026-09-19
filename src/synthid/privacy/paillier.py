"""Paillier (additivement homomorphe) : clés via `phe`, chiffrement « brut » parallélisé.

- Packing : ⌊(|n| − 2) / slot_bits⌋ compteurs par chiffré ; la somme homomorphe reste exacte tant qu'aucun
  slot ne déborde (sommes < 2^slot_bits).
- Virgule fixe signée (échelle 2^FRAC) pour le scoring chiffré : Enc(x)^w = Enc(w·x).
Clé de 3072 bits ≈ 128 bits de sécurité (NIST SP 800-57, équivalence avec la factorisation).
"""
from __future__ import annotations

import os
import secrets
from concurrent.futures import ThreadPoolExecutor

import gmpy2
from phe import paillier

FRAC = 40  # bits de partie fractionnaire en virgule fixe
SECURITY_BITS = {2048: 112, 3072: 128, 4096: 152}


def keypair(bits: int):
    return paillier.generate_paillier_keypair(n_length=bits)


def _encrypt_many(args: tuple[int, list[int]]) -> list[int]:
    n, plaintexts = args
    gmpy2.get_context().allow_release_gil = True  # powmod libère le GIL : parallélisable par threads
    N, nsq = gmpy2.mpz(n), gmpy2.mpz(n * n)
    out = []
    for m in plaintexts:
        r = gmpy2.mpz(secrets.randbelow(n - 1) + 1)
        out.append(int((1 + gmpy2.mpz(m) * N) % nsq * gmpy2.powmod(r, N, nsq) % nsq))
    return out


def encrypt_raw(n: int, plaintexts: list[int], workers: int | None = None) -> list[int]:
    """Enc(m) = (1 + m·n) · r^n mod n² (g = n + 1), réparti sur des threads.

    Des threads plutôt que des processus : gmpy2 libère le GIL pendant powmod, et un pool de processus
    « spawn » se bloque sous Windows quand la CLI est lancée via le lanceur `synthid.exe`."""
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    if len(plaintexts) < 4 or workers == 1:
        return _encrypt_many((n, plaintexts))
    chunks = [(n, plaintexts[i::workers]) for i in range(workers)]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        parts = list(ex.map(_encrypt_many, chunks))
    out = [0] * len(plaintexts)
    for i, part in enumerate(parts):
        out[i::workers] = part
    return out


def add(n: int, a: int, b: int) -> int:
    return a * b % (n * n)


def scalar_mul(n: int, c: int, k: int) -> int:
    return int(gmpy2.powmod(c, k % n, n * n))


def decrypt_raw(sk, c: int) -> int:
    return sk.raw_decrypt(c)


# ------------------------------------------------------------------ packing
def slots_per_ciphertext(n: int, slot_bits: int) -> int:
    return (n.bit_length() - 2) // slot_bits


def pack(values: list[int], n: int, slot_bits: int) -> list[int]:
    s = slots_per_ciphertext(n, slot_bits)
    out = []
    for i in range(0, len(values), s):
        m = 0
        for j, v in enumerate(values[i:i + s]):
            if not 0 <= v < 2**slot_bits:
                raise OverflowError("compteur hors de la capacité d'un slot")
            m |= int(v) << (slot_bits * j)
        out.append(m)
    return out


def unpack(plaintexts: list[int], length: int, slot_bits: int, n: int) -> list[int]:
    s = slots_per_ciphertext(n, slot_bits)
    mask = 2**slot_bits - 1
    out = []
    for m in plaintexts:
        out.extend((m >> (slot_bits * j)) & mask for j in range(s))
    return out[:length]


# ------------------------------------------------------------ virgule fixe
def encode_fixed(x: float, n: int) -> int:
    return int(round(x * 2**FRAC)) % n


def decode_fixed(m: int, n: int, scale_bits: int) -> float:
    if m > n // 2:
        m -= n
    return m / 2**scale_bits
