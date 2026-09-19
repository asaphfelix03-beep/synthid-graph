"""OPRF « 2HashDH » sur secp256k1 (libsecp256k1 via coincurve).

    token(x) = H2(x, k · H1(x))

La banque aveugle H1(x) avec un scalaire aléatoire r, le service de tokenisation applique sa clé k sans voir x,
la banque retire r. Résultat : un jeton identique pour toutes les banques, qu'aucune banque ne peut calculer
hors ligne (pas d'attaque par dictionnaire sans interroger le service, qui peut limiter le débit), et que le
service ne peut pas relier à une PII (il ne voit que des points aléatoires).

PoC : hachage vers la courbe par « try-and-increment » ; en production, suivre la RFC 9497 (OPRF) et la
RFC 9380 (hash-to-curve), idéalement avec ristretto255 ou P-256.
"""
from __future__ import annotations

import hashlib
import secrets

from coincurve import PublicKey

ORDER = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141


def hash_to_curve(value: bytes) -> PublicKey:
    for ctr in range(256):
        x = hashlib.sha256(b"synthid-h2c|" + bytes([ctr]) + value).digest()
        try:
            return PublicKey(b"\x02" + x)
        except Exception:
            continue
    raise ValueError("hash_to_curve : échec")


class TokenizationService:
    """Détient la clé OPRF k. Ne reçoit que des points aveuglés ; journalise ce qu'il voit (tests de fuite)."""

    def __init__(self, max_evaluations: int | None = None):
        self._k = (secrets.randbelow(ORDER - 1) + 1).to_bytes(32, "big")
        self.max_evaluations = max_evaluations
        self.evaluations = 0
        self.received: list[bytes] = []

    def evaluate(self, blinded: list[bytes]) -> list[bytes]:
        self.evaluations += len(blinded)
        if self.max_evaluations is not None and self.evaluations > self.max_evaluations:
            raise PermissionError("limite de débit OPRF atteinte")
        self.received.extend(blinded)
        return [PublicKey(b).multiply(self._k).format() for b in blinded]


def tokenize(values: list[str], service: TokenizationService, batch: int = 5000) -> dict[str, str]:
    """Côté banque : aveuglement → évaluation par le service → dé-aveuglement → jeton."""
    out: dict[str, str] = {}
    for i in range(0, len(values), batch):
        chunk = values[i:i + batch]
        rs = [secrets.randbelow(ORDER - 1) + 1 for _ in chunk]
        blinded = [hash_to_curve(v.encode()).multiply(r.to_bytes(32, "big")).format() for v, r in zip(chunk, rs)]
        evaluated = service.evaluate(blinded)
        for v, r, e in zip(chunk, rs, evaluated):
            point = PublicKey(e).multiply(pow(r, -1, ORDER).to_bytes(32, "big")).format()
            out[v] = hashlib.sha256(b"synthid-oprf|" + v.encode() + b"|" + point).hexdigest()[:32]
    return out
