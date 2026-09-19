"""Normalisation des PII avant tokenisation : deux saisies d'une même valeur doivent donner le même jeton."""
import re
import unicodedata


def strip_accents(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


def norm_phone(raw: str) -> str:
    """Format E.164 (+33…), quelle que soit la saisie : '06 12 34 56 78', '+33612345678', '0612345678'."""
    digits = re.sub(r"\D", "", raw)
    if digits.startswith("33"):
        return "+" + digits
    if digits.startswith("0"):
        return "+33" + digits[1:]
    return "+" + digits


def norm_email(raw: str) -> str:
    return raw.strip().lower()


def norm_address(raw: str) -> str:
    s = strip_accents(raw).lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def norm_name(raw: str) -> str:
    s = strip_accents(raw).lower().replace("-", " ")
    s = re.sub(r"[^a-z ]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def norm_nid(raw: str) -> str:
    return re.sub(r"\D", "", raw)


def norm_plain(raw: str) -> str:
    return raw.strip().lower()


NORMALIZERS = {
    "phone": norm_phone,
    "email": norm_email,
    "address": norm_address,
    "nid": norm_nid,
    "device": norm_plain,
    "ip": norm_plain,
    "employer": norm_address,
}


def identity_key(first_name: str, last_name: str, dob: str) -> str:
    return f"{norm_name(first_name)} {norm_name(last_name)}|{dob}"
