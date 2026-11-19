"""Cryptographic Trust module: DKIM-style signing/verification + graded Trust Score in [0,1].

Research-prototype simplification: Ed25519 signatures over a canonical string of
selected headers + SHA-256 body hash (DKIM-like, not RFC 6376 byte-exact).
"""
import base64
import hashlib
import json
from dataclasses import dataclass, field

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

SIGNED_HEADERS = ("from", "to", "subject", "date", "message-id")


def body_hash(body: str) -> str:
    return base64.b64encode(hashlib.sha256(body.encode()).digest()).decode()


def canonical(headers: dict, bh: str) -> bytes:
    parts = [f"{h}:{headers.get(h, '').strip()}" for h in SIGNED_HEADERS]
    parts.append(f"bh:{bh}")
    return "\n".join(parts).encode()


def new_keypair():
    priv = Ed25519PrivateKey.generate()
    return priv, priv.public_key()


def pub_bytes(pub: Ed25519PublicKey) -> bytes:
    return pub.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


def fingerprint(pub: Ed25519PublicKey) -> str:
    return hashlib.sha256(pub_bytes(pub)).hexdigest()[:16]


def sign_email(headers: dict, body: str, priv: Ed25519PrivateKey, domain: str, selector: str = "s1") -> dict:
    """Return a signature block to attach to the email."""
    bh = body_hash(body)
    sig = priv.sign(canonical(headers, bh))
    return {"d": domain, "s": selector, "bh": bh, "b": base64.b64encode(sig).decode()}


def domain_of(addr: str) -> str:
    return addr.rsplit("@", 1)[-1].strip(" >").lower()


@dataclass
class KeyRegistry:
    """Trust-on-first-use registry: sender domain -> (selector, fingerprint, pubkey)."""
    keys: dict = field(default_factory=dict)

    def learn(self, domain, selector, pub):
        self.keys[domain] = {"selector": selector, "fp": fingerprint(pub), "pub": pub}


@dataclass
class TrustResult:
    score: float
    checks: dict
    reasons: list


# weights sum to 1.0 -> tune later (or learn via meta-classifier)
WEIGHTS = {"signature": 0.35, "body_hash": 0.10, "alignment": 0.15, "key_continuity": 0.15,
           "header_consistency": 0.10, "domain_legitimacy": 0.15}


def lookalike_of(domain: str, registry: "KeyRegistry"):
    """Return a known domain this one imitates (similar but not identical), else None."""
    import difflib
    for known in registry.keys:
        if known != domain and difflib.SequenceMatcher(None, domain, known).ratio() >= 0.8:
            return known
    return None


def verify(email: dict, registry: KeyRegistry) -> TrustResult:
    """email = {headers:{...}, body:str, sig:dict|None, pubkey:Ed25519PublicKey|None}"""
    h, body, sig = email["headers"], email["body"], email.get("sig")
    checks = {k: 0.0 for k in WEIGHTS}
    reasons = []
    from_dom = domain_of(h.get("from", ""))
    imitated = lookalike_of(from_dom, registry)
    checks["domain_legitimacy"] = 0.0 if imitated else 1.0
    if imitated:
        reasons.append(f"Lookalike domain: {from_dom} resembles known domain {imitated}")

    if not sig:
        reasons.append("No signature present")
        checks["key_continuity"] = 0.5  # nothing to compare
        checks["header_consistency"] = 1.0 if domain_of(h.get("return-path", h.get("from", ""))) == from_dom else 0.0
        return _finish(checks, reasons)

    # body hash
    checks["body_hash"] = 1.0 if sig["bh"] == body_hash(body) else 0.0
    if not checks["body_hash"]:
        reasons.append("Body hash mismatch (content altered after signing)")

    # signature: verify using key published for the signing domain (registry or supplied DNS key)
    pub = email.get("pubkey") or (registry.keys.get(sig["d"]) or {}).get("pub")
    if pub is None:
        reasons.append("No public key available for signing domain")
    else:
        try:
            pub.verify(base64.b64decode(sig["b"]), canonical(h, sig["bh"]))
            checks["signature"] = 1.0
        except InvalidSignature:
            reasons.append("Signature invalid (headers tampered or wrong key)")

    # alignment: signing domain must match From domain (DMARC-like)
    checks["alignment"] = 1.0 if sig["d"] == from_dom else 0.0
    if not checks["alignment"]:
        reasons.append(f"Signing domain {sig['d']} != From domain {from_dom}")

    # key continuity (TOFU)
    known = registry.keys.get(from_dom)
    if known is None:
        checks["key_continuity"] = 0.5
        reasons.append("First time seeing this sender domain")
    else:
        same_key = pub is not None and known["fp"] == fingerprint(pub)
        same_sel = known["selector"] == sig["s"]
        checks["key_continuity"] = 1.0 if same_key and same_sel else 0.0
        if not same_key:
            reasons.append("DKIM key fingerprint changed vs. stored key")
        elif not same_sel:
            reasons.append("New DKIM selector")

    # header consistency: From vs Return-Path
    rp = domain_of(h.get("return-path", h.get("from", "")))
    checks["header_consistency"] = 1.0 if rp == from_dom else 0.0
    if not checks["header_consistency"]:
        reasons.append("From and Return-Path domains differ")

    return _finish(checks, reasons)


def _finish(checks, reasons):
    score = sum(WEIGHTS[k] * v for k, v in checks.items())
    if checks["body_hash"] == 0 or checks["signature"] == 0:
        score = min(score, 0.4)  # a broken signature/body hash is a hard integrity failure, not a soft penalty
    return TrustResult(round(score, 3), checks, reasons)
