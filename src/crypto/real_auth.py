"""Real-world email authentication: DKIM (dkimpy), SPF (pyspf) and DMARC-style alignment on raw .eml files.

verify_eml(raw_bytes) -> {"email": <pipeline-format dict>, "trust": TrustResult, "details": {...}}
Key continuity uses a persistent trust-on-first-use store of each domain's DKIM public-key fingerprint.
`dnsfunc` lets tests/demos supply DKIM TXT records offline; by default real DNS is used (needs network).
"""
import difflib
import email
import hashlib
import json
import logging
import os
import re
from email import policy
from email.utils import parseaddr, parsedate_to_datetime

import dkim

from crypto.trust import WEIGHTS, _finish  # reuse the same graded scoring + integrity cap

STORE = os.path.join(os.path.dirname(__file__), "..", "..", "data", "real_key_store.json")


def _load_store(path):
    try:
        return json.load(open(path))
    except Exception:
        return {}


def _org(domain):
    """Crude organizational domain (last two labels); enough for alignment in a prototype."""
    p = domain.lower().strip(".").split(".")
    return ".".join(p[-2:])


def _tags(sig_header: str):
    return dict(m.groups() for m in re.finditer(r"\b([a-z]{1,2})=([^;]*)", re.sub(r"\s+", "", sig_header)))


def parse_eml(raw: bytes):
    """Raw RFC 5322 bytes -> (email dict in the pipeline's format, parsed message)."""
    msg = email.message_from_bytes(raw, policy=policy.default)
    from_addr = parseaddr(str(msg.get("From", "")))[1].lower()
    rp = parseaddr(str(msg.get("Return-Path", "")))[1].lower() or from_addr
    try:
        date = parsedate_to_datetime(str(msg.get("Date", ""))).replace(tzinfo=None).isoformat()
    except Exception:
        date = "2000-01-01T12:00:00"
    try:
        part = msg.get_body(preferencelist=("plain", "html"))
        body = part.get_content() if part else ""
    except Exception:
        body = ""
    body = " ".join(str(body).split())[:5000]
    d = {"from": from_addr, "true_sender": from_addr, "body": body, "date": date,
         "n_received": len(msg.get_all("Received", []) or []), "attack_type": "unknown", "label": 0,
         "headers": {"from": from_addr, "to": parseaddr(str(msg.get("To", "")))[1].lower() or "user@local",
                     "subject": str(msg.get("Subject", ""))[:200], "date": date,
                     "message-id": str(msg.get("Message-ID", "")), "return-path": rp},
         "sig": None, "pub_hex": None}
    return d, msg


def _dkim_check(raw, dnsfunc):
    log = logging.getLogger("dkim_probe")
    log.handlers.clear()
    log.propagate = False
    records = []

    class H(logging.Handler):
        def emit(self, r):
            records.append(r.getMessage())
    log.addHandler(H())
    log.setLevel(logging.DEBUG)
    ok, err = False, None
    try:
        ok = bool(dkim.DKIM(raw, logger=log).verify(dnsfunc=dnsfunc) if dnsfunc else dkim.DKIM(raw, logger=log).verify())
    except Exception as ex:  # DNS failure, malformed signature, ...
        err = str(ex)
    return ok, err, " ".join(records)


def _spf_check(msg, mailfrom, dns_ok=True):
    try:
        import spf
        recv = msg.get_all("Received", []) or []
        m = re.search(r"\[(\d+\.\d+\.\d+\.\d+)\]", str(recv[-1] if recv else ""))
        if not m or "@" not in mailfrom:
            return "none"
        helo = mailfrom.split("@")[1]
        return spf.check2(i=m.group(1), s=mailfrom, h=helo)[0]
    except Exception:
        return "temperror"


def verify_eml(raw: bytes, dnsfunc=None, store_path=STORE, learn=False, check_spf=True):
    e, msg = parse_eml(raw)
    store = _load_store(store_path)
    from_dom = e["from"].split("@")[-1]
    checks = {k: 0.0 for k in WEIGHTS}
    reasons, det = [], {}

    # domain legitimacy: lookalike of a domain already trusted in the store
    imitated = next((d for d in store if d != from_dom and difflib.SequenceMatcher(None, from_dom, d).ratio() >= 0.8), None)
    checks["domain_legitimacy"] = 0.0 if imitated else 1.0
    if imitated:
        reasons.append(f"Lookalike domain: {from_dom} resembles trusted domain {imitated}")

    sig_hdr = msg.get("DKIM-Signature")
    rp_dom = e["headers"]["return-path"].split("@")[-1]
    checks["header_consistency"] = 1.0 if rp_dom == from_dom else 0.0
    if checks["header_consistency"] == 0:
        reasons.append("From and Return-Path domains differ")

    if not sig_hdr:
        reasons.append("No DKIM signature present")
        checks["key_continuity"] = 0.5
        det["dkim"] = "none"
    else:
        t = _tags(str(sig_hdr))
        sd, sel = t.get("d", "").lower(), t.get("s", "")
        ok, err, logtxt = _dkim_check(raw, dnsfunc)
        det.update(dkim="pass" if ok else "fail", d=sd, s=sel, dkim_error=err)
        body_bad = "body hash mismatch" in logtxt.lower()
        checks["signature"] = 1.0 if ok else 0.0
        checks["body_hash"] = 0.0 if body_bad else (1.0 if ok else 0.5)
        if not ok:
            reasons.append("DKIM verification FAILED" + (" (body hash mismatch: content altered)" if body_bad else ""))
        checks["alignment"] = 1.0 if _org(sd) == _org(from_dom) else 0.0
        if not checks["alignment"]:
            reasons.append(f"DKIM domain {sd} not aligned with From domain {from_dom}")
        # key continuity via TOFU on the published key fingerprint
        fp = None
        try:
            txt = dnsfunc(f"{sel}._domainkey.{sd}.".encode()) if dnsfunc else None
            m = re.search(rb"p=([A-Za-z0-9+/=]+)", txt or b"")
            fp = hashlib.sha256(m.group(1)).hexdigest()[:16] if m else None
        except Exception:
            pass
        known = store.get(from_dom)
        if known is None or fp is None:
            checks["key_continuity"] = 0.5
            reasons.append("First time seeing this sender domain" if known is None else "Could not read DKIM key")
        else:
            same = known["fp"] == fp and known["selector"] == sel
            checks["key_continuity"] = 1.0 if same else 0.0
            if not same:
                reasons.append("DKIM key fingerprint/selector changed vs. stored key")
        if learn and ok and fp and checks["alignment"]:
            store[from_dom] = {"selector": sel, "fp": fp}
            json.dump(store, open(store_path, "w"), indent=1)

    if check_spf:
        det["spf"] = _spf_check(msg, e["headers"]["return-path"])
    return {"email": e, "trust": _finish(checks, reasons), "details": det}
