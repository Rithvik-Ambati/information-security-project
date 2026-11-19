"""Generate real RFC-5322 .eml samples signed with REAL RSA DKIM (dkimpy) for the upload demo / tests.
Writes demo/samples/*.eml plus demo/samples/dns_records.json (the DKIM TXT records, served offline by the demo)
and data/real_key_store.json (trusted key fingerprints, i.e. the 'known sender' history)."""
import base64
import hashlib
import json
import os
import sys

import dkim
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.path.join(ROOT, "demo", "samples")
os.makedirs(OUT, exist_ok=True)


def keypair():
    k = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = k.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL,
                          serialization.NoEncryption())
    der = k.public_key().public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
    return pem, base64.b64encode(der).decode()


def raw(frm, to, subj, body, date, rp=None, hour_note=""):
    rp = rp or frm
    return (f"Return-Path: <{rp}>\r\nReceived: from mail.{frm.split('@')[1]} ([203.0.113.7]) by mx.corp.example; {date}\r\n"
            f"From: {frm}\r\nTo: {to}\r\nSubject: {subj}\r\nDate: {date}\r\nMessage-ID: <{abs(hash(body)) % 10**9}@{frm.split('@')[1]}>\r\n"
            f"MIME-Version: 1.0\r\nContent-Type: text/plain; charset=utf-8\r\n\r\n{body}\r\n").encode()


def sign(msg, pem, domain, selector="s1"):
    return dkim.sign(msg, selector.encode(), domain.encode(), pem, include_headers=[b"from", b"to", b"subject", b"date"]) + msg


pem_c, pub_c = keypair()          # corp.example (the victim's real domain)
pem_x, pub_x = keypair()          # attacker's own look-alike domain
dns = {"s1._domainkey.corp.example.": f"v=DKIM1; k=rsa; p={pub_c}",
       "s1._domainkey.c0rp.example.": f"v=DKIM1; k=rsa; p={pub_x}"}
TO = "me@corp.example"
D1, D2 = "Mon, 16 Nov 2026 10:15:00 +0000", "Tue, 17 Nov 2026 03:12:00 +0000"
OK_BODY = "Hi, attaching the weekly report. Let me know if you have questions. Thanks, Alice"
BEC = ("Are you at your desk? I need you to wire $4,800 to a new vendor today and keep this confidential. "
       "I am in a meeting and cannot talk.")

samples = {
    "1_legit_signed.eml": sign(raw("alice@corp.example", TO, "Weekly report", OK_BODY, D1), pem_c, "corp.example"),
    "2_tampered_after_signing.eml": sign(raw("alice@corp.example", TO, "Weekly report", OK_BODY, D1), pem_c,
                                         "corp.example").replace(OK_BODY.encode(), BEC.encode()),
    "3_spoofed_unsigned.eml": raw("alice@corp.example", TO, "Urgent payment", BEC, D2, rp="bounce@evil.test"),
    "4_account_takeover_valid_dkim.eml": sign(raw("alice@corp.example", TO, "Quick favour", BEC, D2), pem_c,
                                              "corp.example"),
    "5_lookalike_domain.eml": sign(raw("alice@c0rp.example", TO, "Quick favour", BEC, D2), pem_x, "c0rp.example"),
}
for name, data in samples.items():
    open(os.path.join(OUT, name), "wb").write(data)
json.dump(dns, open(os.path.join(OUT, "dns_records.json"), "w"), indent=1)
# trusted-key history: corp.example's key has been seen before
store = {"corp.example": {"selector": "s1", "fp": hashlib.sha256(pub_c.encode()).hexdigest()[:16]}}
json.dump(store, open(os.path.join(ROOT, "data", "real_key_store.json"), "w"), indent=1)
print("wrote", len(samples), "samples to", OUT)
