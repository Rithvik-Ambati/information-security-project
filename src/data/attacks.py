"""Synthetic authentication layer: sign legitimate mail and generate attack variants.

Scenarios (attack_type):
  legit      - legitimate mail; valid signature (or, ~8%, unsigned: lists/forwarders/legacy servers)
  spoof      - phishing mail forging a known sender, no valid signature, mismatched Return-Path
  takeover   - phishing body sent through a real sender's signing key (valid DKIM!)
  lookalike  - phishing from a typosquatted domain with its own valid signature
  tamper     - legit signed mail whose body was swapped after signing

Realism knobs: some domains rotate their DKIM key/selector mid-timeline (legit mail that breaks key
continuity), and attack send-times are only *sometimes* unusual (half are uniformly random hours,
half keep the victim's normal hour).
"""
import random
from datetime import datetime

from crypto.trust import new_keypair, pub_bytes, sign_email


def lookalike(domain: str, rng) -> str:
    name, _, tld = domain.rpartition(".")
    swaps = [("o", "0"), ("l", "1"), ("i", "l"), ("m", "rn"), ("a", "e")]
    for a, b in rng.sample(swaps, len(swaps)):
        if a in name:
            return name.replace(a, b, 1) + "." + tld
    return name + "-secure." + tld


class Signer:
    """One keypair per sending domain and key generation (as with real DKIM selectors)."""

    def __init__(self, seed=13, p_unsigned=0.08, p_odd_hour=0.5):
        self.rng = random.Random(seed)
        self.keys = {}        # (domain, generation) -> (priv, pub)
        self.rotation = {}    # domain -> ISO date after which generation 1 (selector s2) is used
        self.p_unsigned, self.p_odd_hour = p_unsigned, p_odd_hour

    def set_rotation(self, domain, date_iso):
        self.rotation[domain] = date_iso

    def _gen(self, domain, date):
        return 1 if (domain in self.rotation and date and date >= self.rotation[domain]) else 0

    def key(self, domain, gen=0):
        if (domain, gen) not in self.keys:
            self.keys[(domain, gen)] = new_keypair()
        return self.keys[(domain, gen)]

    def registry_dump(self):
        return {f"{d}#{g}": {"selector": f"s{g + 1}", "pub_hex": pub_bytes(p).hex()}
                for (d, g), (_, p) in self.keys.items()}

    def _headers(self, r, from_addr, rp=None):
        return {"from": from_addr, "to": r["to"] or "user@corp.com", "subject": r["subject"],
                "date": r["date"] or "", "message-id": f"<{self.rng.randint(1, 10**12)}@{from_addr.split('@')[1]}>",
                "return-path": rp or from_addr}

    def make(self, r, attack, phish_body=None):
        sender = r["from"]
        dom = sender.split("@")[1]
        orig_date = r["date"]
        gen = self._gen(dom, orig_date)
        body, from_addr, rp, sig_dom = r["body"], sender, None, dom
        sig_priv, sig_pub = self.key(dom, gen)
        selector = f"s{gen + 1}"
        if attack == "spoof":
            body = phish_body
            rp = f"bounce@{lookalike(dom, self.rng)}"
        elif attack == "takeover":
            body = phish_body
        elif attack == "lookalike":
            body = phish_body
            sig_dom = lookalike(dom, self.rng)
            from_addr = f"{sender.split('@')[0]}@{sig_dom}"
            sig_priv, sig_pub = self.key(sig_dom, 0)
            selector = "s1"
        date = orig_date
        if attack in ("spoof", "takeover", "lookalike") and date and self.rng.random() < self.p_odd_hour:
            date = datetime.fromisoformat(date).replace(hour=self.rng.randint(0, 23)).isoformat()
        r = dict(r, date=date)
        h = self._headers(r, from_addr, rp)
        unsigned = attack == "spoof" or (attack == "legit" and self.rng.random() < self.p_unsigned)
        sig = None if unsigned else sign_email(h, body, sig_priv, sig_dom, selector)
        if attack == "tamper":
            body = phish_body
        return {"from": from_addr, "true_sender": sender, "headers": h, "body": body, "sig": sig,
                "pub_hex": None if unsigned else pub_bytes(sig_pub).hex(),
                "date": date, "n_received": r.get("n_received", 0),
                "attack_type": attack, "label": 0 if attack == "legit" else 1}
