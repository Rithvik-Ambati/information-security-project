"""Synthetic signed-mail world: legit senders + 5 attack scenarios. Used by the demo and tests."""
import copy
import random

from crypto.trust import KeyRegistry, new_keypair, sign_email
from behavior.risk import SenderProfile

LEGIT_BODIES = [
    "Hi team, attaching the weekly report. Let me know if you have questions.",
    "Meeting moved to 3pm tomorrow. Agenda is in the shared doc.",
    "Thanks for the update, I will review the draft by Friday.",
    "Lunch on Thursday? The new place near the office looks good.",
    "Reminder: sprint demo is next week. Please update your tickets.",
    "Sharing the notes from today's call. Action items are at the bottom.",
]
PHISH_BODY = ("URGENT: please wire the pending invoice payment to the new bank account immediately. "
              "Verify your details at http://secure-pay-verify.example/login and keep this confidential.")


class World:
    def __init__(self, seed=7):
        self.rng = random.Random(seed)
        self.registry = KeyRegistry()
        self.profiles = {}
        self.senders = {}
        domain_keys = {}  # DKIM keys are per signing domain, shared by its users
        for name, dom in [("alice", "corp.com"), ("bob", "corp.com"), ("carol", "partner.org")]:
            if dom not in domain_keys:
                domain_keys[dom] = new_keypair()
            priv, pub = domain_keys[dom]
            addr = f"{name}@{dom}"
            self.senders[addr] = {"priv": priv, "pub": pub, "domain": dom}
            self.registry.learn(dom, "s1", pub)
            prof = SenderProfile()
            for _ in range(30):  # sender history: business hours, calm tone
                prof.update(self.rng.randint(9, 17), self.rng.choice(LEGIT_BODIES))
            self.profiles[addr] = prof

    def _base(self, addr, body, hour):
        s = self.senders[addr]
        h = {"from": addr, "to": "me@corp.com", "subject": "Update", "date": f"2026-11-15 {hour:02d}:00",
             "message-id": f"<{self.rng.randint(1, 10**9)}@{s['domain']}>", "return-path": addr}
        return {"headers": h, "body": body, "hour": hour,
                "sig": sign_email(h, body, s["priv"], s["domain"])}

    def scenario(self, name: str):
        addr = "alice@corp.com"
        if name == "Legitimate":
            return self._base(addr, self.rng.choice(LEGIT_BODIES), 11), addr
        if name == "Spoofed sender (no signature)":
            e = self._base(addr, PHISH_BODY, 3); e["sig"] = None
            e["headers"]["return-path"] = "bounce@evil.example"
            return e, addr
        if name == "Account takeover (valid signature!)":
            return self._base(addr, PHISH_BODY, 3), addr
        if name == "Lookalike domain":
            priv, pub = new_keypair()
            fake = "alice@c0rp.com"
            h = {"from": fake, "to": "me@corp.com", "subject": "Update", "date": "2026-11-15 03:00",
                 "message-id": "<1@c0rp.com>", "return-path": fake}
            return {"headers": h, "body": PHISH_BODY, "hour": 3, "pubkey": pub,
                    "sig": sign_email(h, PHISH_BODY, priv, "c0rp.com")}, fake
        if name == "Header/body tampering":
            e = self._base(addr, self.rng.choice(LEGIT_BODIES), 11)
            e["body"] = PHISH_BODY  # content swapped after signing
            return e, addr
        raise KeyError(name)


SCENARIOS = ["Legitimate", "Spoofed sender (no signature)", "Account takeover (valid signature!)",
             "Lookalike domain", "Header/body tampering"]
