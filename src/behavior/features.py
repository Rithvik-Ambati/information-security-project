"""Behavioral feature extraction: sender profile + metadata + content + writing-style.
Sender profiles are built incrementally in time order (no look-ahead leakage)."""
import difflib
import re
from datetime import datetime

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

URGENT = re.compile(r"\b(urgent|immediately|asap|act now|suspend\w*|verify|expires?|within \d+ (hours?|minutes?)|confidential)\b", re.I)
PAYMENT = re.compile(r"\b(wire|transfer|gift cards?|invoice|bank account|payment|bitcoin|password|credentials?)\b", re.I)
URL = re.compile(r"https?://[^\s]+", re.I)
ODD_URL = re.compile(r"//\d+\.\d+|@|-login|-verify|-secure|-reset")
GREET = re.compile(r"^\W*(hi|hello|hey|dear|good (morning|afternoon|evening))\b", re.I)
CLOSE = re.compile(r"\b(thanks|thank you|regards|cheers|best|sincerely|thx)\b\W*\w{0,12}\W*$", re.I)
NAMES = ["sender_known", "n_prior", "hour_typical", "len_ratio", "new_recipient", "prior_urgent_rate",
         "urgent", "payment", "n_urls", "url_odd", "lookalike_domain", "rp_mismatch",
         "n_received", "unsigned", "domain_known", "odd_hour",
         # appended (indices 16+): sender-specific novelty signals
         "payment_novel", "url_novel", "prior_payment_rate", "prior_url_rate", "rcpt_domain_new", "caps_dev",
         # appended (indices 22+): writing-style / relationship signals for the out-of-character detector
         "style_sim", "wordlen_dev", "punct_dev", "greeting", "closing", "is_reply", "rcpt_prior_count",
         "style_known", "greet_dev"]


def _wordlen(body):
    w = body.split()
    return sum(len(x) for x in w) / max(len(w), 1)


def _punct(body):
    return sum(c in ",.;:!?'\"-()" for c in body) / max(len(body), 1)


class Profiles:
    def __init__(self):
        self.s = {}          # sender -> profile dict
        self.domains = set()

    def features(self, e, v=None):
        f = e["from"]
        dom = f.split("@")[1]
        p = self.s.get(f)
        body = e["headers"]["subject"] + " " + e["body"]
        hour = datetime.fromisoformat(e["date"]).hour
        urls = URL.findall(body)
        known = p is not None and p["n"] > 0
        pay = bool(PAYMENT.search(body))
        caps = sum(c.isupper() for c in body) / max(len(body), 1)
        if known:
            near = np.mean([min(abs(hour - h), 24 - abs(hour - h)) <= 2 for h in p["hours"]])
            len_ratio = len(body) / max(p["avg_len"], 1)
            new_rcpt = float(e["headers"]["to"] not in p["rcpts"])
            urate = p["urgent"] / p["n"]
        else:
            near, len_ratio, new_rcpt, urate = 0.5, 1.0, 1.0, 0.0
        look = any(d != dom and difflib.SequenceMatcher(None, dom, d).ratio() >= 0.8 for d in self.domains)
        rp = e["headers"]["return-path"].split("@")[-1] != dom
        return [float(known), p["n"] if known else 0, near, len_ratio, new_rcpt, urate,
                float(bool(URGENT.search(body))), float(bool(PAYMENT.search(body))), len(urls),
                float(any(ODD_URL.search(u) for u in urls)),
                float(look), float(rp), e.get("n_received", 0), float(e["sig"] is None),
                float(dom in self.domains), float(hour < 6 or hour >= 21)] \
            + self._novelty(e, p, known, pay, urls, caps) + self._style(e, p, known, v)

    def _novelty(self, e, p, known, pay, urls, caps):
        if not known:
            return [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        pr, ur = p["pay"] / p["n"], p["url"] / p["n"]
        rd = e["headers"]["to"].split("@")[-1]
        return [float(pay and pr < 0.05), float(bool(urls) and ur < 0.05), pr, ur,
                float(rd not in p["rdoms"]), abs(caps - p["caps"] / p["n"])]

    def _style(self, e, p, known, v):
        """Similarity of this message to the sender's historical voice + relationship signals."""
        body = e["body"]
        greet, close = float(bool(GREET.search(body))), float(bool(CLOSE.search(body)))
        reply = float(e["headers"]["subject"].strip().lower().startswith(("re:", "fw:", "fwd:")))
        if not known or p["n"] < 3 or v is None:
            return [0.5, 0.0, 0.0, greet, close, reply, 0.0, 0.0, 0.0]
        c = p["centroid"]
        cn = np.sqrt(c.multiply(c).sum())
        sim = float(v.multiply(c).sum() / cn) if cn > 0 else 0.5
        n = p["n"]
        return [sim, abs(_wordlen(body) - p["wl"] / n), abs(_punct(body) - p["pu"] / n), greet, close, reply,
                float(p["rcpt_n"].get(e["headers"]["to"], 0)), 1.0, abs(greet - p["gr"] / n)]

    def update(self, e, v=None):
        """Call only for mail accepted as legitimate."""
        f = e["from"]
        body = e["headers"]["subject"] + " " + e["body"]
        p = self.s.setdefault(f, {"n": 0, "hours": [], "avg_len": 0.0, "rcpts": set(), "urgent": 0,
                                  "pay": 0, "url": 0, "rdoms": set(), "caps": 0.0,
                                  "centroid": None, "wl": 0.0, "pu": 0.0, "gr": 0.0, "rcpt_n": {}})
        p["pay"] += bool(PAYMENT.search(body))
        p["url"] += bool(URL.search(body))
        p["rdoms"].add(e["headers"]["to"].split("@")[-1])
        p["caps"] += sum(c.isupper() for c in body) / max(len(body), 1)
        p["wl"] += _wordlen(e["body"])
        p["pu"] += _punct(e["body"])
        p["gr"] += float(bool(GREET.search(e["body"])))
        p["rcpt_n"][e["headers"]["to"]] = p["rcpt_n"].get(e["headers"]["to"], 0) + 1
        if v is not None:
            p["centroid"] = v.copy() if p["centroid"] is None else p["centroid"] + v
        p["avg_len"] = (p["avg_len"] * p["n"] + len(body)) / (p["n"] + 1)
        p["n"] += 1
        p["hours"].append(datetime.fromisoformat(e["date"]).hour)
        p["rcpts"].add(e["headers"]["to"])
        p["urgent"] += bool(URGENT.search(body))
        self.domains.add(f.split("@")[1])


def style_vectors(emails, fit_until):
    """Char n-gram TF-IDF fit on legitimate mail BEFORE `fit_until` only (no leakage), applied to all mail."""
    texts = [e["body"] for e in emails]
    fit_txt = [t for t, e in zip(texts[:fit_until], emails[:fit_until]) if e["label"] == 0]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), max_features=30000, sublinear_tf=True,
                          min_df=2).fit(fit_txt)
    return vec.transform(texts).tocsr(), vec


def build_matrix(emails, fit_until=None, return_state=False):
    """Walk emails in time order; features use only the past. fit_until: end of the history window.
    return_state=True also returns (Profiles after the last email, fitted style vectorizer)."""
    fit_until = fit_until or int(0.4 * len(emails))
    V, vec = style_vectors(emails, fit_until)
    pr, X = Profiles(), []
    for i, e in enumerate(emails):
        X.append(pr.features(e, V[i]))
        if e["label"] == 0:
            pr.update(e, V[i])
    if return_state:
        return np.array(X, dtype=float), NAMES, pr, vec
    return np.array(X, dtype=float), NAMES
