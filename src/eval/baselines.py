"""Baselines on the time-ordered split.
  A. content-only   TF-IDF + logistic regression (trained on 'train', scored on 'test')
  B. auth-only      Trust Score thresholded (SPF/DKIM/DMARC-style, no behavior)
  C. simple combo   content OR auth  (flag if either fires)
Our fusion (Trust + Risk) is compared against these later.
"""
import json
import os
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from crypto.trust import KeyRegistry, verify

P = os.environ.get("EMAIL_DATA_DIR") or os.path.join(ROOT, "data", "processed")


def load():
    """Returns (emails, splits, None). Each signed email carries pub_hex = the key its domain publishes
    (simulated DNS lookup of the selector); the third value is kept only for backward compatibility."""
    emails = [json.loads(l) for l in open(os.path.join(P, "emails.jsonl"))]
    sp = json.load(open(os.path.join(P, "splits.json")))
    return emails, sp, None


def trust_timeline(emails, dns=None):
    """Walk mail in time order. Verify each email against the key registry, THEN (for mail the user/admin
    confirms legitimate) learn its domain key (trust-on-first-use, updated on confirmed rotations).
    Returns a list of TrustResult. Only past mail influences each verdict (no look-ahead)."""
    reg = KeyRegistry()
    out = []
    for e in emails:
        sig = e["sig"]
        pub = Ed25519PublicKey.from_public_bytes(bytes.fromhex(e["pub_hex"])) if e.get("pub_hex") else None
        out.append(verify(dict(e, pubkey=pub), reg))
        if e["label"] == 0 and sig and pub is not None:
            d = e["from"].split("@")[1]
            if sig["d"] == d:
                reg.learn(d, sig["s"], pub)
    return out


def trust_scores(emails, sp=None, dns=None):
    return np.array([t.score for t in trust_timeline(emails)])


def report(name, y, score, thr=0.5, atk=None):
    pred = (score >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    auc = roc_auc_score(y, score) if len(set(y)) > 1 else float("nan")
    print(f"{name:16s} acc={(tp+tn)/len(y):.3f} P={precision_score(y,pred,zero_division=0):.3f} "
          f"R={recall_score(y,pred):.3f} F1={f1_score(y,pred):.3f} FPR={fp/max(fp+tn,1):.3f} AUC={auc:.3f}")
    if atk is not None:
        per = {a: round(float(pred[atk == a].mean()), 2) for a in ("spoof", "takeover", "lookalike", "tamper")
               if (atk == a).any()}
        print(f"{'':16s} recall per attack: {per}")


def main():
    emails, sp, dns = load()
    text = [e["headers"]["subject"] + " " + e["body"] for e in emails]
    y = np.array([e["label"] for e in emails])
    atk = np.array([e["attack_type"] for e in emails])
    tr, te = slice(*sp["train"]), slice(*sp["test"])

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    Xtr = vec.fit_transform(text[tr.start:tr.stop])
    clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xtr, y[tr])
    content = clf.predict_proba(vec.transform(text[te.start:te.stop]))[:, 1]

    auth = 1 - trust_scores(emails, sp, dns)[te]          # high = suspicious
    combo = np.maximum(content, auth)

    print(f"test emails: {te.stop - te.start}, attack rate: {y[te].mean():.2f}")
    report("A content-only", y[te], content, atk=atk[te])
    report("B auth-only", y[te], auth, thr=0.2, atk=atk[te])
    report("C simple combo", y[te], combo, atk=atk[te])


if __name__ == "__main__":
    main()
