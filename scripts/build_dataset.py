"""Build the evaluation dataset.

  python scripts/build_dataset.py --demo          # tiny synthetic corpus, no downloads
  python scripts/build_dataset.py                  # uses data/raw/{enron,spamassassin,nazario}
  python scripts/build_dataset.py --seed 7 --out data/processed_seed7

Output (<out>/): emails.jsonl (time-ordered), registry.json, splits.json (history/train/test by time), stats.txt

Realism measures: near-duplicate phishing removed; train/test phishing pools disjoint; BEC templates held out
by template; SpamAssassin ham adds unseen-sender legitimate mail; ~8% of legit mail is unsigned; some domains
rotate DKIM keys; only half of attacks use unusual send times.
"""
import argparse
import hashlib
import json
import os
import pickle
import random
import sys
from datetime import datetime, timedelta

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from data.attacks import Signer
from data.bec import bec_email
from data.parse import load_dir, load_enron_sent, load_mbox

RAW = os.path.join(ROOT, "data", "raw")
DEFAULT_OUT = os.path.join(ROOT, "data", "processed")
SUBTLE = []
ATTACKS = ["spoof", "takeover", "lookalike", "tamper"]

def dedupe_near(bodies, thr=0.7):
    """Greedy near-duplicate removal (TF-IDF cosine >= thr) so test text is not a paraphrase of train text."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    if len(bodies) < 2:
        return bodies
    X = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True).fit_transform(bodies)
    kept = []
    for i in range(len(bodies)):
        if not kept or (X[i] @ X[kept].T).max() < thr:
            kept.append(i)
    return [bodies[i] for i in kept]


def demo_corpus(rng):
    users = [f"{n}@{d}" for n in ("alice", "bob", "carol", "dave") for d in ("corp.com", "acme.org")]
    ham_t = ["Please review the attached report before the {d} meeting.", "Thanks for the update on the {d} project.",
             "Can we reschedule our {d} sync to later this week?", "Notes from {d} are in the shared folder."]
    phish_t = ["URGENT: your account will be suspended, verify at http://login-{d}.example now.",
               "Please wire the {d} invoice payment today to the new bank account and keep this confidential.",
               "Your password expires immediately. Click http://reset-{d}.example to avoid losing access."]
    ham = [{"from": rng.choice(users), "to": "me@corp.com", "subject": f"Update {i}", "n_received": rng.randint(1, 4),
            "date": f"2012-{1 + i % 12:02d}-{1 + i % 27:02d}T{rng.randint(9, 17):02d}:00:00",
            "body": rng.choice(ham_t).format(d=rng.choice(["Q1", "budget", "design", "hiring"])) + f" ref{i}",
            "label": 0, "source": "demo"} for i in range(600)]
    phish = [{"body": rng.choice(phish_t).format(d=rng.choice(["bank", "payroll", "cloud"])) + f" ref{i}"}
             for i in range(150)]
    subtle_t = ["Hi, could you send me the updated {d} account details when you get a chance? Thanks.",
                "Quick favour: please share the {d} login for the vendor portal, I am locked out.",
                "Following up on the {d} transfer we discussed, what are the new details?"]
    global SUBTLE
    SUBTLE = [rng.choice(subtle_t).format(d=rng.choice(["vendor", "payroll", "supplier"])) + f" ref{i}"
              for i in range(150)]
    return ham, phish


def load_real(max_ham, max_phish, max_sa):
    cache = os.path.join(RAW, f"_parsed_cache_{max_ham}_{max_phish}.pkl")
    if os.path.exists(cache):
        return pickle.load(open(cache, "rb"))
    enron = list(load_enron_sent(os.path.join(RAW, "enron"), per_user=max(1, max_ham // 150)))
    sa = []
    for d in ("easy_ham", "easy_ham_2"):
        p = os.path.join(RAW, "spamassassin", d)
        if os.path.isdir(p):
            sa += list(load_dir(p, 0, "spamassassin"))
    phish = []
    nz = os.path.join(RAW, "nazario")
    if os.path.isdir(nz):
        for f in os.listdir(nz):
            p = os.path.join(nz, f)
            if os.path.isfile(p) and not f.endswith((".bz2", ".gz")):
                try:
                    phish += list(load_mbox(p, 1, "nazario", max_phish))
                except OSError as ex:  # e.g. antivirus quarantined the file
                    print(f"skipping {f}: {ex}")
    if not enron or not phish:
        sys.exit("Missing data in data/raw/. Run scripts/download_data.py or use --demo.")
    pickle.dump((enron, sa, phish), open(cache, "wb"))
    return enron, sa, phish


def main(demo, max_ham, max_phish, seed, out_dir, max_sa):
    rng = random.Random(seed)
    if demo:
        ham, phish = demo_corpus(rng)
    else:
        enron, sa, phish = load_real(max_ham, max_phish, max_sa)
        ham = list(enron)
        # SpamAssassin ham = legitimate mail from many unseen external senders (hard negatives for FPR).
        # Its native dates (2002) would cluster at the end, so spread them across the Enron span (documented).
        ds = sorted(r["date"] for r in enron)
        lo, hi = datetime.fromisoformat(ds[0]), datetime.fromisoformat(ds[-1])
        rs = random.Random(seed + 1)
        sa = list(sa)
        rs.shuffle(sa)
        for r in sa[:max_sa]:
            t = lo + timedelta(seconds=rs.random() * (hi - lo).total_seconds())
            ham.append(dict(r, date=t.replace(hour=rs.randint(7, 19)).isoformat(), source="spamassassin"))

    seen, clean = set(), []
    for r in ham:
        h = hashlib.md5(r["body"].encode()).hexdigest()
        if h not in seen and r["date"]:
            seen.add(h)
            clean.append(r)
    ham = sorted(clean, key=lambda r: r["date"])
    uniq = sorted({p["body"] for p in phish})
    n_before = len(uniq)
    rng.shuffle(uniq)
    phish_bodies = uniq if demo else dedupe_near(uniq)
    # three disjoint eras matching the evaluation windows: fit [0, .55), validation [.55, .7), test [.7, 1]
    c1, c2 = int(0.55 * len(phish_bodies)), int(0.7 * len(phish_bodies))
    pools = {"train": phish_bodies[:c1] or phish_bodies, "val": phish_bodies[c1:c2] or phish_bodies,
             "test": phish_bodies[c2:] or phish_bodies}

    signer, out = Signer(seed=seed + 13), []
    # a few domains rotate their DKIM key/selector mid-timeline (legit mail that breaks key continuity)
    doms = sorted({r["from"].split("@")[1] for r in ham})
    d0 = datetime.fromisoformat(ham[0]["date"])
    span = (datetime.fromisoformat(ham[-1]["date"]) - d0).total_seconds()
    for dom in doms:
        if dom == "enron.com" or rng.random() < 0.15:
            frac = 0.8 if dom == "enron.com" else rng.uniform(0.3, 0.9)
            signer.set_rotation(dom, (d0 + timedelta(seconds=frac * span)).isoformat())

    for i, r in enumerate(ham):
        out.append(signer.make(r, "legit"))
        if rng.random() < 0.25:  # ~25% of legit volume gets an attack variant, dated like the victim mail
            atk = rng.choice(ATTACKS)
            era = "test" if i >= 0.7 * len(ham) else ("val" if i >= 0.55 * len(ham) else "train")
            if demo:
                pool = SUBTLE if (i > 0.6 * len(ham) and rng.random() < 0.8) else phish_bodies
                body = rng.choice(pool)
            elif atk == "takeover" and rng.random() < 0.6:
                body = bec_email(rng, era)   # combinatorial BEC grammar; the 'ask' phrases are disjoint per era
            else:  # mass-phishing text: near-dup-free, era pools disjoint
                body = rng.choice(pools[era])
            out.append(signer.make(r, atk, body))
    out.sort(key=lambda e: e["date"])

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "emails.jsonl"), "w") as f:
        for i, e in enumerate(out):
            e["id"] = i
            f.write(json.dumps(e) + "\n")
    json.dump(signer.registry_dump(), open(os.path.join(out_dir, "registry.json"), "w"), indent=1)

    n = len(out)  # time-ordered: first 40% = sender history, next 30% = train, last 30% = test (drift)
    splits = {"history": [0, int(.4 * n)], "train": [int(.4 * n), int(.7 * n)], "test": [int(.7 * n), n]}
    json.dump(splits, open(os.path.join(out_dir, "splits.json"), "w"))
    counts = {}
    for e in out:
        counts[e["attack_type"]] = counts.get(e["attack_type"], 0) + 1
    unsigned_legit = sum(1 for e in out if e["label"] == 0 and e["sig"] is None)
    stats = (f"seed={seed}\ntotal={n}\n" + "\n".join(f"{k}={v}" for k, v in counts.items())
             + f"\nunsigned_legit={unsigned_legit}\nphish_unique_before_neardup={n_before} after={len(phish_bodies)}"
             + f"\nkey_rotations={len(signer.rotation)}\nsplits={splits}\n")
    open(os.path.join(out_dir, "stats.txt"), "w").write(stats)
    print(stats)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--max-ham", type=int, default=20000)
    ap.add_argument("--max-phish", type=int, default=5000)
    ap.add_argument("--max-sa", type=int, default=3000, help="SpamAssassin ham emails added as unseen-sender legit mail")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args()
    main(a.demo, a.max_ham, a.max_phish, a.seed, a.out, a.max_sa)
