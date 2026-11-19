"""Parse raw corpora (maildir-style folders, mbox files) into normalized records."""
import email
import mailbox
import os
import re
from email import policy
from email.utils import parseaddr, parsedate_to_datetime


def _body(msg) -> str:
    try:
        part = msg.get_body(preferencelist=("plain", "html"))
        text = part.get_content() if part else ""
        if part is not None and part.get_content_type() == "text/html":
            text = re.sub(r"<(script|style).*?</>|<[^>]+>", " ", str(text), flags=re.S | re.I)
    except Exception:
        text = ""
    return " ".join(str(text).split())[:5000]


def _to_record(msg, label, source):
    name, addr = parseaddr(str(msg.get("From", "")))
    addr = addr.lower()
    if "@" not in addr:
        return None
    try:
        ts = parsedate_to_datetime(str(msg.get("Date", ""))).replace(tzinfo=None)
    except Exception:
        ts = None
    body = _body(msg)
    if len(body) < 20:
        return None
    return {"from": addr, "display_name": name, "to": parseaddr(str(msg.get("To", "")))[1].lower(),
            "subject": str(msg.get("Subject", ""))[:200], "date": ts.isoformat() if ts else None,
            "body": body, "label": label, "source": source,
            "n_received": len(msg.get_all("Received", []))}


def load_dir(path, label, source, limit=None):
    """Folder of one-message-per-file emails (SpamAssassin, Enron maildir)."""
    n = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                with open(os.path.join(root, f), "rb") as fh:
                    msg = email.message_from_binary_file(fh, policy=policy.default)
            except Exception:
                continue
            r = _to_record(msg, label, source)
            if r:
                yield r
                n += 1
                if limit and n >= limit:
                    return


def load_mbox(path, label, source, limit=None):
    """mbox file (Nazario phishing corpus)."""
    n = 0
    for m in mailbox.mbox(path, factory=lambda f: email.message_from_binary_file(f, policy=policy.default)):
        r = _to_record(m, label, source)
        if r:
            yield r
            n += 1
            if limit and n >= limit:
                return


SENT_FOLDERS = {"sent", "sent_items", "_sent_mail"}


def load_enron_sent(root, per_user=150, min_year=1998, max_year=2003):
    """Enron maildir: each user's *sent* folders, so every sender has a real per-sender history.
    Caps messages per user and drops bogus dates."""
    maildir = os.path.join(root, "maildir")
    for user in sorted(os.listdir(maildir)):
        n = 0
        for folder in sorted(os.listdir(os.path.join(maildir, user))):
            if folder.lower() not in SENT_FOLDERS:
                continue
            for r in load_dir(os.path.join(maildir, user, folder), 0, "enron"):
                if r["date"] and min_year <= int(r["date"][:4]) <= max_year:
                    yield r
                    n += 1
                    if n >= per_user:
                        break
            if n >= per_user:
                break
