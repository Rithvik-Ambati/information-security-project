"""Download public corpora into data/raw/. Run this yourself:  python scripts/download_data.py [spamassassin|nazario|enron]

Sizes (approx.): SpamAssassin ~10 MB, Nazario ~15-30 MB per year file, Enron ~420 MB.
Check the URLs still resolve; hosts occasionally move files. Each corpus has its own terms of use.
"""
import os
import sys
import tarfile
import urllib.request

RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
SOURCES = {
    "spamassassin": [  # legitimate ("ham") + spam
        "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham.tar.bz2",
        "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham_2.tar.bz2",
        "https://spamassassin.apache.org/old/publiccorpus/20050311_spam_2.tar.bz2",
    ],
    "nazario": [  # phishing mbox files (José Nazario); save as-is
        "https://monkey.org/~jose/phishing/phishing-2015",
        "https://monkey.org/~jose/phishing/phishing3.mbox",
    ],
    "enron": ["https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz"],
}


def fetch(url, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    out = os.path.join(dest_dir, url.rsplit("/", 1)[-1])
    print("downloading", url)
    urllib.request.urlretrieve(url, out)
    if out.endswith((".tar.bz2", ".tar.gz")):
        with tarfile.open(out) as t:
            t.extractall(dest_dir, filter="data")


if __name__ == "__main__":
    for name in (sys.argv[1:] or SOURCES):
        for u in SOURCES[name]:
            try:
                fetch(u, os.path.join(RAW, name))
            except Exception as e:
                print("  failed:", e)
