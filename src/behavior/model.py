"""Risk model: gradient boosting on behavior features + out-of-fold text score + IsolationForest anomaly."""
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from behavior.embed import TransformerEmbedder

# n_prior, hour_typical, len_ratio, new_recipient, prior_urgent_rate, odd_hour, payment/url novelty, rcpt domain, caps dev
PROFILE_COLS = [1, 2, 3, 4, 5, 15, 16, 17, 20, 21, 22, 23, 24, 28]


class RiskModel:
    def __init__(self, backend="tfidf", model_name=None, local_only=False):
        """backend: 'tfidf' or 'transformer' (DistilBERT by default)."""
        self.backend, self.model_name, self.local_only = backend, model_name, local_only

    def _text_pipeline(self):
        lr = LogisticRegression(max_iter=2000, class_weight="balanced")
        if self.backend == "transformer":
            kw = {"model_name": self.model_name} if self.model_name else {}
            return make_pipeline(TransformerEmbedder(local_only=self.local_only, **kw), StandardScaler(), lr)
        return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), lr)

    def fit(self, X, text, y, hist_mask):
        """hist_mask: legit rows with enough history, used to fit the unsupervised anomaly detector."""
        self.text = self._text_pipeline()
        oof = cross_val_predict(self.text, text, y, cv=5, method="predict_proba")[:, 1]
        self.text.fit(text, y)
        self.iso = IsolationForest(n_estimators=200, random_state=0).fit(X[hist_mask][:, PROFILE_COLS])
        self.gb = GradientBoostingClassifier(random_state=0).fit(self._aug(X, oof), y)
        return self

    def _anom(self, X):
        return -self.iso.score_samples(X[:, PROFILE_COLS])  # higher = more out-of-character

    def _aug(self, X, t):
        return np.column_stack([X, t, self._anom(X)])

    def risk(self, X, text):
        return self.gb.predict_proba(self._aug(X, self.text.predict_proba(text)[:, 1]))[:, 1]
