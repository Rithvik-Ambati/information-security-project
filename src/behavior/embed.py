"""Transformer text features: mean-pooled embeddings from a Hugging Face encoder (default DistilBERT).

Weights are downloaded by Hugging Face on first use (~270 MB for distilbert-base-uncased).
Set local_only=True to use only already-cached weights (e.g. BAAI/bge-small-en-v1.5)."""
import hashlib
import os
import pickle

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

DEFAULT_MODEL = "distilbert-base-uncased"


class TransformerEmbedder(BaseEstimator, TransformerMixin):
    def __init__(self, model_name=DEFAULT_MODEL, local_only=False, max_len=128, batch=32):
        self.model_name, self.local_only, self.max_len, self.batch = model_name, local_only, max_len, batch

    def _load(self):
        if getattr(self, "_m", None) is None:
            import torch
            from transformers import AutoModel, AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(self.model_name, local_files_only=self.local_only)
            self._m = AutoModel.from_pretrained(self.model_name, local_files_only=self.local_only).eval()
            self._torch = torch
            self._path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw",
                                      "_emb_" + self.model_name.replace("/", "_") + f"_{self.max_len}.pkl")
            try:
                self._cache = pickle.load(open(self._path, "rb"))
            except Exception:
                self._cache = {}

    def fit(self, X, y=None):
        return self

    def transform(self, texts):
        self._load()
        texts = list(texts)
        keys = [hashlib.md5(t.encode()).hexdigest() for t in texts]
        todo = [i for i, k in enumerate(keys) if k not in self._cache]
        for s in range(0, len(todo), self.batch):
            ids = todo[s:s + self.batch]
            enc = self._tok([texts[i] for i in ids], padding=True, truncation=True,
                            max_length=self.max_len, return_tensors="pt")
            with self._torch.no_grad():
                h = self._m(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1)
            emb = ((h * mask).sum(1) / mask.sum(1)).numpy()
            for i, v in zip(ids, emb):
                self._cache[keys[i]] = v
        if todo:  # persist so later runs/seeds reuse embeddings (the slow CPU step)
            try:
                pickle.dump(self._cache, open(self._path, "wb"))
            except Exception:
                pass
        return np.stack([self._cache[k] for k in keys])
