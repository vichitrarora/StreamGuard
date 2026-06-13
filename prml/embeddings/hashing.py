import hashlib
from .base import BaseEmbedder


class HashingEmbedder(BaseEmbedder):
    """Maps tokens to a fixed number of buckets via hashing"""

    def __init__(self, n_features: int = 2**14):
        self.n_features = n_features

    def transform(self, text: str) -> dict[str, float]:
        tokens = self._tokenize(text)
        features: dict[str, float] = {}
        for tok in tokens:
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            idx = h % self.n_features
            sign = 1.0 if (h // self.n_features) % 2 == 0 else -1.0
            key = f"h_{idx}"
            features[key] = features.get(key, 0.0) + sign
        return features
