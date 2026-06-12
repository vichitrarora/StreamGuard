"""Bag-of-Words embedder — simple term-frequency counts."""

from .base import BaseEmbedder


class BoWEmbedder(BaseEmbedder):
    """Each unique token becomes a feature with its count as the value."""

    def transform(self, text: str) -> dict[str, float]:
        tokens = self._tokenize(text)
        counts: dict[str, float] = {}
        for tok in tokens:
            counts[tok] = counts.get(tok, 0.0) + 1.0
        return counts
