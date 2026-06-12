import math
from .base import BaseEmbedder


class TfidfEmbedder(BaseEmbedder):
    """Streaming TF-IDF: tracks DF counts and total docs seen so far."""

    def __init__(self):
        self.doc_count: int = 0
        self.df: dict[str, int] = {}  

    def transform(self, text: str) -> dict[str, float]:
        tokens = self._tokenize(text)
        if not tokens:
            return {}

        self.doc_count += 1
        unique_tokens = set(tokens)
        for tok in unique_tokens:
            self.df[tok] = self.df.get(tok, 0) + 1

        tf: dict[str, float] = {}
        for tok in tokens:
            tf[tok] = tf.get(tok, 0.0) + 1.0
        doc_len = len(tokens)

        result: dict[str, float] = {}
        for tok, count in tf.items():
            idf = math.log((1 + self.doc_count) / (1 + self.df.get(tok, 0))) + 1
            result[tok] = (count / doc_len) * idf
        return result
