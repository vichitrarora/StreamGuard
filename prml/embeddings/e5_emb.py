"""E5-base-v2 sentence embedding via sentence-transformers (GPU-friendly).

Microsoft's E5 model expects input prefixed with "query: " for best results.
Outputs 768-dim dense vectors.
"""

from .base import BaseEmbedder

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("intfloat/e5-base-v2")
    return _model


class E5Embedder(BaseEmbedder):
    """768-dim dense embeddings from E5-base-v2."""

    def transform(self, text: str) -> dict[str, float]:
        model = _get_model()
        vec = model.encode(f"query: {text}", show_progress_bar=False)
        return {f"e5_{i}": float(v) for i, v in enumerate(vec)}

    def transform_batch(self, texts: list[str]) -> list[dict[str, float]]:
        model = _get_model()
        prefixed = [f"query: {t}" for t in texts]
        vecs = model.encode(prefixed, show_progress_bar=False, batch_size=64)
        return [
            {f"e5_{i}": float(v) for i, v in enumerate(vec)}
            for vec in vecs
        ]
