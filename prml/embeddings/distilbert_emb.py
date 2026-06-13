import numpy as np
import torch
from .base import BaseEmbedder

_tokenizer = None
_model = None


def _load():
    global _tokenizer, _model
    if _model is None:
        from transformers import DistilBertModel, DistilBertTokenizer
        _tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        _model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        _model.eval()
    return _tokenizer, _model


def _mean_pool(text: str) -> np.ndarray:
    """Tokenize, run DistilBERT, mean-pool over tokens → 768-dim vector."""
    tokenizer, model = _load()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    pooled = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
    return pooled.squeeze(0).numpy()


class DistilBertEmbedder(BaseEmbedder):
    """768-dim dense embeddings from DistilBERT"""

    def transform(self, text: str) -> dict[str, float]:
        vec = _mean_pool(text)
        return {f"d_{i}": float(v) for i, v in enumerate(vec)}

    def transform_batch(self, texts: list[str]) -> list[dict[str, float]]:
        tokenizer, model = _load()
        inputs = tokenizer(texts, return_tensors="pt", truncation=True,
                           max_length=512, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        pooled = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
        vecs = pooled.numpy()
        return [
            {f"d_{i}": float(v) for i, v in enumerate(vec)}
            for vec in vecs
        ]
