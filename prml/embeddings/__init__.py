from .bow import BoWEmbedder
from .tfidf import TfidfEmbedder
from .hashing import HashingEmbedder
from .distilbert_emb import DistilBertEmbedder
from .e5_emb import E5Embedder

EMBEDDERS = {
    "bow": BoWEmbedder,
    "tfidf": TfidfEmbedder,
    "hashing": HashingEmbedder,
    "distilbert": DistilBertEmbedder,
    "e5": E5Embedder,
}
