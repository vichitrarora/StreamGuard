from abc import ABC, abstractmethod
import re


class BaseEmbedder(ABC):
    """All embedders convert a raw text string into a dict[str, float]
    compatible with river's online learners."""

    @abstractmethod
    def transform(self, text: str) -> dict[str, float]:
        ...

    def transform_batch(self, texts: list[str]) -> list[dict[str, float]]:
        return [self.transform(t) for t in texts]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower())
