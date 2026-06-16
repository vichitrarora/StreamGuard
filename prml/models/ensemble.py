"""Multi-label voting ensemble using river online learners.

Models per label (3 diverse learners):
  1. Adaptive Random Forest (ARF) — tree ensemble
  2. Hoeffding Adaptive Tree (HAT) — drift-adaptive tree
  3. Logistic Regression (Adam) — linear baseline

"""

from __future__ import annotations

import pickle
from pathlib import Path

from river import forest, linear_model, optim, tree

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def _make_models():
    """Return a dict of {model_name: river_classifier} for a single label."""
    return {
        "arf": forest.ARFClassifier(n_models=3, seed=42),
        "hat": tree.HoeffdingAdaptiveTreeClassifier(
            grace_period=50, delta=1e-5, leaf_prediction="nba",
        ),
        "lr": linear_model.LogisticRegression(
            optimizer=optim.Adam(lr=0.01), l2=0.0001,
        ),
    }


DEFAULT_THRESHOLD = 0.5
MAX_NEG_RATIO = 3  


class MultiLabelEnsemble:
    """One voting ensemble per label — 3 models × 6 labels = 18 models."""

    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        self.models: dict[str, dict] = {
            label: _make_models() for label in LABELS
        }
        self.threshold = threshold
        self._pos: dict[str, int] = {label: 0 for label in LABELS}
        self._neg: dict[str, int] = {label: 0 for label in LABELS}

    def learn_one(self, x: dict[str, float], y: dict[str, int]):
        if not x:
            return
        for label in LABELS:
            label_val = y.get(label, 0)
            if label_val == 1:
                self._pos[label] += 1
            else:
                if self._neg[label] >= MAX_NEG_RATIO * max(self._pos[label], 1):
                    continue
                self._neg[label] += 1
            for model in self.models[label].values():
                try:
                    model.learn_one(x, label_val)
                except ValueError:
                    pass 

    def predict_one(self, x: dict[str, float]) -> dict[str, int]:
        """Probability-threshold prediction for all 6 labels.

        Averages predicted P(y=1) across models; labels with
        avg probability >= self.threshold are predicted as 1.
        """
        probas = self.predict_proba_one(x)
        return {label: int(p >= self.threshold) for label, p in probas.items()}

    def predict_proba_one(self, x: dict[str, float]) -> dict[str, float]:
        """Average predicted probability across the models per label."""
        if not x:
            return {label: 0.0 for label in LABELS}
        probas: dict[str, float] = {}
        for label in LABELS:
            p_sum = 0.0
            n = 0
            for model in self.models[label].values():
                try:
                    dist = model.predict_proba_one(x)
                    p_sum += dist.get(1, 0.0)
                    n += 1
                except Exception:
                    pass
            probas[label] = p_sum / max(n, 1)
        return probas

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.models, f)
        print(f"Ensemble saved → {path}")

    def load(self, path: str | Path):
        with open(path, "rb") as f:
            self.models = pickle.load(f)
        print(f"Ensemble loaded ← {path}")

    @classmethod
    def from_file(cls, path: str | Path) -> "MultiLabelEnsemble":
        ens = cls.__new__(cls)
        ens.load(path)
        return ens
