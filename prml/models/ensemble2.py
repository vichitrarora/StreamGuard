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

DEFAULT_THRESHOLD = 0.5
MAX_NEG_RATIO = 3
_EMA_ALPHA = 0.05  


def _make_models():
    """Return a dict of {model_name: river_classifier} for a single label."""
    return {
        "arf": forest.ARFClassifier(
            n_models=10,
            max_features="sqrt",
            lambda_value=6,
            drift_detector=None,      
            warning_detector=None,
            seed=42,
        ),
        "hat": tree.HoeffdingAdaptiveTreeClassifier(
            grace_period=200,
            delta=1e-7,
            leaf_prediction="mc",
            bootstrap_sampling=True,
            drift_window_threshold=300,
        ),
        "lr": linear_model.LogisticRegression(
            optimizer=optim.Adam(lr=0.01), l2=0.0001,
        ),
    }


def _make_weights() -> dict[str, float]:
    """Each model starts with equal weight; updated online from prediction error."""
    return {"arf": 1.0, "hat": 1.0, "lr": 1.0}


class MultiLabelEnsemble:
    """One voting ensemble per label — 3 models × 6 labels = 18 models."""

    def __init__(self, threshold: float = DEFAULT_THRESHOLD, max_neg_ratio: int = MAX_NEG_RATIO):
        self.models: dict[str, dict] = {
            label: _make_models() for label in LABELS
        }
        self._weights: dict[str, dict[str, float]] = {
            label: _make_weights() for label in LABELS
        }
        self._ema_error: dict[str, dict[str, float]] = {
            label: {"arf": 0.5, "hat": 0.5, "lr": 0.5} for label in LABELS
        }
        self.threshold = threshold
        self.max_neg_ratio = max_neg_ratio
        self._pos: dict[str, int] = {label: 0 for label in LABELS}
        self._neg: dict[str, int] = {label: 0 for label in LABELS}

    def _update_weights(self, label: str, name: str, true_val: int, pred_proba: float):
        """Recompute one model's weight from its rolling prediction error.

        EMA smooths over recent mistakes so a single bad example doesn't
        instantly tank a model that's been performing well.
        """
        error = abs(true_val - pred_proba)
        prev = self._ema_error[label][name]
        self._ema_error[label][name] = _EMA_ALPHA * error + (1 - _EMA_ALPHA) * prev
        self._weights[label][name] = 1.0 / max(self._ema_error[label][name], 1e-6)

    def learn_one(self, x: dict[str, float], y: dict[str, int]):
        if not x:
            return
        for label in LABELS:
            label_val = y.get(label, 0)
            if label_val == 1:
                self._pos[label] += 1
            else:
                if self._neg[label] >= self.max_neg_ratio * max(self._pos[label], 1):
                    continue
                self._neg[label] += 1
            for name, model in self.models[label].items():
                try:
                    dist = model.predict_proba_one(x)
                    pred_proba = dist.get(1, 0.0)
                    self._update_weights(label, name, label_val, pred_proba)
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
            weighted_sum = 0.0
            total_weight = 0.0
            for name, model in self.models[label].items():
                try:
                    dist = model.predict_proba_one(x)
                    w = self._weights[label][name]
                    weighted_sum += w * dist.get(1, 0.0)
                    total_weight += w
                except Exception:
                    pass
            probas[label] = weighted_sum / max(total_weight, 1e-9)
        return probas

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "models": self.models,
            "weights": self._weights,
            "ema_error": self._ema_error,
            "pos": self._pos,
            "neg": self._neg,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        print(f"Ensemble saved → {path}")

    def load(self, path: str | Path):
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.models = state["models"]
        self._weights = state["weights"]
        self._ema_error = state["ema_error"]
        self._pos = state["pos"]
        self._neg = state["neg"]
        print(f"Ensemble loaded ← {path}")

    @classmethod
    def from_file(cls, path: str | Path, threshold: float = DEFAULT_THRESHOLD) -> "MultiLabelEnsemble":
        ens = cls(threshold=threshold)
        ens.load(path)
        return ens