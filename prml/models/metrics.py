from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import numpy as np

from .ensemble import LABELS


class MetricsTracker:
    """Tracks per-label TP/FP/FN/TN and rolling accuracy over a stream."""

    def __init__(self):
        self.n = 0
        self.tp = {l: 0 for l in LABELS}
        self.fp = {l: 0 for l in LABELS}
        self.fn = {l: 0 for l in LABELS}
        self.tn = {l: 0 for l in LABELS}
        self.history_n: list[int] = []
        self.history_acc: list[float] = []
        self.history_f1: list[float] = []
        self._correct = 0
        self._total_labels = 0

    def update(self, y_true: dict[str, int], y_pred: dict[str, int]):
        self.n += 1
        for label in LABELS:
            t = y_true.get(label, 0)
            p = y_pred.get(label, 0)
            if t == 1 and p == 1:
                self.tp[label] += 1
            elif t == 0 and p == 1:
                self.fp[label] += 1
            elif t == 1 and p == 0:
                self.fn[label] += 1
            else:
                self.tn[label] += 1
            self._correct += int(t == p)
            self._total_labels += 1

        if self.n % 500 == 0:
            self.history_n.append(self.n)
            self.history_acc.append(self.accuracy())
            self.history_f1.append(self.macro_f1())

    def accuracy(self) -> float:
        return self._correct / max(self._total_labels, 1)

    def label_f1(self, label: str) -> float:
        tp = self.tp[label]
        fp = self.fp[label]
        fn = self.fn[label]
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def macro_f1(self) -> float:
        return np.mean([self.label_f1(l) for l in LABELS])

    def summary(self) -> dict:
        return {
            "samples": self.n,
            "accuracy": round(self.accuracy(), 4),
            "macro_f1": round(self.macro_f1(), 4),
            "per_label_f1": {l: round(self.label_f1(l), 4) for l in LABELS},
        }

    def print_summary(self, embedding_name: str = ""):
        s = self.summary()
        tag = f" [{embedding_name}]" if embedding_name else ""
        print(f"\n{'='*55}")
        print(f"  Metrics Summary{tag}  ({s['samples']} samples)")
        print(f"{'─'*55}")
        print(f"  Accuracy : {s['accuracy']:.4f}")
        print(f"  Macro-F1 : {s['macro_f1']:.4f}")
        print(f"{'─'*55}")
        for l in LABELS:
            print(f"    {l:20s}  F1 = {s['per_label_f1'][l]:.4f}")
        print(f"{'='*55}\n")

    def plot_rolling(self, out_dir: str | Path, tag: str = ""):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if not self.history_n:
            return

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.plot(self.history_n, self.history_acc, label="Accuracy")
        ax.plot(self.history_n, self.history_f1, label="Macro-F1")
        ax.set_xlabel("Samples seen")
        ax.set_ylabel("Score")
        ax.set_title(f"Rolling Metrics Over Stream{f' ({tag})' if tag else ''}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fname = out_dir / f"rolling_{tag or 'default'}.png"
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"  Plot saved → {fname}")

    def plot_per_label_f1(self, out_dir: str | Path, tag: str = ""):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        f1s = [self.label_f1(l) for l in LABELS]
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.barh(LABELS, f1s, color="steelblue")
        ax.set_xlim(0, 1)
        ax.set_xlabel("F1 Score")
        ax.set_title(f"Per-Label F1{f' ({tag})' if tag else ''}")
        for bar, val in zip(bars, f1s):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=9)
        fig.tight_layout()
        fname = out_dir / f"f1_per_label_{tag or 'default'}.png"
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"  Plot saved → {fname}")

    def plot_confusion(self, out_dir: str | Path, tag: str = ""):
        """Plot a small confusion matrix per label."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        for ax, label in zip(axes.flat, LABELS):
            cm = np.array([
                [self.tn[label], self.fp[label]],
                [self.fn[label], self.tp[label]],
            ])
            ax.imshow(cm, cmap="Blues")
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                            fontsize=12, color="white" if cm[i, j] > cm.max() / 2 else "black")
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Pred 0", "Pred 1"])
            ax.set_yticklabels(["True 0", "True 1"])
            ax.set_title(label)
        fig.suptitle(f"Confusion Matrices{f' ({tag})' if tag else ''}", fontsize=13)
        fig.tight_layout()
        fname = out_dir / f"confusion_{tag or 'default'}.png"
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"  Plot saved → {fname}")

    def save_json(self, out_dir: str | Path, tag: str = ""):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = out_dir / f"metrics_{tag or 'default'}.json"
        with open(fname, "w") as f:
            json.dump(self.summary(), f, indent=2)
        print(f"  Metrics JSON → {fname}")


def plot_embedding_comparison(trackers: dict[str, MetricsTracker], out_dir: str | Path):
    """Grouped bar chart comparing macro-F1 and accuracy across embeddings."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    names = list(trackers.keys())
    accs = [t.accuracy() for t in trackers.values()]
    f1s = [t.macro_f1() for t in trackers.values()]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, accs, width, label="Accuracy", color="steelblue")
    ax.bar(x + width / 2, f1s, width, label="Macro-F1", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Embedding Comparison — Accuracy & Macro-F1")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fname = out_dir / "embedding_comparison.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Comparison plot saved → {fname}")
