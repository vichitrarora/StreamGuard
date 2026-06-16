"""CLI inference — type a sentence, get toxicity predictions.
Usage:
    python cli.py                              # interactive mode (bow)
    python cli.py --embedding tfidf            # use TF-IDF embedder
    python cli.py --text "you are an idiot"    # single prediction, then exit
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from embeddings import EMBEDDERS
from models.ensemble import MultiLabelEnsemble, LABELS

MODEL_DIR = Path(__file__).resolve().parent.parent / "saved_models"


def load_ensemble(embedding_name: str) -> MultiLabelEnsemble:
    model_path = MODEL_DIR / f"ensemble_{embedding_name}.pkl"
    if not model_path.exists():
        print(f"Error: No saved model found at {model_path}")
        print(f"Train first via: python consumer.py --embedding {embedding_name}")
        sys.exit(1)
    return MultiLabelEnsemble.from_file(model_path)


def predict(ensemble: MultiLabelEnsemble, embedder, text: str):
    x = embedder.transform(text)
    preds = ensemble.predict_one(x)
    probas = ensemble.predict_proba_one(x)

    print(f"\n  Input : {text}")
    print(f"  {'─'*50}")
    flagged = []
    for label in LABELS:
        marker = "██" if preds[label] == 1 else "  "
        print(f"  {marker} {label:20s}  pred={preds[label]}  prob={probas[label]:.3f}")
        if preds[label] == 1:
            flagged.append(label)

    if flagged:
        print(f"\n  ⚠  Flagged: {', '.join(flagged)}")
    else:
        print(f"\n  ✔  Not toxic")
    print()


def main():
    parser = argparse.ArgumentParser(description="CLI toxicity inference")
    parser.add_argument("--embedding", default="bow",
                        choices=list(EMBEDDERS.keys()),
                        help="Embedding type (default: bow)")
    parser.add_argument("--text", type=str, default=None,
                        help="Single sentence to classify (skips interactive mode)")
    args = parser.parse_args()

    print(f"Loading ensemble for '{args.embedding}' …")
    ensemble = load_ensemble(args.embedding)
    embedder = EMBEDDERS[args.embedding]()

    if args.text:
        predict(ensemble, embedder, args.text)
        return

    print("Interactive mode — type a sentence and press Enter. Ctrl+C to quit.\n")
    try:
        while True:
            text = input(">>> ").strip()
            if not text:
                continue
            predict(ensemble, embedder, text)
    except (KeyboardInterrupt, EOFError):
        print("\nBye!")


if __name__ == "__main__":
    main()
