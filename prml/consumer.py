"""Kafka consumer — trains the ensemble on streamed data with prequential evaluation.

Usage:
    python consumer.py --embedding bow       # use bag-of-words
    python consumer.py --embedding tfidf     # use online TF-IDF
    python consumer.py --embedding all       # run ALL embeddings simultaneously
"""

import argparse
import json
import sys
from pathlib import Path

from confluent_kafka import Consumer, KafkaError, KafkaException
from config import get_consumer_config

sys.path.insert(0, str(Path(__file__).resolve().parent))

from embeddings import EMBEDDERS
from models.ensemble import MultiLabelEnsemble, LABELS
from models.metrics import MetricsTracker, plot_embedding_comparison

TRAIN_TOPIC = "hate_detection_train"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
MODEL_DIR = Path(__file__).resolve().parent.parent / "saved_models"
THRESHOLD = 0.5    


def _consume_topic(topic: str, handler, group_suffix: str = "",
                   timeout_no_msg: int = 10):
    """Generic consume loop. Calls handler(row_dict) for each message.
    Stops after *timeout_no_msg* seconds with no new messages.
    """
    import time
    run_id = int(time.time())
    conf = get_consumer_config(
        group_id=f"hatedetection_{topic}_{group_suffix}_{run_id}"
    )
    conf["auto.offset.reset"] = "earliest"
    consumer = Consumer(conf)
    consumer.subscribe([topic])

    last_msg_time = time.time()
    count = 0

    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                if time.time() - last_msg_time > timeout_no_msg:
                    print(f"  No messages for {timeout_no_msg}s — stopping.")
                    break
                continue

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                raise KafkaException(msg.error())

            last_msg_time = time.time()
            row = json.loads(msg.value().decode("utf-8"))
            handler(row)
            count += 1

            if count % 1000 == 0:
                print(f"  [{topic}] Consumed {count:,} messages …")

        consumer.commit()
    finally:
        consumer.close()

    print(f"✔ [{topic}] Consumed {count:,} messages total.")
    return count


def run_pipeline(embedding_name: str):
    """Prequential train pipeline for one embedding type."""
    print(f"\n{'#'*60}")
    print(f"  Embedding: {embedding_name}")
    print(f"{'#'*60}")

    embedder = EMBEDDERS[embedding_name]()
    ensemble = MultiLabelEnsemble(threshold=THRESHOLD)
    tracker = MetricsTracker()

    print("\nTraining (prequential) …")
    train_count = 0

    def train_handler(row: dict):
        nonlocal train_count

        text = row.get("comment_text", "")
        y = {l: int(row.get(l, 0)) for l in LABELS}
        x = embedder.transform(text)

        y_pred = ensemble.predict_one(x)
        tracker.update(y, y_pred)
        ensemble.learn_one(x, y)

        train_count += 1
        if train_count % 1000 == 0:
            print(f"    [{embedding_name}] {train_count:,} rows  "
                  f"(acc={tracker.accuracy():.4f}  macro-f1={tracker.macro_f1():.4f})")

    _consume_topic(TRAIN_TOPIC, train_handler, group_suffix=embedding_name)

    tracker.print_summary(f"{embedding_name} — prequential")
    tag = f"{embedding_name}_train"
    tracker.plot_rolling(OUTPUT_DIR, tag)
    tracker.plot_per_label_f1(OUTPUT_DIR, tag)
    tracker.save_json(OUTPUT_DIR, tag)

    ensemble.save(MODEL_DIR / f"ensemble_{embedding_name}.pkl")

    return tracker


def run_all_pipelines(skip: list[str] | None = None):
    """Train ALL embeddings in a single pass over the stream.

    Consumes the topic only once and feeds every embedding/ensemble
    simultaneously.
    """
    skip = set(skip or [])
    active = {k: v for k, v in EMBEDDERS.items() if k not in skip}

    if skip:
        print(f"\nSkipping: {', '.join(skip)}")

    print(f"\n{'#'*60}")
    print(f"  Running embeddings: {', '.join(active.keys())}")
    print(f"{'#'*60}")

    pipelines: dict[str, dict] = {}
    for name, EmbCls in active.items():
        pipelines[name] = {
            "embedder": EmbCls(),
            "ensemble": MultiLabelEnsemble(threshold=THRESHOLD),
            "tracker": MetricsTracker(),
        }

    print("\nTraining (prequential, all embeddings) …")

    _all_count = 0

    def train_handler_all(row: dict):
        nonlocal _all_count
        text = row.get("comment_text", "")
        y = {l: int(row.get(l, 0)) for l in LABELS}

        for name, p in pipelines.items():
            x = p["embedder"].transform(text)
            y_pred = p["ensemble"].predict_one(x)
            p["tracker"].update(y, y_pred)
            p["ensemble"].learn_one(x, y)

        _all_count += 1
        if _all_count % 1000 == 0:
            stats = "  |  ".join(
                f"{n}: acc={p['tracker'].accuracy():.4f} f1={p['tracker'].macro_f1():.4f}"
                for n, p in pipelines.items()
            )
            print(f"    {_all_count:,} rows  {stats}")

    _consume_topic(TRAIN_TOPIC, train_handler_all, group_suffix="all_train")

    all_trackers: dict[str, MetricsTracker] = {}
    for name, p in pipelines.items():
        p["tracker"].print_summary(f"{name} — prequential")
        tag = f"{name}_train"
        p["tracker"].plot_rolling(OUTPUT_DIR, tag)
        p["tracker"].plot_per_label_f1(OUTPUT_DIR, tag)
        p["tracker"].save_json(OUTPUT_DIR, tag)
        p["ensemble"].save(MODEL_DIR / f"ensemble_{name}.pkl")
        all_trackers[name] = p["tracker"]

    plot_embedding_comparison(all_trackers, OUTPUT_DIR)
    print("\n✔ All embeddings trained and evaluated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consume Kafka stream, train ensemble (prequential)")
    parser.add_argument("--embedding", default="bow",
                        choices=list(EMBEDDERS.keys()) + ["all"],
                        help="Which embedding to use (default: bow)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Probability threshold for positive prediction (default: 0.5)")
    parser.add_argument("--skip", nargs="+", default=[],
                        choices=list(EMBEDDERS.keys()),
                        help="Embeddings to skip when using --embedding all (e.g. --skip distilbert e5)")
    args = parser.parse_args()

    THRESHOLD = args.threshold

    if args.embedding == "all":
        run_all_pipelines(skip=args.skip)
    else:
        run_pipeline(args.embedding)