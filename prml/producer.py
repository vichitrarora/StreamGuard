import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from confluent_kafka import Producer
from config import get_producer_config

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
TRAIN_CSV = DATASET_DIR / "train.csv"

TRAIN_TOPIC = "hate_detection_train"


def acked(err, msg):
    if err is not None:
        print(f"Failed to deliver message: {err}", file=sys.stderr)


def produce_csv(csv_path: Path, topic: str, labels_path: Path | None = None):
    """Stream rows from *csv_path* to Kafka *topic*.

    For test data, if *labels_path* is given the labels are merged into
    each message so the consumer can evaluate.
    """
    conf = get_producer_config()
    producer = Producer(conf)

    labels_map: dict[str, dict] = {}
    if labels_path:
        with open(labels_path, newline="", encoding="utf-8") as lf:
            for row in csv.DictReader(lf):
                if row.get("toxic") == "-1":
                    continue
                labels_map[row["id"]] = row

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        count = 0
        for i, row in enumerate(reader):
            if labels_path:
                lid = row.get("id", "")
                if lid not in labels_map:
                    continue  
                row.update(labels_map[lid])

            payload = json.dumps(row).encode("utf-8")
            producer.produce(topic, key=row.get("id", str(i)), value=payload, callback=acked)
            count += 1

            if count % 10_000 == 0:
                producer.flush()
                print(f"  [{topic}] Produced {count:,} messages …")

    producer.flush()
    print(f"✔ [{topic}] Done — {count:,} messages produced from {csv_path.name}")


if __name__ == "__main__":
    print("\n→ Streaming training data …")
    produce_csv(TRAIN_CSV, TRAIN_TOPIC)