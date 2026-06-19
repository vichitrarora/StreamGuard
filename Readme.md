# Hate Speech Detection — Online Learning with Kafka

A multi-label toxicity classifier trained on a live Kafka stream using prequential (test-then-train) evaluation. Supports five embedding strategies and a three-model voting ensemble per label.

---

## Project Structure

```
prml/
├── cli.py
├── producer.py
├── consumer.py
├── config.py
├── config.ini
├── requirements.txt
├── embeddings/             # Embedding strategies
│   ├── bow.py
│   ├── tfidf.py
│   ├── hashing.py
│   ├── distilbert_emb.py
│   └── e5_emb.py
└── models/                 # Ensemble and metrics
    ├── ensemble.py
    └── metrics.py
```

---

## Setup

### 1. Create and activate a virtual environment

```bash
uv venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
uv pip install -r prml/requirements.txt
```

> The `torch`, `transformers`, and `sentence-transformers` packages are only required when using the `distilbert` or `e5` embedders. If you are using `bow`, `tfidf`, or `hashing` only, you may omit them.

### 3. Configure Kafka

Copy the sample config and fill in your Kafka broker details:

```bash
cp prml/config.ini prml/config.ini
```

Edit `prml/config.ini`:

```ini
[default]
bootstrap.servers=<broker-host>:<port>
security.protocol=SASL_SSL          # or PLAINTEXT for local brokers
sasl.mechanisms=PLAIN
sasl.username=<api-key>
sasl.password=<api-secret>

[consumer]
group.id=hate_detection_group
```

Alternatively, set the `KAFKA_CONFIG_PATH` environment variable to point to a config file at any path:

```bash
export KAFKA_CONFIG_PATH=/path/to/config.ini
```

---

## Running

All commands are run from the `prml/` directory.

```bash
cd prml
```

### Produce data to Kafka

Stream the training CSV to the `hate_detection_train` topic:

```bash
python producer.py
```

### Train the ensemble (consumer)

Consume the stream and train the model prequentially:

```bash
# Train using Bag-of-Words (default)
python consumer.py

# Train using a specific embedding
python consumer.py --embedding tfidf
python consumer.py --embedding hashing
python consumer.py --embedding distilbert
python consumer.py --embedding e5

# Train all embeddings simultaneously (single pass over stream)
python consumer.py --embedding all

# Train all, skipping specific embedders
python consumer.py --embedding all --skip distilbert e5

# Adjust the positive-class probability threshold (default: 0.5)
python consumer.py --embedding bow --threshold 0.4
```

Saved models are written to `../saved_models/ensemble_<embedding>.pkl`.
Plots and metrics JSON are written to `../outputs/`.

### Run inference (CLI)

After training, classify text interactively or in single-shot mode:

```bash
# Interactive mode (default: bow)
python cli.py

# Interactive mode with a specific embedding
python cli.py --embedding tfidf

# Single prediction, then exit
python cli.py --text "you are an idiot"
python cli.py --embedding distilbert --text "some input text"
```

Available embedding choices: `bow`, `tfidf`, `hashing`, `distilbert`, `e5`.

---

## Available Embedders

| Key          | Description                              |
|--------------|------------------------------------------|
| `bow`        | Bag-of-Words (sparse token counts)       |
| `tfidf`      | Streaming TF-IDF                         |
| `hashing`    | Feature hashing (fixed-width sparse)     |
| `distilbert` | DistilBERT mean-pooled (768-dim dense)   |
| `e5`         | E5-base-v2 via sentence-transformers (768-dim dense) |

## Ensemble Architecture

Each of the 6 toxicity labels (toxic, severe\_toxic, obscene, threat, insult, identity\_hate) is backed by an independent voting ensemble of three online learners:

- **ARF** — Adaptive Random Forest
- **HAT** — Hoeffding Adaptive Tree
- **LR** — Logistic Regression (Adam optimizer)

Predictions are made by averaging predicted probabilities across the three models and applying a configurable threshold.
