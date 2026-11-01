import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input, Embedding, LSTM, Dense, Dropout, SpatialDropout1D
)
from tensorflow.keras.preprocessing.sequence import pad_sequences

SEED = 42
CLIENTS = 3
ROUNDS = 5
LOCAL_EPOCHS = 1
BATCH_SIZE = 32
MAX_LEN = 153

DATA_PATH = Path(
    "REM_Datasets/04_Unified_Dataset/processed/public_labeled_candidate.csv"
)
MANIFEST_PATH = Path(
    "REM_Models/API_Sequence_Grouped/grouped_split_manifest.csv"
)
VOCAB_PATH = Path("REM_Models/LSTM_Baseline/vocabulary.json")
OUT_DIR = Path("REM_Experiments/Federated_Learning")
MODEL_DIR = OUT_DIR / "artifacts"
API_COLS = [str(i) for i in range(MAX_LEN)]


def make_sequence(row):
    ignored = {"", "nan", "none", "null"}
    return [
        token for value in row
        if (token := str(value).strip()).lower() not in ignored
    ]


def build_model(vocab_size):
    model = Sequential([
        Input(shape=(MAX_LEN,), dtype="int32"),
        Embedding(input_dim=vocab_size, output_dim=64, mask_zero=True),
        SpatialDropout1D(0.20),
        LSTM(64, dropout=0.20),
        Dense(32, activation="relu"),
        Dropout(0.30),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def weighted_average(client_weights, client_sizes):
    total = float(sum(client_sizes))
    if total <= 0:
        raise ValueError("Cannot aggregate empty clients.")

    aggregated = []
    for layer_values in zip(*client_weights):
        value = np.zeros_like(layer_values[0], dtype=np.float64)
        for weights, size in zip(layer_values, client_sizes):
            value += weights.astype(np.float64) * (size / total)
        aggregated.append(value.astype(layer_values[0].dtype))
    return aggregated


def metrics_for(model, X, y):
    scores = model.predict(X, batch_size=BATCH_SIZE, verbose=0).ravel()
    predictions = (scores >= 0.5).astype(int)
    result = {
        "samples": int(len(y)),
        "threshold": 0.5,
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "roc_auc": None,
        "pr_auc": None,
    }
    if len(np.unique(y)) == 2:
        result["roc_auc"] = float(roc_auc_score(y, scores))
        result["pr_auc"] = float(average_precision_score(y, scores))
    return result


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    if not Path("REM_Experiments/Federated_Learning/partition_metadata.json").exists():
        raise FileNotFoundError("Run prepare_client_partitions.py first.")
    if not DATA_PATH.exists() or not MANIFEST_PATH.exists():
        raise FileNotFoundError("Dataset or split manifest is missing.")
    if not VOCAB_PATH.exists():
        raise FileNotFoundError("Existing LSTM vocabulary is missing.")

    df = pd.read_csv(DATA_PATH, low_memory=False)
    manifest = pd.read_csv(MANIFEST_PATH)
    vocab = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))

    if vocab.get("<PAD>") != 0 or vocab.get("<UNK>") != 1:
        raise ValueError("Vocabulary PAD/UNK IDs do not match expected values.")

    df["sha256"] = df["sha256"].astype(str).str.strip().str.lower()
    manifest["sha256"] = (
        manifest["sha256"].astype(str).str.strip().str.lower()
    )

    df["api_tokens"] = df[API_COLS].apply(make_sequence, axis=1)
    df["sequence_hash"] = df["api_tokens"].map(
        lambda tokens: hashlib.sha256(
            " ".join(tokens).encode("utf-8")
        ).hexdigest()
    )

    # Quarantine any API sequence whose labels conflict.
    label_counts = df.groupby("sequence_hash")["source_label"].nunique()
    conflicts = set(label_counts[label_counts > 1].index)
    df = df[~df["sequence_hash"].isin(conflicts)].copy()

    df = df.merge(
        manifest[["sha256", "split"]],
        on="sha256",
        how="inner",
        validate="one_to_one",
    )
    df = df.drop_duplicates("sequence_hash").copy()

    train = df[df["split"] == "train"].copy()
    val = df[df["split"] == "validation"].copy()

    # Do not use the test split in model selection or development.
    if len(train) != 1035:
        raise ValueError(
            f"Expected 1035 effective training sequences, got {len(train)}. "
            "Stop and investigate before training."
        )
    if val.empty or train["source_label"].nunique() != 2:
        raise ValueError("Training or validation split is invalid.")

    # Encode all splits using the existing, training-derived vocabulary.
    def encode(tokens):
        return [int(vocab.get(token, 1)) for token in tokens]

    X_train = pad_sequences(
        [encode(x) for x in train["api_tokens"]],
        maxlen=MAX_LEN, padding="post", truncating="post",
    )
    X_val = pad_sequences(
        [encode(x) for x in val["api_tokens"]],
        maxlen=MAX_LEN, padding="post", truncating="post",
    )
    y_train = train["source_label"].astype(int).to_numpy()
    y_val = val["source_label"].astype(int).to_numpy()

    # Recreate the audited, deterministic three-client assignment.
    rng = np.random.default_rng(SEED)
    assignments = {}
    for label, group in train.groupby("source_label"):
        hashes = group["sequence_hash"].drop_duplicates().to_numpy().copy()
        rng.shuffle(hashes)
        for index, seq_hash in enumerate(hashes):
            assignments[seq_hash] = index % CLIENTS

    client_indices = [
        np.flatnonzero(
            train["sequence_hash"].map(assignments).to_numpy() == cid
        )
        for cid in range(CLIENTS)
    ]

    if any(len(indices) == 0 for indices in client_indices):
        raise ValueError("At least one client is empty.")
    if sum(map(len, client_indices)) != len(train):
        raise ValueError("Client partitions do not cover training data.")

    # Ensure every client uses exactly the same model initialization schema.
    tf.keras.backend.clear_session()
    server_model = build_model(len(vocab))
    history = []
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("REM FedAvg experiment")
    print("Clients:", CLIENTS, "| rounds:", ROUNDS,
          "| local epochs:", LOCAL_EPOCHS)
    print("Training samples:", len(train), "| validation samples:", len(val))
    print("Test split is not loaded for evaluation.")
    print("Client sample counts:", [len(x) for x in client_indices])

    for round_number in range(1, ROUNDS + 1):
        start = time.time()
        global_weights = server_model.get_weights()
        client_weights = []
        client_sizes = []
        client_losses = []

        for client_id, indices in enumerate(client_indices):
            tf.keras.backend.clear_session()
            local_model = build_model(len(vocab))
            local_model.set_weights(global_weights)

            local_history = local_model.fit(
                X_train[indices],
                y_train[indices],
                epochs=LOCAL_EPOCHS,
                batch_size=BATCH_SIZE,
                shuffle=True,
                verbose=0,
            )

            client_weights.append(local_model.get_weights())
            client_sizes.append(len(indices))
            client_losses.append(float(local_history.history["loss"][-1]))
            del local_model

        server_model.set_weights(weighted_average(client_weights, client_sizes))
        val_metrics = metrics_for(server_model, X_val, y_val)

        row = {
            "round": round_number,
            "client_samples": client_sizes,
            "mean_client_training_loss": float(np.mean(client_losses)),
            "elapsed_seconds": round(time.time() - start, 3),
            **{f"validation_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        print(json.dumps(row))

        # Checkpoint each round separately to preserve progress.
        checkpoint = MODEL_DIR / f"fedavg_lstm_round_{round_number:02d}.keras"
        server_model.save(checkpoint)

    server_model.save(MODEL_DIR / "fedavg_lstm_final.keras")
    pd.DataFrame(history).to_csv(OUT_DIR / "fedavg_round_metrics.csv", index=False)

    summary = {
        "experiment": "REM simulated three-client FedAvg LSTM",
        "seed": SEED,
        "clients": CLIENTS,
        "rounds": ROUNDS,
        "local_epochs": LOCAL_EPOCHS,
        "batch_size": BATCH_SIZE,
        "training_samples": len(train),
        "validation_samples": len(val),
        "test_evaluated": False,
        "threshold": 0.5,
        "vocabulary_size": len(vocab),
        "partition_strategy": "deterministic label-stratified sequence groups",
        "aggregation": "sample-weighted FedAvg",
        "note": (
            "Simulated federated learning on one machine; clients are "
            "not independent organizations. Validation metrics are for "
            "development and the test set remains reserved."
        ),
    }
    (OUT_DIR / "fedavg_run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("\nTraining complete. Outputs saved under:", OUT_DIR.resolve())
    print("Centralized REM artifacts were not targeted by this script.")


if __name__ == "__main__":
    main()
