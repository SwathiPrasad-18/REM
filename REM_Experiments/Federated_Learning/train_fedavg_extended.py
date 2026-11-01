import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input, Embedding, LSTM, Dense, Dropout, SpatialDropout1D
)
from tensorflow.keras.preprocessing.sequence import pad_sequences

SEED = 42
CLIENTS = 3
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
BASE_DIR = Path("REM_Experiments/Federated_Learning")
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

    result = []
    for layer_values in zip(*client_weights):
        value = np.zeros_like(layer_values[0], dtype=np.float64)
        for weights, size in zip(layer_values, client_sizes):
            value += weights.astype(np.float64) * (size / total)
        result.append(value.astype(layer_values[0].dtype))
    return result


def evaluate(model, X, y):
    scores = model.predict(X, batch_size=BATCH_SIZE, verbose=0).ravel()
    pred = (scores >= 0.5).astype(int)
    return {
        "samples": int(len(y)),
        "threshold": 0.5,
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": (
            float(roc_auc_score(y, scores))
            if len(np.unique(y)) == 2 else None
        ),
        "pr_auc": (
            float(average_precision_score(y, scores))
            if len(np.unique(y)) == 2 else None
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rounds", type=int, required=True, choices=[10, 20],
        help="Number of independent FedAvg rounds."
    )
    args = parser.parse_args()
    rounds = args.rounds

    out_dir = BASE_DIR / "extended_rounds" / f"rounds_{rounds}"
    model_dir = out_dir / "artifacts"

    # Fail closed: never overwrite an earlier experiment.
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"Output directory already contains files: {out_dir}. "
            "Choose a new output directory before running again."
        )

    for path in (DATA_PATH, MANIFEST_PATH, VOCAB_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required input is missing: {path}")

    partition_metadata = BASE_DIR / "partition_metadata.json"
    if not partition_metadata.exists():
        raise FileNotFoundError(
            "Run prepare_client_partitions.py first."
        )

    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)

    df = pd.read_csv(DATA_PATH, low_memory=False)
    manifest = pd.read_csv(MANIFEST_PATH)
    vocab = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))

    required = {"sha256", "source_label", *API_COLS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")
    if not {"sha256", "split"}.issubset(manifest.columns):
        raise ValueError("Split manifest must contain sha256 and split.")
    if vocab.get("<PAD>") != 0 or vocab.get("<UNK>") != 1:
        raise ValueError("Existing vocabulary PAD/UNK IDs are invalid.")

    # Validate binary labels before filtering.
    df["source_label"] = pd.to_numeric(df["source_label"], errors="coerce")
    df = df[df["source_label"].isin([0, 1])].copy()
    df["source_label"] = df["source_label"].astype(int)

    # Apply explicit supervised-training eligibility safeguards when present.
    if "eligible_for_supervised_training" in df.columns:
        eligible = df["eligible_for_supervised_training"]
        if eligible.dtype != bool:
            eligible = eligible.astype(str).str.strip().str.lower().isin(
                ["true", "1", "yes"]
            )
        df = df[eligible].copy()

    if "hash_label_conflict" in df.columns:
        conflict = df["hash_label_conflict"]
        if conflict.dtype != bool:
            conflict = conflict.astype(str).str.strip().str.lower().isin(
                ["true", "1", "yes"]
            )
        df = df[~conflict].copy()

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

    # Quarantine sequences with contradictory labels.
    counts = df.groupby("sequence_hash")["source_label"].nunique()
    conflicts = set(counts[counts > 1].index)
    df = df[~df["sequence_hash"].isin(conflicts)].copy()

    # Enforce a single manifest mapping per file hash.
    if manifest["sha256"].duplicated().any():
        raise ValueError("Manifest contains duplicate SHA256 values.")

    df = df.merge(
        manifest[["sha256", "split"]],
        on="sha256", how="inner", validate="many_to_one"
    )
    df = df.drop_duplicates("sequence_hash").copy()

    train = df[df["split"] == "train"].copy().reset_index(drop=True)
    val = df[df["split"] == "validation"].copy().reset_index(drop=True)

    if len(train) != 1035:
        raise ValueError(
            f"Expected 1035 effective training sequences, got {len(train)}. "
            "Stop and investigate eligibility filtering and split alignment."
        )
    if val.empty or train["source_label"].nunique() != 2:
        raise ValueError("Training or validation split is invalid.")
    if val["source_label"].nunique() != 2:
        raise ValueError("Validation split must contain both classes.")

    def encode(tokens):
        return [int(vocab.get(token, 1)) for token in tokens]

    X_train = pad_sequences(
        [encode(x) for x in train["api_tokens"]],
        maxlen=MAX_LEN, padding="post", truncating="post"
    )
    X_val = pad_sequences(
        [encode(x) for x in val["api_tokens"]],
        maxlen=MAX_LEN, padding="post", truncating="post"
    )
    y_train = train["source_label"].to_numpy()
    y_val = val["source_label"].to_numpy()

    # Reproduce the deterministic three-client sequence assignment.
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

    # Reset the same seed for independent 10- and 20-round initialization.
    tf.keras.backend.clear_session()
    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    server_model = build_model(len(vocab))

    # Create output only after input and split checks have passed.
    model_dir.mkdir(parents=True, exist_ok=False)
    history = []

    print("REM independent simulated FedAvg experiment")
    print("Rounds:", rounds, "| clients:", CLIENTS, "| seed:", SEED)
    print("Train:", len(train), "| validation:", len(val))
    print("Client sample counts:", [len(x) for x in client_indices])
    print("Test split is not loaded or evaluated.")

    for round_number in range(1, rounds + 1):
        start = time.time()
        global_weights = server_model.get_weights()
        client_weights, client_sizes, client_losses = [], [], []

        for client_id, indices in enumerate(client_indices):
            tf.keras.backend.clear_session()
            local_model = build_model(len(vocab))
            local_model.set_weights(global_weights)

            local_history = local_model.fit(
                X_train[indices], y_train[indices],
                epochs=LOCAL_EPOCHS,
                batch_size=BATCH_SIZE,
                shuffle=True,
                verbose=0,
            )
            client_weights.append(local_model.get_weights())
            client_sizes.append(len(indices))
            client_losses.append(float(local_history.history["loss"][-1]))
            del local_model

        server_model.set_weights(
            weighted_average(client_weights, client_sizes)
        )
        metrics = evaluate(server_model, X_val, y_val)
        row = {
            "round": round_number,
            "client_samples": json.dumps(client_sizes),
            "mean_client_training_loss": float(np.mean(client_losses)),
            "elapsed_seconds": round(time.time() - start, 3),
            **{f"validation_{k}": v for k, v in metrics.items()},
        }
        history.append(row)
        print(json.dumps(row))

        server_model.save(
            model_dir / f"fedavg_lstm_round_{round_number:02d}.keras"
        )
        pd.DataFrame(history).to_csv(
            out_dir / "fedavg_round_metrics.csv", index=False
        )

    final_path = model_dir / "fedavg_lstm_final.keras"
    server_model.save(final_path)

    metrics_df = pd.DataFrame(history)
    best_pr_row = metrics_df.loc[
        pd.to_numeric(metrics_df["validation_pr_auc"]).idxmax()
    ]
    best_f1_row = metrics_df.loc[
        pd.to_numeric(metrics_df["validation_f1"]).idxmax()
    ]

    summary = {
        "experiment": "REM simulated three-client FedAvg LSTM",
        "seed": SEED,
        "clients": CLIENTS,
        "rounds": rounds,
        "local_epochs": LOCAL_EPOCHS,
        "batch_size": BATCH_SIZE,
        "training_samples": len(train),
        "validation_samples": len(val),
        "test_evaluated": False,
        "threshold_for_reported_metrics": 0.5,
        "vocabulary_size": len(vocab),
        "partition_strategy": "deterministic label-stratified sequence groups",
        "aggregation": "sample-weighted FedAvg",
        "best_validation_pr_auc_round": int(best_pr_row["round"]),
        "best_validation_pr_auc": float(best_pr_row["validation_pr_auc"]),
        "best_validation_f1_at_0_5_round": int(best_f1_row["round"]),
        "best_validation_f1_at_0_5": float(best_f1_row["validation_f1"]),
        "final_model": str(final_path),
        "note": (
            "Simulated federated learning on one machine; clients are not "
            "independent organizations. Validation metrics are for development. "
            "The test set remains reserved. The final checkpoint is the last "
            "round; select a checkpoint using validation results only."
        ),
    }
    (out_dir / "fedavg_run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("\nTraining complete. Outputs saved under:", out_dir.resolve())
    print("Best validation PR-AUC round:", int(best_pr_row["round"]))
    print("Centralized and original five-round artifacts were not targeted.")


if __name__ == "__main__":
    main()
