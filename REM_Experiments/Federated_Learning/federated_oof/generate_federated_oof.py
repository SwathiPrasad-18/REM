
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score, average_precision_score
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input, Embedding, LSTM, Dense, Dropout, SpatialDropout1D
)
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ============================================================
# REM: Five-fold simulated federated LSTM OOF experiment
# ============================================================

SEED = 42
CLIENTS = 3
FOLDS = 5
ROUNDS = 17
LOCAL_EPOCHS = 1
BATCH_SIZE = 32
MAX_LEN = 153

DATA_PATH = Path(
    "REM_Datasets/04_Unified_Dataset/processed/public_labeled_candidate.csv"
)
MANIFEST_PATH = Path(
    "REM_Models/API_Sequence_Grouped/grouped_split_manifest.csv"
)
RF_IF_OOF_PATH = Path(
    "REM_Models/Hybrid_REM/OOF_Predictions/train_oof_rf_if.csv"
)
CENTRAL_OOF_PATH = Path(
    "REM_Models/Hybrid_REM/OOF_Predictions/train_oof_lstm.csv"
)

OUT_DIR = Path(
    "REM_Experiments/Federated_Learning/federated_oof"
)
ARTIFACT_DIR = OUT_DIR / "artifacts"

API_COLS = [str(i) for i in range(MAX_LEN)]
OUTPUT_CSV = OUT_DIR / "federated_lstm_oof.csv"
OUTPUT_AUDIT = OUT_DIR / "federated_oof_audit.json"


def make_sequence(row):
    """Convert API columns into a cleaned token sequence."""
    ignored = {"", "nan", "none", "null"}
    tokens = []

    for value in row:
        token = str(value).strip()
        if token.lower() not in ignored:
            tokens.append(token)

    return tokens


def calculate_sequence_hash(tokens):
    """Hash a normalized API token sequence."""
    content = " ".join(tokens).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def build_model(vocab_size):
    """Build the REM LSTM architecture used in the FedAvg experiments."""
    model = Sequential([
        Input(shape=(MAX_LEN,), dtype="int32"),
        Embedding(
            input_dim=vocab_size,
            output_dim=64,
            mask_zero=True,
        ),
        SpatialDropout1D(0.20),
        LSTM(64, dropout=0.20),
        Dense(32, activation="relu"),
        Dropout(0.30),
        Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="binary_crossentropy",
    )
    return model


def weighted_fedavg(client_weights, client_sizes):
    """Aggregate client weights according to client sample counts."""
    total = float(sum(client_sizes))
    if total <= 0:
        raise ValueError("Cannot aggregate empty clients.")

    aggregated = []

    for layer_values in zip(*client_weights):
        average = np.zeros_like(
            layer_values[0], dtype=np.float64
        )

        for weights, size in zip(layer_values, client_sizes):
            average += weights.astype(np.float64) * (size / total)

        aggregated.append(average.astype(layer_values[0].dtype))

    return aggregated


def make_client_indices(frame, seed):
    """
    Reproduce the deterministic label-stratified sequence-group
    assignment used in the REM FedAvg experiment.
    """
    rng = np.random.default_rng(seed)
    assignments = {}

    for label, group in frame.groupby("label", sort=True):
        hashes = (
            group["sequence_hash"]
            .drop_duplicates()
            .to_numpy()
            .copy()
        )
        rng.shuffle(hashes)

        for index, seq_hash in enumerate(hashes):
            assignments[seq_hash] = index % CLIENTS

    client_indices = [
        np.flatnonzero(
            frame["sequence_hash"].map(assignments).to_numpy()
            == client_id
        )
        for client_id in range(CLIENTS)
    ]

    if any(len(indices) == 0 for indices in client_indices):
        raise ValueError("At least one client has no training samples.")

    if sum(len(indices) for indices in client_indices) != len(frame):
        raise ValueError("Client assignments do not cover the fold.")

    return client_indices


def encode_and_pad(sequences, vocab):
    """Encode sequences and pad them to MAX_LEN."""
    encoded = [
        [vocab.get(token, 1) for token in sequence]
        for sequence in sequences
    ]

    return pad_sequences(
        encoded,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post",
    )


def validate_input_files():
    """Verify required input files before reading or training."""
    required_paths = [
        DATA_PATH,
        MANIFEST_PATH,
        RF_IF_OOF_PATH,
        CENTRAL_OOF_PATH,
    ]

    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(
                f"Required input file not found: {path}"
            )

    if not OUT_DIR.is_dir():
        raise FileNotFoundError(
            f"Expected workspace directory: {OUT_DIR}"
        )

    if not ARTIFACT_DIR.is_dir():
        raise FileNotFoundError(
            f"Expected artifact directory: {ARTIFACT_DIR}"
        )

    # Protect against overwriting prior experiment outputs.
    for output_path in (OUTPUT_CSV, OUTPUT_AUDIT):
        if output_path.exists():
            raise FileExistsError(
                f"Output already exists: {output_path}. "
                "Back up or choose a new output directory."
            )

    existing_artifacts = list(ARTIFACT_DIR.iterdir())
    if existing_artifacts:
        raise FileExistsError(
            "The artifacts directory is not empty. "
            "Inspect it and use a new output directory before training."
        )


def load_and_validate_oof_files():
    """Verify the original RF/IF and centralized LSTM OOF alignment."""
    keys = ["sha256", "sequence_hash", "label"]

    rf_if = pd.read_csv(RF_IF_OOF_PATH)
    central = pd.read_csv(CENTRAL_OOF_PATH)

    required_oof_columns = set(keys + ["fold"])

    for name, frame in (
        ("RF/IF", rf_if),
        ("Central LSTM", central),
    ):
        missing = required_oof_columns - set(frame.columns)
        if missing:
            raise ValueError(
                f"{name} OOF file is missing columns: {sorted(missing)}"
            )

        if frame.duplicated(keys).any():
            raise ValueError(
                f"{name} OOF file contains duplicate prediction keys."
            )

    rf_if["sha256"] = (
        rf_if["sha256"].astype(str).str.strip().str.lower()
    )
    central["sha256"] = (
        central["sha256"].astype(str).str.strip().str.lower()
    )

    for frame in (rf_if, central):
        frame["sequence_hash"] = (
            frame["sequence_hash"].astype(str).str.strip().str.lower()
        )
        frame["label"] = pd.to_numeric(
            frame["label"], errors="raise"
        ).astype(int)
        frame["fold"] = pd.to_numeric(
            frame["fold"], errors="raise"
        ).astype(int)

    alignment = rf_if[keys + ["fold"]].merge(
        central[keys + ["fold"]],
        on=keys,
        how="outer",
        suffixes=("_rfif", "_central"),
        indicator=True,
        validate="one_to_one",
    )

    all_matched = bool((alignment["_merge"] == "both").all())
    if not all_matched:
        raise ValueError(
            "RF/IF and centralized LSTM OOF keys do not align."
        )

    matched = alignment[alignment["_merge"] == "both"]

    fold_match = bool(
        (
            matched["fold_rfif"] == matched["fold_central"]
        ).all()
    )
    if not fold_match:
        raise ValueError(
            "RF/IF and centralized LSTM OOF fold assignments differ."
        )

    expected_folds = set(range(1, FOLDS + 1))

    if set(rf_if["fold"].unique()) != expected_folds:
        raise ValueError("RF/IF OOF must contain fold IDs 1 through 5.")

    if len(rf_if) != 1035 or len(central) != 1035:
        raise ValueError(
            "Expected 1,035 rows in each existing OOF file."
        )

    return rf_if, central, {
        "rows_rf_if": len(rf_if),
        "rows_central_lstm": len(central),
        "duplicate_keys_rf_if": 0,
        "duplicate_keys_central_lstm": 0,
        "all_keys_matched": all_matched,
        "fold_assignments_match": fold_match,
        "samples_per_fold": {
            str(int(fold)): int(count)
            for fold, count in
            rf_if["fold"].value_counts().sort_index().items()
        },
    }


def prepare_training_data():
    """Load, clean, split, and align the supervised training sequences."""
    data = pd.read_csv(DATA_PATH, low_memory=False)
    manifest = pd.read_csv(MANIFEST_PATH)

    required = {"sha256", "source_label", *API_COLS}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}"
        )

    if not {"sha256", "split"}.issubset(manifest.columns):
        raise ValueError(
            "Split manifest must contain sha256 and split columns."
        )

    data["source_label"] = pd.to_numeric(
        data["source_label"], errors="coerce"
    )
    data = data[data["source_label"].isin([0, 1])].copy()
    data["source_label"] = data["source_label"].astype(int)

    # Apply eligibility safeguards if the columns exist.
    if "eligible_for_supervised_training" in data.columns:
        eligible = data["eligible_for_supervised_training"]

        if eligible.dtype != bool:
            eligible = eligible.astype(str).str.strip().str.lower().isin(
                ["true", "1", "yes"]
            )

        data = data[eligible].copy()

    if "hash_label_conflict" in data.columns:
        conflict = data["hash_label_conflict"]

        if conflict.dtype != bool:
            conflict = conflict.astype(str).str.strip().str.lower().isin(
                ["true", "1", "yes"]
            )

        data = data[~conflict].copy()

    data["sha256"] = (
        data["sha256"].astype(str).str.strip().str.lower()
    )
    manifest["sha256"] = (
        manifest["sha256"].astype(str).str.strip().str.lower()
    )

    if manifest["sha256"].duplicated().any():
        raise ValueError("Split manifest contains duplicate SHA256 values.")

    data["api_tokens"] = data[API_COLS].apply(
        make_sequence, axis=1
    )
    data["sequence_hash"] = data["api_tokens"].map(
        calculate_sequence_hash
    )

    # Exclude sequences with contradictory labels.
    label_counts = data.groupby("sequence_hash")["source_label"].nunique()
    conflicting_hashes = set(label_counts[label_counts > 1].index)

    data = data[
        ~data["sequence_hash"].isin(conflicting_hashes)
    ].copy()

    data = data.merge(
        manifest[["sha256", "split"]],
        on="sha256",
        how="inner",
        validate="many_to_one",
    )

    data = data[data["split"] == "train"].copy()
    data = data.drop_duplicates("sequence_hash").reset_index(drop=True)

    if len(data) != 1035:
        raise ValueError(
            f"Expected 1,035 training sequences, got {len(data)}."
        )

    if data["sequence_hash"].duplicated().any():
        raise ValueError("Training sequence hashes are not unique.")

    if data["source_label"].nunique() != 2:
        raise ValueError("Training data must contain both label classes.")

    data = data.rename(columns={"source_label": "label"})

    return data


def train_fold_model(train, held, fold):
    """Train a fold-isolated FedAvg LSTM and predict its held-out fold."""
    start_time = time.time()

    if set(train["sequence_hash"]) & set(held["sequence_hash"]):
        raise ValueError(
            f"Sequence leakage detected in fold {fold}."
        )

    if train["label"].nunique() != 2:
        raise ValueError(f"Fold {fold} training data lacks a class.")

    if held["label"].nunique() != 2:
        raise ValueError(f"Fold {fold} held-out data lacks a class.")

    # Build vocabulary from this fold's training sequences only.
    vocab = {"<PAD>": 0, "<UNK>": 1}

    for tokens in train["api_tokens"]:
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab)

    X_train = encode_and_pad(train["api_tokens"], vocab)
    X_held = encode_and_pad(held["api_tokens"], vocab)
    y_train = train["label"].to_numpy()

    # Same deterministic partition seed for each fold.
    client_indices = make_client_indices(train, SEED)

    # Reproducible but fold-specific model initialization.
    fold_seed = SEED + fold
    random.seed(fold_seed)
    np.random.seed(fold_seed)
    tf.keras.utils.set_random_seed(fold_seed)
    tf.keras.backend.clear_session()

    server_model = build_model(len(vocab))

    for round_number in range(1, ROUNDS + 1):
        global_weights = server_model.get_weights()
        client_weights = []
        client_sizes = []

        for indices in client_indices:
            tf.keras.backend.clear_session()

            local_model = build_model(len(vocab))
            local_model.set_weights(global_weights)

            local_model.fit(
                X_train[indices],
                y_train[indices],
                epochs=LOCAL_EPOCHS,
                batch_size=BATCH_SIZE,
                shuffle=True,
                verbose=0,
            )

            client_weights.append(local_model.get_weights())
            client_sizes.append(len(indices))
            del local_model

        server_model.set_weights(
            weighted_fedavg(client_weights, client_sizes)
        )

    scores = server_model.predict(
        X_held, batch_size=BATCH_SIZE, verbose=0
    ).ravel()

    if len(scores) != len(held):
        raise ValueError(
            f"Fold {fold} prediction count does not match held-out rows."
        )

    if not np.isfinite(scores).all():
        raise ValueError(f"Fold {fold} produced non-finite scores.")

    if ((scores < 0) | (scores > 1)).any():
        raise ValueError(f"Fold {fold} scores are outside [0, 1].")

    # Save fold-specific artifacts without overwriting prior files.
    model_path = ARTIFACT_DIR / f"fedavg_lstm_fold_{fold}.keras"
    vocab_path = ARTIFACT_DIR / f"vocabulary_fold_{fold}.json"

    if model_path.exists() or vocab_path.exists():
        raise FileExistsError(
            f"Fold {fold} artifact already exists; refusing to overwrite."
        )

    server_model.save(model_path)
    vocab_path.write_text(
        json.dumps(vocab, indent=2), encoding="utf-8"
    )

    predictions = pd.DataFrame({
        "sha256": held["sha256"].to_numpy(),
        "sequence_hash": held["sequence_hash"].to_numpy(),
        "label": held["label"].to_numpy(),
        "fold": fold,
        "federated_lstm_oof_score": scores,
    })

    fold_audit = {
        "fold": fold,
        "train_samples": int(len(train)),
        "heldout_samples": int(len(held)),
        "client_samples": [int(len(x)) for x in client_indices],
        "vocabulary_size": int(len(vocab)),
        "rounds": ROUNDS,
        "local_epochs": LOCAL_EPOCHS,
        "heldout_used_for_training": False,
        "sequence_overlap": False,
        "elapsed_seconds": round(time.time() - start_time, 2),
        "model_path": str(model_path),
        "vocabulary_path": str(vocab_path),
    }

    del server_model, X_train, X_held
    tf.keras.backend.clear_session()

    return predictions, fold_audit


def main():
    validate_input_files()

    print("REM federated LSTM OOF experiment")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Folds: {FOLDS} | rounds per fold: {ROUNDS}")
    print(f"Clients per fold: {CLIENTS}")
    print("Reserved test set will not be loaded or evaluated.")

    rf_if, _, source_oof_audit = load_and_validate_oof_files()
    data = prepare_training_data()

    keys = ["sha256", "sequence_hash", "label"]

    fold_map = rf_if[keys + ["fold"]].copy()
    fold_map["sha256"] = (
        fold_map["sha256"].astype(str).str.strip().str.lower()
    )
    fold_map["sequence_hash"] = (
        fold_map["sequence_hash"].astype(str).str.strip().str.lower()
    )
    fold_map["label"] = fold_map["label"].astype(int)

    data = data.merge(
        fold_map,
        on=keys,
        how="left",
        validate="one_to_one",
    )

    if data["fold"].isna().any():
        raise ValueError(
            "Some training sequences have no matching existing OOF fold."
        )

    data["fold"] = data["fold"].astype(int)

    expected_folds = set(range(1, FOLDS + 1))
    if set(data["fold"].unique()) != expected_folds:
        raise ValueError("Expected fold IDs 1 through 5.")

    if len(data) != len(rf_if):
        raise ValueError("Training data and RF/IF OOF row counts differ.")

    if data.duplicated(keys).any():
        raise ValueError("Duplicate training prediction keys detected.")

    all_predictions = []
    fold_audits = []

    for fold in range(1, FOLDS + 1):
        print(f"\nStarting fold {fold}/{FOLDS} ...")

        train = (
            data[data["fold"] != fold]
            .copy()
            .reset_index(drop=True)
        )
        held = (
            data[data["fold"] == fold]
            .copy()
            .reset_index(drop=True)
        )

        predictions, fold_audit = train_fold_model(
            train, held, fold
        )

        all_predictions.append(predictions)
        fold_audits.append(fold_audit)

        print(json.dumps(fold_audit))

    oof = pd.concat(all_predictions, ignore_index=True)

    if len(oof) != len(data):
        raise ValueError("Final OOF row count mismatch.")

    if oof.duplicated(keys).any():
        raise ValueError("Duplicate keys in final federated OOF predictions.")

    if oof["sequence_hash"].nunique() != len(oof):
        raise ValueError("Final OOF sequence hashes are not unique.")

    if oof["federated_lstm_oof_score"].isna().any():
        raise ValueError("Missing federated OOF scores.")

    if not np.isfinite(
        oof["federated_lstm_oof_score"].to_numpy()
    ).all():
        raise ValueError("Non-finite federated OOF scores.")

    # Verify exact key and fold alignment with RF/IF OOF predictions.
    check = rf_if[keys + ["fold"]].merge(
        oof[keys + ["fold"]],
        on=keys,
        how="outer",
        suffixes=("_rfif", "_federated"),
        indicator=True,
        validate="one_to_one",
    )

    alignment_ok = bool((check["_merge"] == "both").all())

    if not alignment_ok:
        raise ValueError(
            "Federated OOF records do not align with RF/IF OOF."
        )

    fold_alignment_ok = bool(
        (
            check.loc[check["_merge"] == "both", "fold_rfif"]
            ==
            check.loc[check["_merge"] == "both", "fold_federated"]
        ).all()
    )

    if not fold_alignment_ok:
        raise ValueError(
            "Federated OOF fold assignments differ from RF/IF OOF."
        )

    # Calculate metrics from OOF predictions only.
    labels = oof["label"].to_numpy()
    scores = oof["federated_lstm_oof_score"].to_numpy()

    roc_auc = float(roc_auc_score(labels, scores))
    pr_auc = float(average_precision_score(labels, scores))

    # Recheck output paths immediately before writing.
    if OUTPUT_CSV.exists() or OUTPUT_AUDIT.exists():
        raise FileExistsError(
            "Output appeared during the run; refusing to overwrite."
        )

    oof.to_csv(OUTPUT_CSV, index=False)

    audit = {
        "experiment": "Five-fold simulated federated LSTM OOF",
        "seed": SEED,
        "clients": CLIENTS,
        "folds": FOLDS,
        "rounds_per_fold": ROUNDS,
        "local_epochs": LOCAL_EPOCHS,
        "batch_size": BATCH_SIZE,
        "training_samples": int(len(data)),
        "oof_rows": int(len(oof)),
        "source_oof_audit": source_oof_audit,
        "folds_match_existing_oof": alignment_ok,
        "fold_assignments_match": fold_alignment_ok,
        "fold_specific_vocabulary": True,
        "heldout_used_for_training": False,
        "test_evaluated": False,
        "roc_auc_oof": roc_auc,
        "average_precision_oof": pr_auc,
        "fold_details": fold_audits,
        "output_csv": str(OUTPUT_CSV),
        "note": (
            "Simulated federated learning on one machine. OOF models use "
            "fold-specific vocabularies and train only on four folds. "
            "Their score distribution may differ from a full-training "
            "model. These OOF metrics are development diagnostics, not "
            "a substitute for final reserved-test evaluation."
        ),
    }

    # Write audit last, after prediction file creation succeeds.
    OUTPUT_AUDIT.write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )

    print("\nFederated OOF generation completed successfully.")
    print(f"OOF rows: {len(oof)}")
    print(f"OOF ROC-AUC: {roc_auc:.6f}")
    print(f"OOF average precision: {pr_auc:.6f}")
    print(f"Fold alignment passed: {fold_alignment_ok}")
    print(f"Predictions: {OUTPUT_CSV.resolve()}")
    print(f"Audit: {OUTPUT_AUDIT.resolve()}")


if __name__ == "__main__":
    main()
