
import json
import hashlib
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    Embedding,
    LSTM,
    Dense,
    Dropout,
    SpatialDropout1D,
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def make_sequence(row):
    ignored = {"", "nan", "none", "null"}
    return [
        token
        for value in row
        if (token := str(value).strip()).lower() not in ignored
    ]


def main():
    root = Path(".")
    data_path = (
        root / "REM_Datasets" / "04_Unified_Dataset"
        / "processed" / "public_labeled_candidate.csv"
    )
    manifest_path = (
        root / "REM_Models" / "API_Sequence_Grouped"
        / "grouped_split_manifest.csv"
    )
    out_dir = (
        root / "REM_Models" / "LSTM_Baseline"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path, low_memory=False)
    manifest = pd.read_csv(manifest_path)

    api_cols = [str(i) for i in range(153)]
    if any(c not in df.columns for c in api_cols):
        raise ValueError("One or more API columns are missing.")

    df["sha256"] = df["sha256"].astype(str)
    manifest["sha256"] = manifest["sha256"].astype(str)

    # Rebuild ordered API sequences and identify identical sequences.
    df["api_tokens"] = df[api_cols].apply(make_sequence, axis=1)
    df["sequence_hash"] = df["api_tokens"].map(
        lambda tokens: hashlib.sha256(
            " ".join(tokens).encode("utf-8")
        ).hexdigest()
    )

    # Quarantine sequences that have conflicting labels.
    label_counts = df.groupby("sequence_hash")["source_label"].nunique()
    conflicting_hashes = set(label_counts[label_counts > 1].index)
    df = df[~df["sequence_hash"].isin(conflicting_hashes)].copy()

    # Reuse the original group-aware splits.
    df = df.merge(
        manifest[["sha256", "split"]],
        on="sha256",
        how="inner",
        validate="one_to_one",
    )

    # Keep one example per unique sequence.
    df = df.drop_duplicates("sequence_hash").copy()

    train = df[df["split"] == "train"].copy()
    val = df[df["split"] == "validation"].copy()
    test = df[df["split"] == "test"].copy()

    if train.empty or val.empty or test.empty:
        raise ValueError("Train, validation, or test split is empty.")

    # Fit vocabulary using training data only.
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for tokens in train["api_tokens"]:
        for token in tokens:
            if token not in vocab:
                vocab[token] = len(vocab)

    def encode(tokens):
        return [vocab.get(token, 1) for token in tokens]

    train_sequences = [encode(x) for x in train["api_tokens"]]
    val_sequences = [encode(x) for x in val["api_tokens"]]
    test_sequences = [encode(x) for x in test["api_tokens"]]

    # Use a fixed length compatible with the original 153 API columns.
    max_len = 153

    X_train = pad_sequences(
        train_sequences, maxlen=max_len,
        padding="post", truncating="post"
    )
    X_val = pad_sequences(
        val_sequences, maxlen=max_len,
        padding="post", truncating="post"
    )
    X_test = pad_sequences(
        test_sequences, maxlen=max_len,
        padding="post", truncating="post"
    )

    y_train = train["source_label"].astype(int).to_numpy()
    y_val = val["source_label"].astype(int).to_numpy()
    y_test = test["source_label"].astype(int).to_numpy()

    model = Sequential([
        Input(shape=(max_len,)),
        Embedding(
            input_dim=len(vocab),
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
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1,
    )

    # Select threshold using validation data only.
    val_scores = model.predict(X_val, verbose=0).ravel()
    threshold_rows = []

    for threshold in np.round(np.arange(0.10, 0.901, 0.01), 2):
        pred = (val_scores >= threshold).astype(int)
        threshold_rows.append({
            "threshold": float(threshold),
            "precision": float(
                precision_score(y_val, pred, zero_division=0)
            ),
            "recall": float(
                recall_score(y_val, pred, zero_division=0)
            ),
            "f1": float(f1_score(y_val, pred, zero_division=0)),
        })

    threshold_df = pd.DataFrame(threshold_rows)
    threshold_df.to_csv(
        out_dir / "validation_threshold_metrics.csv",
        index=False,
    )

    eligible = threshold_df[threshold_df["precision"] >= 0.80]

    if eligible.empty:
        chosen = threshold_df.sort_values(
            ["f1", "recall", "threshold"],
            ascending=[False, False, False],
        ).iloc[0]
        selection_rule = "highest F1; precision target unmet"
    else:
        chosen = eligible.sort_values(
            ["recall", "f1", "threshold"],
            ascending=[False, False, False],
        ).iloc[0]
        selection_rule = "highest recall with precision >= 0.80"

    threshold = float(chosen["threshold"])

    # Evaluate the held-out test set once at the chosen threshold.
    test_scores = model.predict(X_test, verbose=0).ravel()
    test_pred = (test_scores >= threshold).astype(int)

    results = {
        "seed": SEED,
        "train_samples": int(len(train)),
        "validation_samples": int(len(val)),
        "test_samples": int(len(test)),
        "vocabulary_size": int(len(vocab)),
        "max_sequence_length": max_len,
        "epochs_trained": len(history.history["loss"]),
        "selected_threshold": threshold,
        "threshold_selection_rule": selection_rule,
        "validation_selected_metrics": {
            "precision": float(chosen["precision"]),
            "recall": float(chosen["recall"]),
            "f1": float(chosen["f1"]),
        },
        "test": {
            "accuracy": float(accuracy_score(y_test, test_pred)),
            "roc_auc": float(roc_auc_score(y_test, test_scores)),
            "confusion_matrix_labels_0_1": confusion_matrix(
                y_test, test_pred, labels=[0, 1]
            ).tolist(),
            "classification_report": classification_report(
                y_test,
                test_pred,
                labels=[0, 1],
                target_names=["benign", "malicious"],
                output_dict=True,
                zero_division=0,
            ),
        },
        "sequence_overlap_checks": {
            "train_validation": len(
                set(train["sequence_hash"]) &
                set(val["sequence_hash"])
            ),
            "train_test": len(
                set(train["sequence_hash"]) &
                set(test["sequence_hash"])
            ),
            "validation_test": len(
                set(val["sequence_hash"]) &
                set(test["sequence_hash"])
            ),
        },
        "note": (
            "Threshold selected on validation data only. "
            "Test metrics are held-out evaluation results."
        ),
    }

    model.save(out_dir / "rem_lstm.keras")

    with open(out_dir / "vocabulary.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2)

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    pd.DataFrame({
        "sha256": test["sha256"].to_numpy(),
        "actual_label": y_test,
        "malware_probability": test_scores,
        "predicted_label": test_pred,
    }).to_csv(out_dir / "test_predictions.csv", index=False)

    print(json.dumps(results, indent=2))
    print(f"\nSaved results to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
