import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from tensorflow.keras.preprocessing.sequence import pad_sequences

DATA_PATH = Path(
    "REM_Datasets/04_Unified_Dataset/processed/public_labeled_candidate.csv"
)
MANIFEST_PATH = Path(
    "REM_Models/API_Sequence_Grouped/grouped_split_manifest.csv"
)
VOCAB_PATH = Path("REM_Models/LSTM_Baseline/vocabulary.json")
MODEL_PATH = Path(
    "REM_Experiments/Federated_Learning/artifacts/fedavg_lstm_final.keras"
)
OUT_DIR = Path("REM_Experiments/Federated_Learning")
API_COLS = [str(i) for i in range(153)]


def make_sequence(row):
    ignored = {"", "nan", "none", "null"}
    return [
        token for value in row
        if (token := str(value).strip()).lower() not in ignored
    ]


def main():
    for path in [DATA_PATH, MANIFEST_PATH, VOCAB_PATH, MODEL_PATH]:
        if not path.exists():
            raise FileNotFoundError(path)

    df = pd.read_csv(DATA_PATH, low_memory=False)
    manifest = pd.read_csv(MANIFEST_PATH)
    vocab = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))

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

    counts = df.groupby("sequence_hash")["source_label"].nunique()
    conflict_hashes = set(counts[counts > 1].index)
    df = df[~df["sequence_hash"].isin(conflict_hashes)].copy()
    df = df.merge(
        manifest[["sha256", "split"]],
        on="sha256",
        how="inner",
        validate="one_to_one",
    )
    df = df.drop_duplicates("sequence_hash")
    val = df[df["split"] == "validation"].copy()

    if val.empty:
        raise ValueError("Validation split is empty.")

    def encode(tokens):
        return [int(vocab.get(token, 1)) for token in tokens]

    X_val = pad_sequences(
        [encode(tokens) for tokens in val["api_tokens"]],
        maxlen=153,
        padding="post",
        truncating="post",
    )
    y_val = val["source_label"].astype(int).to_numpy()

    model = tf.keras.models.load_model(MODEL_PATH)
    scores = model.predict(X_val, batch_size=32, verbose=0).ravel()

    rows = []
    for threshold in np.round(np.arange(0.10, 0.901, 0.01), 2):
        pred = (scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y_val, pred, labels=[0, 1]
        ).ravel()
        rows.append({
            "threshold": float(threshold),
            "accuracy": float(accuracy_score(y_val, pred)),
            "precision": float(precision_score(y_val, pred, zero_division=0)),
            "recall": float(recall_score(y_val, pred, zero_division=0)),
            "f1": float(f1_score(y_val, pred, zero_division=0)),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        })

    results = pd.DataFrame(rows)
    results.to_csv(OUT_DIR / "fedavg_validation_thresholds.csv", index=False)

    # Report several useful choices; do not use the test set here.
    best_f1 = results.loc[results["f1"].idxmax()]
    recall_80 = results[results["recall"] >= 0.80]
    print("Validation samples:", len(y_val))
    print("ROC-AUC:", round(float(roc_auc_score(y_val, scores)), 6))
    print("PR-AUC:", round(float(average_precision_score(y_val, scores)), 6))
    print("\nBest validation F1 threshold:")
    print(best_f1.to_string())
    print("\nLowest threshold achieving recall >= 0.80:")
    if recall_80.empty:
        print("No threshold reached 0.80 recall.")
    else:
        print(recall_80.sort_values("threshold").iloc[0].to_string())
    print("\nSaved full threshold table to:",
          (OUT_DIR / "fedavg_validation_thresholds.csv").resolve())


if __name__ == "__main__":
    main()
