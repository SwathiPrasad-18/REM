import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)
from tensorflow.keras.preprocessing.sequence import pad_sequences

DATA = Path("REM_Datasets/04_Unified_Dataset/processed/public_labeled_candidate.csv")
MANIFEST = Path("REM_Models/API_Sequence_Grouped/grouped_split_manifest.csv")
VOCAB = Path("REM_Models/LSTM_Baseline/vocabulary.json")
CENTRAL = Path("REM_Models/LSTM_Baseline/rem_lstm.keras")
FEDERATED = Path("REM_Experiments/Federated_Learning/artifacts/fedavg_lstm_final.keras")
OUT = Path("REM_Experiments/Federated_Learning")
API_COLS = [str(i) for i in range(153)]


def make_sequence(row):
    ignored = {"", "nan", "none", "null"}
    return [
        token for value in row
        if (token := str(value).strip()).lower() not in ignored
    ]


def metrics(y, scores, threshold):
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def main():
    for p in [DATA, MANIFEST, VOCAB, CENTRAL, FEDERATED]:
        if not p.exists():
            raise FileNotFoundError(p)

    df = pd.read_csv(DATA, low_memory=False)
    manifest = pd.read_csv(MANIFEST)
    vocab = json.loads(VOCAB.read_text(encoding="utf-8"))

    df["sha256"] = df["sha256"].astype(str).str.strip().str.lower()
    manifest["sha256"] = manifest["sha256"].astype(str).str.strip().str.lower()
    df["api_tokens"] = df[API_COLS].apply(make_sequence, axis=1)
    df["sequence_hash"] = df["api_tokens"].map(
        lambda tokens: hashlib.sha256(
            " ".join(tokens).encode("utf-8")
        ).hexdigest()
    )

    label_counts = df.groupby("sequence_hash")["source_label"].nunique()
    conflicts = set(label_counts[label_counts > 1].index)
    df = df[~df["sequence_hash"].isin(conflicts)].copy()
    df = df.merge(
        manifest[["sha256", "split"]],
        on="sha256", how="inner", validate="one_to_one",
    )
    df = df.drop_duplicates("sequence_hash")
    val = df[df["split"] == "validation"].copy()

    if len(val) != 239:
        raise ValueError(f"Expected 239 validation rows; found {len(val)}")

    X = pad_sequences(
        [[int(vocab.get(token, 1)) for token in tokens]
         for tokens in val["api_tokens"]],
        maxlen=153, padding="post", truncating="post",
    )
    y = val["source_label"].astype(int).to_numpy()

    central_model = tf.keras.models.load_model(CENTRAL)
    federated_model = tf.keras.models.load_model(FEDERATED)
    central_scores = central_model.predict(X, batch_size=32, verbose=0).ravel()
    federated_scores = federated_model.predict(X, batch_size=32, verbose=0).ravel()

    rows = []
    for model_name, scores in [
        ("centralized_lstm", central_scores),
        ("federated_fedavg_lstm", federated_scores),
    ]:
        threshold_rows = [
            metrics(y, scores, float(t))
            for t in np.round(np.arange(0.10, 0.901, 0.01), 2)
        ]
        table = pd.DataFrame(threshold_rows)
        best_f1 = table.loc[table["f1"].idxmax()].to_dict()

        eligible = table[table["precision"] >= 0.80]
        recall_first = None
        if not eligible.empty:
            recall_first = eligible.sort_values(
                ["recall", "precision", "threshold"],
                ascending=[False, False, True],
            ).iloc[0].to_dict()

        rows.append({
            "model": model_name,
            "validation_samples": len(y),
            "roc_auc": float(roc_auc_score(y, scores)),
            "pr_auc": float(average_precision_score(y, scores)),
            "max_f1_operating_point": best_f1,
            "recall_first_precision_at_least_0.80": recall_first,
        })

    out_json = OUT / "centralized_vs_fedavg_validation.json"
    out_csv = OUT / "centralized_vs_fedavg_validation.csv"
    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    flat = []
    for row in rows:
        for policy in ["max_f1_operating_point",
                       "recall_first_precision_at_least_0.80"]:
            result = row[policy]
            flat.append({
                "model": row["model"],
                "policy": policy,
                "roc_auc": row["roc_auc"],
                "pr_auc": row["pr_auc"],
                **(result or {}),
            })
    pd.DataFrame(flat).to_csv(out_csv, index=False)

    print("Same validation set used for both models:", len(y), "samples")
    print("Test set evaluated: False")
    for row in rows:
        print("\nMODEL:", row["model"])
        print("ROC-AUC:", round(row["roc_auc"], 6))
        print("PR-AUC:", round(row["pr_auc"], 6))
        print("Maximum-F1 operating point:")
        print(pd.Series(row["max_f1_operating_point"]).to_string())
        print("Recall-first, precision >= 0.80:")
        print(
            pd.Series(row["recall_first_precision_at_least_0.80"]).to_string()
            if row["recall_first_precision_at_least_0.80"] is not None
            else "No threshold satisfies the precision constraint."
        )

    print("\nSaved:", out_json.resolve())
    print("Saved:", out_csv.resolve())


if __name__ == "__main__":
    main()
