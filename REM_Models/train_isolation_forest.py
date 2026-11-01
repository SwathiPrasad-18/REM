
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


SEED = 42


def clean_tokens(row):
    ignored = {"", "nan", "none", "null"}
    return [
        token
        for value in row
        if (token := str(value).strip()).lower() not in ignored
    ]


def extract_features(tokens):
    # Frequency of each API plus sequence-level statistics.
    features = {}
    for token in tokens:
        key = "api_freq=" + token
        features[key] = features.get(key, 0) + 1

    n = len(tokens)
    features["sequence_length"] = n
    features["unique_api_count"] = len(set(tokens))
    features["unique_api_ratio"] = (
        len(set(tokens)) / n if n else 0.0
    )

    # Adjacent API-call transitions retain some ordering information.
    for a, b in zip(tokens, tokens[1:]):
        key = "transition=" + a + "->" + b
        features[key] = features.get(key, 0) + 1

    return features


def main():
    data_path = Path(
        "REM_Datasets/04_Unified_Dataset/processed/"
        "public_labeled_candidate.csv"
    )
    manifest_path = Path(
        "REM_Models/API_Sequence_Grouped/grouped_split_manifest.csv"
    )
    out_dir = Path("REM_Models/Isolation_Forest_Baseline")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path, low_memory=False)
    manifest = pd.read_csv(manifest_path)

    api_cols = [str(i) for i in range(153)]
    if any(c not in df.columns for c in api_cols):
        raise ValueError("One or more API columns are missing.")

    df["sha256"] = df["sha256"].astype(str)
    manifest["sha256"] = manifest["sha256"].astype(str)
    df["source_label"] = pd.to_numeric(
        df["source_label"], errors="raise"
    ).astype(int)

    df["tokens"] = df[api_cols].apply(clean_tokens, axis=1)
    df["sequence_hash"] = df["tokens"].map(
        lambda t: hashlib.sha256(
            " ".join(t).encode("utf-8")
        ).hexdigest()
    )

    # Quarantine sequences with conflicting labels.
    label_counts = df.groupby("sequence_hash")["source_label"].nunique()
    conflict_hashes = set(label_counts[label_counts > 1].index)

    df[df["sequence_hash"].isin(conflict_hashes)].drop(
        columns=["tokens"]
    ).to_csv(out_dir / "quarantined_conflicts.csv", index=False)

    df = df[~df["sequence_hash"].isin(conflict_hashes)].copy()

    # Reuse existing split assignments.
    df = df.merge(
        manifest[["sha256", "split"]],
        on="sha256",
        how="inner",
        validate="one_to_one",
    )

    # One observation per unique API sequence.
    df = df.drop_duplicates("sequence_hash").copy()

    train = df[df["split"] == "train"].copy()
    val = df[df["split"] == "validation"].copy()
    test = df[df["split"] == "test"].copy()

    # Unsupervised feature representation.
    vectorizer = DictVectorizer(sparse=True)
    train_dicts = train["tokens"].map(extract_features).tolist()
    X_train_all = vectorizer.fit_transform(train_dicts)

    val_dicts = val["tokens"].map(extract_features).tolist()
    test_dicts = test["tokens"].map(extract_features).tolist()
    X_val = vectorizer.transform(val_dicts)
    X_test = vectorizer.transform(test_dicts)

    # Fit only on presumed-benign training samples.
    benign_mask = train["source_label"].to_numpy() == 0
    if benign_mask.sum() < 10:
        raise ValueError("Too few benign training samples.")

    model = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(X_train_all[benign_mask])

    # Larger anomaly score means more anomalous.
    val_scores = -model.score_samples(X_val)
    test_scores = -model.score_samples(X_test)

    # Select threshold using validation data only.
    val_y = val["source_label"].to_numpy()
    candidates = np.unique(
        np.quantile(val_scores, np.linspace(0, 1, 201))
    )

    threshold_rows = []
    for threshold in candidates:
        pred = (val_scores >= threshold).astype(int)
        threshold_rows.append({
            "threshold": float(threshold),
            "precision_malicious": float(
                precision_score(val_y, pred, zero_division=0)
            ),
            "recall_malicious": float(
                recall_score(val_y, pred, zero_division=0)
            ),
            "f1_malicious": float(
                f1_score(val_y, pred, zero_division=0)
            ),
        })

    threshold_df = pd.DataFrame(threshold_rows)
    threshold_df.to_csv(
        out_dir / "validation_threshold_metrics.csv",
        index=False,
    )

    eligible = threshold_df[
        threshold_df["precision_malicious"] >= 0.80
    ]
    if eligible.empty:
        chosen = threshold_df.sort_values(
            ["f1_malicious", "recall_malicious", "threshold"],
            ascending=[False, False, False],
        ).iloc[0]
        selection_rule = "highest validation F1; precision target unmet"
    else:
        chosen = eligible.sort_values(
            ["recall_malicious", "f1_malicious", "threshold"],
            ascending=[False, False, False],
        ).iloc[0]
        selection_rule = "highest recall with precision >= 0.80"

    threshold = float(chosen["threshold"])
    test_y = test["source_label"].to_numpy()
    test_pred = (test_scores >= threshold).astype(int)

    results = {
        "train_samples": int(len(train)),
        "benign_training_samples_used_for_fit": int(benign_mask.sum()),
        "validation_samples": int(len(val)),
        "test_samples": int(len(test)),
        "conflicting_sequence_groups_quarantined": int(
            len(conflict_hashes)
        ),
        "selected_threshold": threshold,
        "selection_rule": selection_rule,
        "validation_metrics": {
            "precision": float(chosen["precision_malicious"]),
            "recall": float(chosen["recall_malicious"]),
            "f1": float(chosen["f1_malicious"]),
        },
        "test": {
            "accuracy": float(accuracy_score(test_y, test_pred)),
            "roc_auc": float(roc_auc_score(test_y, test_scores)),
            "confusion_matrix_labels_0_1": confusion_matrix(
                test_y, test_pred, labels=[0, 1]
            ).tolist(),
            "classification_report": classification_report(
                test_y,
                test_pred,
                labels=[0, 1],
                target_names=["benign", "malicious"],
                output_dict=True,
                zero_division=0,
            ),
        },
        "warning": (
            "Anomaly scores are not calibrated malware probabilities. "
            "Threshold selected on validation data only."
        ),
    }

    joblib.dump(
        {"vectorizer": vectorizer, "model": model},
        out_dir / "isolation_forest.joblib",
    )

    pd.DataFrame({
        "sha256": test["sha256"].to_numpy(),
        "actual_label": test_y,
        "anomaly_score": test_scores,
        "predicted_label": test_pred,
    }).to_csv(out_dir / "test_predictions.csv", index=False)

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"\nSaved outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
