
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline


def make_sequence(row):
    ignored = {"", "nan", "none", "null"}
    tokens = []
    for value in row:
        token = str(value).strip()
        if token.lower() not in ignored:
            tokens.append(token)
    return " ".join(tokens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    data_path = Path(args.data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path, low_memory=False)
    api_cols = [str(i) for i in range(153)]

    missing = [c for c in api_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing API columns: {missing}")

    df = df.dropna(subset=["source_label"]).copy()
    df["source_label"] = pd.to_numeric(
        df["source_label"], errors="raise"
    ).astype(int)

    if not set(df["source_label"].unique()).issubset({0, 1}):
        raise ValueError("Expected labels 0=benign and 1=malicious.")

    df["api_sequence"] = df[api_cols].apply(make_sequence, axis=1)
    df["sequence_hash"] = df["api_sequence"].map(
        lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest()
    )

    # Exclude any sequence associated with more than one label.
    label_counts = df.groupby("sequence_hash")["source_label"].nunique()
    conflicting_hashes = label_counts[label_counts > 1].index
    conflict_rows = df[df["sequence_hash"].isin(conflicting_hashes)].copy()

    conflict_rows.drop(columns=["api_sequence"]).to_csv(
        out_dir / "quarantined_conflicting_sequences.csv",
        index=False,
    )

    clean = df[~df["sequence_hash"].isin(conflicting_hashes)].copy()

    # Each unique sequence is one group. Identical sequences cannot cross splits.
    groups = clean["sequence_hash"].to_numpy()
    y = clean["source_label"].to_numpy()
    indices = np.arange(len(clean))

    # First: 20% held-out test groups.
    split1 = GroupShuffleSplit(
        n_splits=1, test_size=0.20, random_state=42
    )
    trainval_idx, test_idx = next(
        split1.split(indices, y, groups)
    )

    # Second: approximately 15% of total data for validation.
    trainval = clean.iloc[trainval_idx].copy()
    split2 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.1875,
        random_state=43,
    )
    train_idx2, val_idx2 = next(
        split2.split(
            np.arange(len(trainval)),
            trainval["source_label"],
            trainval["sequence_hash"],
        )
    )

    train = trainval.iloc[train_idx2].copy()
    val = trainval.iloc[val_idx2].copy()
    test = clean.iloc[test_idx].copy()

    # Confirm no sequence overlap between splits.
    train_groups = set(train["sequence_hash"])
    val_groups = set(val["sequence_hash"])
    test_groups = set(test["sequence_hash"])

    assert not (train_groups & val_groups)
    assert not (train_groups & test_groups)
    assert not (val_groups & test_groups)

    # Deduplicate within each split: one record per sequence.
    train = train.drop_duplicates("sequence_hash").copy()
    val = val.drop_duplicates("sequence_hash").copy()
    test = test.drop_duplicates("sequence_hash").copy()

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=False,
            token_pattern=r"(?u)\b[\w.-]+\b",
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )),
        ("rf", RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    model.fit(train["api_sequence"], train["source_label"])

    results = {}
    for name, part in [("validation", val), ("test", test)]:
        X = part["api_sequence"]
        y_true = part["source_label"].to_numpy()
        pred = model.predict(X)
        proba = model.predict_proba(X)
        positive_index = list(model.classes_).index(1)
        scores = proba[:, positive_index]

        results[name] = {
            "samples": int(len(part)),
            "unique_sequences": int(part["sequence_hash"].nunique()),
            "accuracy": float(accuracy_score(y_true, pred)),
            "roc_auc": (
                float(roc_auc_score(y_true, scores))
                if len(np.unique(y_true)) == 2 else None
            ),
            "confusion_matrix_labels_0_1": confusion_matrix(
                y_true, pred, labels=[0, 1]
            ).tolist(),
            "classification_report": classification_report(
                y_true, pred, labels=[0, 1],
                target_names=["benign", "malicious"],
                output_dict=True,
                zero_division=0,
            ),
        }

    # Save a reproducible manifest.
    manifest_parts = []
    for name, part in [
        ("train", train), ("validation", val), ("test", test)
    ]:
        temp = part[["sha256", "source_label", "sequence_hash"]].copy()
        temp["split"] = name
        manifest_parts.append(temp)

    pd.concat(manifest_parts, ignore_index=True).to_csv(
        out_dir / "grouped_split_manifest.csv", index=False
    )

    summary = {
        "input_rows": int(len(df)),
        "unique_api_sequences_before_quarantine": int(
            df["sequence_hash"].nunique()
        ),
        "conflicting_sequence_groups": int(len(conflicting_hashes)),
        "quarantined_rows": int(len(conflict_rows)),
        "remaining_rows": int(len(clean)),
        "unique_sequences_after_quarantine": int(
            clean["sequence_hash"].nunique()
        ),
        "train_rows_unique_sequences": int(len(train)),
        "validation_rows_unique_sequences": int(len(val)),
        "test_rows_unique_sequences": int(len(test)),
        "overlap_checks": {
            "train_validation": len(train_groups & val_groups),
            "train_test": len(train_groups & test_groups),
            "validation_test": len(val_groups & test_groups),
        },
        "results": results,
    }

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    import joblib
    joblib.dump(model, out_dir / "grouped_api_sequence_rf.joblib")

    print(json.dumps({
        "input_rows": summary["input_rows"],
        "conflicting_sequence_groups": summary[
            "conflicting_sequence_groups"
        ],
        "quarantined_rows": summary["quarantined_rows"],
        "unique_sequences_after_quarantine": summary[
            "unique_sequences_after_quarantine"
        ],
        "train": summary["train_rows_unique_sequences"],
        "validation": summary["validation_rows_unique_sequences"],
        "test": summary["test_rows_unique_sequences"],
        "overlap_checks": summary["overlap_checks"],
        "results": results,
        "output_directory": str(out_dir.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
