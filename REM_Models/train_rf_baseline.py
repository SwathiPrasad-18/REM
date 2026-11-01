#!/usr/bin/env python3
"""REM Random Forest baseline with SHA-256-grouped splits.

Label mapping: 0 = benign, 1 = malicious.
Trains only on the public dataset's documented labels. Provisional VirusTotal
labels are not used as ground truth.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

IDENTIFIER_COLS = {
    "sha256", "sha1", "md5", "filename", "file_name", "report_file",
    "sample", "sample_id", "hash", "label_source", "label_status",
    "eligible_for_supervised_training", "hash_label_conflict",
}
TARGET_COLS = {
    "labels", "label", "source_label", "ground_truth_label",
    "proposed_label", "vt_proposed_label", "malware", "class",
}
BAD_NAME_PARTS = (
    "sha256", "sha1", "md5", "filename", "file_name", "path",
    "report_file", "label", "target", "verdict", "hash", "sample_id",
)

def main():
    parser = argparse.ArgumentParser(description="Train REM Random Forest baseline.")
    parser.add_argument("--data", required=True, help="Path to public_labeled_candidate.csv")
    parser.add_argument("--out-dir", default="REM_Models/RF_Baseline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--val-size", type=float, default=0.15,
                        help="Fraction of the full dataset allocated to validation.")
    parser.add_argument("--trees", type=int, default=300)
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {data_path}")

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path, low_memory=False)
    label_col = "source_label" if "source_label" in df.columns else (
        "labels" if "labels" in df.columns else None
    )
    if label_col is None:
        raise ValueError("Expected label column 'source_label' or 'labels'.")
    if "sha256" not in df.columns:
        raise ValueError("A sha256 column is required for group-based splitting.")

    df["sha256"] = df["sha256"].astype(str).str.strip().str.lower()
    labels = pd.to_numeric(df[label_col], errors="coerce")
    valid = labels.isin([0, 1]) & df["sha256"].str.fullmatch(r"[0-9a-f]{64}", na=False)
    df = df.loc[valid].copy().reset_index(drop=True)
    y = pd.to_numeric(df[label_col]).astype(int).reset_index(drop=True)
    groups = df["sha256"].reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("No rows with valid SHA-256 and binary labels 0/1 were found.")
    if y.nunique() != 2:
        raise ValueError(f"Expected both classes 0 and 1; found {sorted(y.unique().tolist())}.")

    feature_cols = []
    for col in df.columns:
        low = col.lower()
        if col == label_col or col in IDENTIFIER_COLS or col in TARGET_COLS:
            continue
        if any(part in low for part in BAD_NAME_PARTS):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)

    if not feature_cols:
        raise ValueError("No numeric feature columns remain after excluding identifiers and label-like fields.")

    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).reset_index(drop=True)

    # First split: test set by SHA-256 group.
    splitter_test = GroupShuffleSplit(
        n_splits=1, test_size=args.test_size, random_state=args.seed
    )
    trainval_idx, test_idx = next(splitter_test.split(X, y, groups))

    X_trainval = X.iloc[trainval_idx].reset_index(drop=True)
    y_trainval = y.iloc[trainval_idx].reset_index(drop=True)
    groups_trainval = groups.iloc[trainval_idx].reset_index(drop=True)

    relative_val_size = args.val_size / (1.0 - args.test_size)
    if not 0 < relative_val_size < 1:
        raise ValueError("--val-size must be > 0 and less than (1 - test-size).")

    # Second split: validation set from train+validation, grouped by SHA-256.
    splitter_val = GroupShuffleSplit(
        n_splits=1, test_size=relative_val_size, random_state=args.seed + 1
    )
    train_idx, val_idx = next(
        splitter_val.split(X_trainval, y_trainval, groups_trainval)
    )

    X_train = X_trainval.iloc[train_idx]
    y_train = y_trainval.iloc[train_idx]
    X_val = X_trainval.iloc[val_idx]
    y_val = y_trainval.iloc[val_idx]
    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    for split_name, split_y in (
        ("train", y_train), ("validation", y_val), ("test", y_test)
    ):
        if set(split_y.unique()) != {0, 1}:
            raise ValueError(
                f"{split_name} split does not contain both classes; "
                f"found {sorted(split_y.unique().tolist())}. Try another --seed."
            )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("classifier", RandomForestClassifier(
            n_estimators=args.trees,
            class_weight="balanced_subsample",
            random_state=args.seed,
            n_jobs=-1,
        )),
    ])
    model.fit(X_train, y_train)

    def evaluate(features, actual, split_name):
        predicted = model.predict(features)
        classes = list(model.named_steps["classifier"].classes_)
        malicious_index = classes.index(1)
        probabilities = model.predict_proba(features)[:, malicious_index]

        precision, recall, f1, support = precision_recall_fscore_support(
            actual, predicted, labels=[0, 1], zero_division=0
        )
        per_class = {
            str(label): {
                "name": "benign" if label == 0 else "malicious",
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate([0, 1])
        }
        return {
            "split": split_name,
            "rows": int(len(actual)),
            "class_counts": {
                str(int(k)): int(v)
                for k, v in actual.value_counts().sort_index().items()
            },
            "accuracy": float(accuracy_score(actual, predicted)),
            "precision_recall_f1_support": per_class,
            "confusion_matrix_labels_0_1": confusion_matrix(
                actual, predicted, labels=[0, 1]
            ).tolist(),
            "classification_report": classification_report(
                actual, predicted,
                labels=[0, 1],
                target_names=["benign_0", "malicious_1"],
                output_dict=True,
                zero_division=0,
            ),
            "roc_auc": float(roc_auc_score(actual, probabilities)),
        }

    # Assert that hash groups do not overlap between splits.
    train_hashes = set(groups_trainval.iloc[train_idx])
    val_hashes = set(groups_trainval.iloc[val_idx])
    test_hashes = set(groups.iloc[test_idx])
    if train_hashes & val_hashes or train_hashes & test_hashes or val_hashes & test_hashes:
        raise RuntimeError("SHA-256 leakage detected between splits.")

    metrics = {
        "label_mapping": {"0": "benign", "1": "malicious"},
        "rows_used": int(len(df)),
        "feature_count": int(len(feature_cols)),
        "features_used": feature_cols,
        "split_method": "GroupShuffleSplit grouped by SHA-256",
        "split_sizes": {
            "train": int(len(train_idx)),
            "validation": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "sha256_overlap": {
            "train_validation": 0,
            "train_test": 0,
            "validation_test": 0,
        },
        "metrics": [
            evaluate(X_train, y_train, "train"),
            evaluate(X_val, y_val, "validation"),
            evaluate(X_test, y_test, "test"),
        ],
        "caution": (
            "SHA-256 grouping prevents exact-hash leakage, but source/dataset "
            "leakage and highly correlated features still need review. "
            "Inspect feature importance before interpreting generalization."
        ),
    }

    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (output_dir / "feature_columns.json").write_text(
        json.dumps(feature_cols, indent=2), encoding="utf-8"
    )
    joblib.dump(model, output_dir / "random_forest_pipeline.joblib")

    split_records = []
    for split_name, hash_series, label_series in (
        ("train", groups_trainval.iloc[train_idx], y_train),
        ("validation", groups_trainval.iloc[val_idx], y_val),
        ("test", groups.iloc[test_idx], y_test),
    ):
        for sample_hash, label in zip(hash_series, label_series):
            split_records.append({
                "sha256": sample_hash,
                "split": split_name,
                "source_label": int(label),
            })
    pd.DataFrame(split_records).to_csv(
        output_dir / "split_manifest.csv", index=False
    )

    # Random Forest importances include original features and missingness indicators.
    classifier = model.named_steps["classifier"]
    importance_values = classifier.feature_importances_
    importance_names = feature_cols + [f"{c}__missing" for c in feature_cols]
    if len(importance_names) != len(importance_values):
        importance_names = [
            f"transformed_feature_{i}" for i in range(len(importance_values))
        ]
    pd.DataFrame({
        "feature": importance_names,
        "importance": importance_values,
    }).sort_values("importance", ascending=False).to_csv(
        output_dir / "feature_importance.csv", index=False
    )

    print(json.dumps({
        "status": "success",
        "rows_used": len(df),
        "feature_count": len(feature_cols),
        "split_sizes": metrics["split_sizes"],
        "metrics_path": str((output_dir / "metrics.json").resolve()),
        "model_path": str((output_dir / "random_forest_pipeline.joblib").resolve()),
        "output_dir": str(output_dir.resolve()),
    }, indent=2))

if __name__ == "__main__":
    main()
