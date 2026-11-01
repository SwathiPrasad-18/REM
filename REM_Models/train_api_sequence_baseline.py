#!/usr/bin/env python3
"""
REM API-Sequence Random Forest baseline.

Input:
  processed/public_labeled_candidate.csv
Uses only API-call sequence columns "0" through "152" and source_label.
Does NOT use parser-derived features, SHA256, or label metadata as model inputs.

Labels: 0 = benign, 1 = malicious (confirm this mapping matches your dataset documentation).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def make_sequence(row: pd.Series, api_columns: list[str]) -> str:
    """Preserve the original left-to-right API call order; ignore empty cells."""
    tokens = []
    for value in row[api_columns].tolist():
        if pd.isna(value):
            continue
        token = str(value).strip()
        if not token or token.lower() in {"nan", "none", "null"}:
            continue
        # API function names should be tokens, not arbitrary whitespace.
        token = re.sub(r"\s+", "_", token)
        tokens.append(token)
    return " ".join(tokens)


def evaluate_split(name, model, texts, y, indices):
    pred = model.predict(texts)
    probabilities = model.predict_proba(texts)
    class_order = list(model.named_steps["classifier"].classes_)
    malicious_column = class_order.index(1) if 1 in class_order else None
    result = {
        "split": name,
        "rows": int(len(y)),
        "class_counts": {str(int(k)): int(v) for k, v in pd.Series(y).value_counts().sort_index().items()},
        "accuracy": float(accuracy_score(y, pred)),
        "confusion_matrix_labels_0_1": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y, pred, labels=[0, 1], target_names=["benign_0", "malicious_1"],
            output_dict=True, zero_division=0
        ),
    }
    p, r, f1, support = precision_recall_fscore_support(
        y, pred, labels=[0, 1], zero_division=0
    )
    result["precision_recall_f1_support"] = {
        str(label): {
            "name": "benign" if label == 0 else "malicious",
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, label in enumerate([0, 1])
    }
    if malicious_column is not None and len(np.unique(y)) == 2:
        result["roc_auc"] = float(roc_auc_score(y, probabilities[:, malicious_column]))
    else:
        result["roc_auc"] = None
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to public_labeled_candidate.csv")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--validation-size", type=float, default=0.15,
                        help="Fraction of all rows used for validation")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--trees", type=int, default=400)
    args = parser.parse_args()

    data_path = Path(args.data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {data_path}")
    if args.test_size <= 0 or args.validation_size <= 0 or args.test_size + args.validation_size >= 1:
        raise ValueError("test-size and validation-size must be > 0 and sum to < 1")

    df = pd.read_csv(data_path, low_memory=False)
    api_columns = [str(i) for i in range(153) if str(i) in df.columns]
    if len(api_columns) != 153:
        raise ValueError(
            f"Expected API-call columns '0' through '152'; found {len(api_columns)}. "
            "Check the input CSV."
        )
    if "source_label" not in df.columns:
        raise ValueError("Required target column 'source_label' is missing.")

    # Use documented source labels only; do not use label_status/metadata as features.
    labels = pd.to_numeric(df["source_label"], errors="coerce")
    valid = labels.isin([0, 1])
    df = df.loc[valid].copy().reset_index(drop=True)
    y = labels.loc[valid].astype(int).to_numpy()

    if len(df) < 20:
        raise ValueError(f"Only {len(df)} labeled rows found; too few for a meaningful split.")

    texts = df.apply(lambda row: make_sequence(row, api_columns), axis=1).tolist()
    empty_count = sum(not t for t in texts)
    if empty_count:
        raise ValueError(f"{empty_count} rows have empty API-call sequences; inspect input before training.")

    # 65/15/20 default split, stratified by label. SHA256 is not a model feature.
    test_fraction = args.test_size
    val_fraction_of_remaining = args.validation_size / (1.0 - test_fraction)
    all_idx = np.arange(len(df))
    trainval_idx, test_idx = train_test_split(
        all_idx, test_size=test_fraction, random_state=args.random_state, stratify=y
    )
    y_trainval = y[trainval_idx]
    train_idx, val_idx = train_test_split(
        trainval_idx, test_size=val_fraction_of_remaining,
        random_state=args.random_state, stratify=y_trainval
    )

    vectorizer = TfidfVectorizer(
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
        ngram_range=(1, 2),
        min_df=args.min_df,
        max_features=args.max_features,
        sublinear_tf=True,
    )
    classifier = RandomForestClassifier(
        n_estimators=args.trees,
        class_weight="balanced_subsample",
        random_state=args.random_state,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    model = Pipeline([("tfidf", vectorizer), ("classifier", classifier)])
    model.fit([texts[i] for i in train_idx], y[train_idx])

    metrics = {
        "status": "success",
        "dataset": str(data_path.resolve()),
        "rows_used": int(len(df)),
        "api_sequence_columns": int(len(api_columns)),
        "feature_representation": "ordered API-call tokens with unigram/bigram TF-IDF",
        "label_mapping": {"0": "benign", "1": "malicious"},
        "model": "TfidfVectorizer(ngram_range=(1,2)) + RandomForestClassifier",
        "random_state": int(args.random_state),
        "split_sizes": {
            "train": int(len(train_idx)),
            "validation": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "class_counts": {
            "0": int((y == 0).sum()),
            "1": int((y == 1).sum()),
        },
        "metrics": [
            evaluate_split("train", model, [texts[i] for i in train_idx], y[train_idx], train_idx),
            evaluate_split("validation", model, [texts[i] for i in val_idx], y[val_idx], val_idx),
            evaluate_split("test", model, [texts[i] for i in test_idx], y[test_idx], test_idx),
        ],
        "cautions": [
            "This is a baseline, not a production malware detector.",
            "The split is stratified by label but does not establish cross-dataset generalization.",
            "Confirm the dataset documentation supports label mapping 0=benign and 1=malicious.",
            "API-call sequence order is retained in text, with unigram and adjacent-bigram TF-IDF features.",
        ],
    }

    dump(model, out_dir / "api_sequence_rf_pipeline.joblib")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "feature_columns.json").write_text(json.dumps(api_columns, indent=2), encoding="utf-8")

    # Record row-level split assignments for reproducibility, without exposing labels as inputs.
    split_map = {}
    for name, idxs in [("train", train_idx), ("validation", val_idx), ("test", test_idx)]:
        for i in idxs:
            split_map[int(i)] = name
    split_manifest = pd.DataFrame({
        "row_index_in_filtered_dataset": np.arange(len(df)),
        "sha256": df["sha256"].astype(str).values if "sha256" in df.columns else "",
        "source_label": y,
        "split": [split_map[i] for i in range(len(df))],
        "api_call_count": [len(t.split()) for t in texts],
    })
    split_manifest.to_csv(out_dir / "split_manifest.csv", index=False)

    # TF-IDF feature importance can be large; save top 100 only.
    fitted_vectorizer = model.named_steps["tfidf"]
    fitted_classifier = model.named_steps["classifier"]
    feature_names = fitted_vectorizer.get_feature_names_out()
    importances = fitted_classifier.feature_importances_
    top_idx = np.argsort(importances)[::-1][:100]
    pd.DataFrame({
        "feature": feature_names[top_idx],
        "importance": importances[top_idx],
    }).to_csv(out_dir / "feature_importance.csv", index=False)

    print(json.dumps({
        "status": "success",
        "rows_used": len(df),
        "api_sequence_columns": len(api_columns),
        "split_sizes": metrics["split_sizes"],
        "metrics_path": str((out_dir / "metrics.json").resolve()),
        "model_path": str((out_dir / "api_sequence_rf_pipeline.joblib").resolve()),
        "output_dir": str(out_dir.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
