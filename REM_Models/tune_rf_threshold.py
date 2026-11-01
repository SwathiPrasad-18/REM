
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--min-precision", type=float, default=0.80)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data, low_memory=False)
    manifest = pd.read_csv(args.split_manifest)
    model = joblib.load(args.model)

    # Use validation samples only. Never tune on the test set.
    validation_ids = manifest.loc[
        manifest["split"] == "validation", "sha256"
    ].astype(str)

    if validation_ids.empty:
        raise ValueError("No validation samples found in the manifest.")

    df["sha256"] = df["sha256"].astype(str)
    val = df[df["sha256"].isin(set(validation_ids))].copy()

    if val.empty:
        raise ValueError("No validation samples matched by SHA256.")

    # Reconstruct API sequences exactly as in the training script.
    api_cols = [str(i) for i in range(153)]
    missing = [c for c in api_cols if c not in val.columns]
    if missing:
        raise ValueError(f"Missing API columns: {missing}")

    ignored = {"", "nan", "none", "null"}

    def make_sequence(row):
        tokens = []
        for value in row:
            token = str(value).strip()
            if token.lower() not in ignored:
                tokens.append(token)
        return " ".join(tokens)

    X = val[api_cols].apply(make_sequence, axis=1)
    y = pd.to_numeric(val["source_label"]).astype(int).to_numpy()

    # The pipeline returns probabilities for class 0 and class 1.
    probabilities = model.predict_proba(X)
    positive_index = list(model.classes_).index(1)
    scores = probabilities[:, positive_index]

    rows = []
    thresholds = np.round(np.arange(0.10, 0.901, 0.01), 2)

    for threshold in thresholds:
        pred = (scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y, pred, labels=[0, 1]
        ).ravel()

        rows.append({
            "threshold": float(threshold),
            "accuracy": float(accuracy_score(y, pred)),
            "precision_malicious": float(
                precision_score(y, pred, zero_division=0)
            ),
            "recall_malicious": float(
                recall_score(y, pred, zero_division=0)
            ),
            "f1_malicious": float(
                f1_score(y, pred, zero_division=0)
            ),
            "false_positive_rate": float(
                fp / (fp + tn) if (fp + tn) else 0
            ),
            "false_negative_rate": float(
                fn / (fn + tp) if (fn + tp) else 0
            ),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        })

    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "validation_threshold_metrics.csv", index=False)

    # Choose the threshold with the highest recall among those meeting
    # the required precision. Break ties using F1, then higher threshold.
    eligible = results[
        results["precision_malicious"] >= args.min_precision
    ].copy()

    if eligible.empty:
        print(
            "WARNING: No threshold met the requested minimum precision. "
            "Selecting the threshold with the highest validation F1."
        )
        chosen = results.sort_values(
            ["f1_malicious", "recall_malicious", "threshold"],
            ascending=[False, False, False],
        ).iloc[0]
        selection_rule = "highest validation F1; precision target unmet"
    else:
        chosen = eligible.sort_values(
            ["recall_malicious", "f1_malicious", "threshold"],
            ascending=[False, False, False],
        ).iloc[0]
        selection_rule = (
            "highest validation recall subject to minimum precision"
        )

    summary = {
        "validation_samples": int(len(val)),
        "minimum_precision_target": args.min_precision,
        "selection_rule": selection_rule,
        "selected_threshold": float(chosen["threshold"]),
        "selected_validation_metrics": {
            key: (
                float(chosen[key])
                if isinstance(chosen[key], (np.integer, np.floating))
                else chosen[key]
            )
            for key in [
                "accuracy",
                "precision_malicious",
                "recall_malicious",
                "f1_malicious",
                "false_positive_rate",
                "false_negative_rate",
                "true_negatives",
                "false_positives",
                "false_negatives",
                "true_positives",
            ]
        },
        "validation_roc_auc": float(roc_auc_score(y, scores)),
        "warning": (
            "Threshold selected on validation data only. "
            "Do not use these metrics as final test performance."
        ),
    }

    with open(out_dir / "threshold_selection.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nSaved results to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
