
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


def make_sequence(row):
    ignored = {"", "nan", "none", "null"}
    return " ".join(
        token for value in row
        if (token := str(value).strip()).lower() not in ignored
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--threshold", type=float, default=0.38)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data, low_memory=False)
    manifest = pd.read_csv(args.split_manifest)

    test_ids = set(
        manifest.loc[manifest["split"] == "test", "sha256"].astype(str)
    )
    df["sha256"] = df["sha256"].astype(str)
    test = df[df["sha256"].isin(test_ids)].copy()

    if test.empty:
        raise ValueError("No test samples matched the split manifest.")

    api_cols = [str(i) for i in range(153)]
    test["source_label"] = pd.to_numeric(
        test["source_label"], errors="raise"
    ).astype(int)

    # Deduplicate by sequence, consistent with grouped evaluation.
    test["api_sequence"] = test[api_cols].apply(make_sequence, axis=1)
    test["sequence_hash"] = test["api_sequence"].map(
        lambda s: __import__("hashlib").sha256(
            s.encode("utf-8")
        ).hexdigest()
    )
    test = test.drop_duplicates("sequence_hash").copy()

    model = joblib.load(args.model)
    y = test["source_label"].to_numpy()
    scores = model.predict_proba(test["api_sequence"])
    positive_index = list(model.classes_).index(1)
    scores = scores[:, positive_index]
    pred = (scores >= args.threshold).astype(int)

    results = {
        "threshold": args.threshold,
        "test_samples": len(test),
        "accuracy": float(accuracy_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "confusion_matrix_labels_0_1": confusion_matrix(
            y, pred, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            y, pred,
            labels=[0, 1],
            target_names=["benign", "malicious"],
            output_dict=True,
            zero_division=0,
        ),
        "warning": "Held-out test evaluation; do not tune on these results.",
    }

    with open(out / "test_threshold_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
