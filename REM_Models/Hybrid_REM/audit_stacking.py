
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path.cwd()
OOF_DIR = ROOT / "REM_Models" / "Hybrid_REM" / "OOF_Predictions"
REPORT_DIR = ROOT / "REM_Models" / "Hybrid_REM" / "Reports"

rfif = pd.read_csv(OOF_DIR / "train_oof_rf_if.csv")
lstm = pd.read_csv(OOF_DIR / "train_oof_lstm.csv")

oof = rfif.merge(
    lstm,
    on=["sha256", "sequence_hash", "label", "fold"],
    how="inner",
    validate="one_to_one",
)

if len(oof) != len(rfif) or len(oof) != len(lstm):
    raise ValueError("OOF files do not align one-to-one.")

if oof["sequence_hash"].duplicated().any():
    raise ValueError("Duplicate sequence hashes in OOF data.")

features = [
    "rf_oof_score",
    "lstm_oof_score",
    "if_oof_anomaly_score",
]

if oof[features].isna().any().any():
    raise ValueError("OOF scores contain missing values.")

# Inspect score distributions by fold and class.
summary = oof.groupby(["fold", "label"])[features].agg(
    ["count", "mean", "std", "min", "median", "max"]
)
summary.to_csv(REPORT_DIR / "oof_fold_score_summary.csv")

# OOF discrimination for each base score.
auc_rows = []
for feature in features:
    auc_rows.append({
        "feature": feature,
        "roc_auc": float(roc_auc_score(oof["label"], oof[feature])),
    })

pd.DataFrame(auc_rows).to_csv(
    REPORT_DIR / "oof_base_score_auc.csv", index=False
)

# Compare OOF training score distributions against the already-generated
# full-training validation/test score files.
hybrid_val_path = REPORT_DIR / "validation_hybrid_predictions.csv"
hybrid_test_path = REPORT_DIR / "test_hybrid_predictions.csv"

distribution_rows = []
score_map = {
    "rf_oof_score": "rf_score",
    "lstm_oof_score": "lstm_score",
    "if_oof_anomaly_score": "if_anomaly_score",
}

for oof_col, pred_col in score_map.items():
    row = {"base_score": oof_col}

    row["oof_mean"] = float(oof[oof_col].mean())
    row["oof_std"] = float(oof[oof_col].std())

    for split, path in [
        ("validation", hybrid_val_path),
        ("test", hybrid_test_path),
    ]:
        if not path.exists():
            continue

        part = pd.read_csv(path)
        if pred_col not in part.columns:
            continue

        row[f"{split}_mean"] = float(part[pred_col].mean())
        row[f"{split}_std"] = float(part[pred_col].std())
        row[f"{split}_min"] = float(part[pred_col].min())
        row[f"{split}_max"] = float(part[pred_col].max())

    distribution_rows.append(row)

pd.DataFrame(distribution_rows).to_csv(
    REPORT_DIR / "oof_vs_full_model_score_distributions.csv",
    index=False,
)

result = {
    "oof_rows": int(len(oof)),
    "fold_counts": {
        str(k): int(v)
        for k, v in oof["fold"].value_counts().sort_index().items()
    },
    "label_counts": {
        str(k): int(v)
        for k, v in oof["label"].value_counts().sort_index().items()
    },
    "matched_rows": int(len(oof)),
    "unique_sequences": int(oof["sequence_hash"].nunique()),
    "missing_scores": int(oof[features].isna().sum().sum()),
    "base_oof_auc": auc_rows,
    "note": (
        "This script audits score alignment and distributions. "
        "It cannot prove fold-level training exclusion without inspecting "
        "the OOF generation procedure."
    ),
}

with open(
    REPORT_DIR / "stacking_audit.json", "w", encoding="utf-8"
) as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))
print("\nSaved audit files to:", REPORT_DIR)
