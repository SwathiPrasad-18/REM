
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

root = Path(".")
rf_path = (
    root / "REM_Models" / "API_Sequence_Grouped"
    / "Threshold_Tuning" / "test_threshold_metrics.json"
)
lstm_path = (
    root / "REM_Models" / "LSTM_Baseline" / "metrics.json"
)
out_dir = root / "REM_Models" / "Model_Comparison"
out_dir.mkdir(parents=True, exist_ok=True)

with open(rf_path, encoding="utf-8") as f:
    rf = json.load(f)

with open(lstm_path, encoding="utf-8") as f:
    lstm = json.load(f)

rf_report = rf["classification_report"]["malicious"]
lstm_report = lstm["test"]["classification_report"]["malicious"]

comparison = pd.DataFrame([
    {
        "model": "Random Forest",
        "threshold": rf["threshold"],
        "accuracy": rf["accuracy"],
        "roc_auc": rf["roc_auc"],
        "malware_precision": rf_report["precision"],
        "malware_recall": rf_report["recall"],
        "malware_f1": rf_report["f1-score"],
        "false_positives": rf["confusion_matrix_labels_0_1"][0][1],
        "false_negatives": rf["confusion_matrix_labels_0_1"][1][0],
    },
    {
        "model": "LSTM",
        "threshold": lstm["selected_threshold"],
        "accuracy": lstm["test"]["accuracy"],
        "roc_auc": lstm["test"]["roc_auc"],
        "malware_precision": lstm_report["precision"],
        "malware_recall": lstm_report["recall"],
        "malware_f1": lstm_report["f1-score"],
        "false_positives": lstm["test"]["confusion_matrix_labels_0_1"][0][1],
        "false_negatives": lstm["test"]["confusion_matrix_labels_0_1"][1][0],
    },
])

comparison.to_csv(out_dir / "model_comparison.csv", index=False)

metrics = [
    ("accuracy", "Accuracy"),
    ("roc_auc", "ROC-AUC"),
    ("malware_precision", "Malware precision"),
    ("malware_recall", "Malware recall"),
    ("malware_f1", "Malware F1"),
]

for column, title in metrics:
    plt.figure(figsize=(7, 4))
    plt.bar(comparison["model"], comparison[column])
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_dir / f"{column}.png", dpi=160)
    plt.close()

report = [
    "# REM Model Comparison",
    "",
    "Both models were evaluated on the same group-aware held-out test split.",
    "Thresholds were selected using validation data, not test data.",
    "",
    comparison.to_markdown(index=False, floatfmt=".4f"),
    "",
    "## Limitations",
    "- Results depend on the dataset and its labels.",
    "- Grouping identical API sequences reduces one leakage risk.",
    "- This evaluation does not establish generalization to unseen malware families.",
    "- Thresholds differ between models; operating points are not identical.",
    "- These baselines are experimental and are not production-ready detectors.",
]

(out_dir / "comparison_report.md").write_text(
    "\n".join(report), encoding="utf-8"
)

print(comparison.to_string(index=False))
print(f"\nSaved report and charts to: {out_dir.resolve()}")
