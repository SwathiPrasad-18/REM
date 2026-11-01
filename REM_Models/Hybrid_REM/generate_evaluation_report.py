
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
)

ROOT = Path.cwd()
REPORT_DIR = ROOT / "REM_Models" / "Hybrid_REM" / "Reports"
PLOT_DIR = REPORT_DIR / "Figures"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

comparison_path = REPORT_DIR / "hybrid_vs_baselines.csv"
predictions_path = REPORT_DIR / "all_models_test_predictions.csv"

comparison = pd.read_csv(comparison_path)
predictions = pd.read_csv(predictions_path)

if len(predictions) != 319:
    raise ValueError(
        f"Expected 319 test samples, found {len(predictions)}."
    )

y_true = predictions["source_label"].astype(int).to_numpy()

models = {
    "Random Forest": {
        "score": "random_forest_score",
        "prediction": "random_forest_pred",
    },
    "LSTM": {
        "score": "lstm_score",
        "prediction": "lstm_pred",
    },
    "Isolation Forest": {
        "score": "isolation_forest_score",
        "prediction": "isolation_forest_pred",
    },
    "REM Hybrid": {
        "score": "rem_hybrid_score",
        "prediction": "rem_hybrid_pred",
    },
}

# The prediction CSV uses lowercase model names. Verify expected columns.
for name, columns in models.items():
    for column in columns.values():
        if column not in predictions.columns:
            raise ValueError(
                f"Missing column {column!r} for {name}. "
                "Check all_models_test_predictions.csv."
            )

# 1. Performance comparison chart
metrics = [
    ("accuracy", "Accuracy"),
    ("roc_auc", "ROC-AUC"),
    ("malware_precision", "Malware Precision"),
    ("malware_recall", "Malware Recall"),
    ("malware_f1", "Malware F1"),
]

for metric, label in metrics:
    if metric not in comparison.columns:
        continue

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(comparison["model"], comparison[metric])
    ax.set_title(f"Model Comparison — {label}")
    ax.set_ylabel(label)
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"{metric}.png", dpi=300)
    plt.close(fig)

# 2. Confusion matrices
for name, columns in models.items():
    y_pred = predictions[columns["prediction"]].astype(int).to_numpy()
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5, 4))
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Benign", "Malicious"],
    )
    display.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title(f"{name} — Confusion Matrix")
    fig.tight_layout()

    filename = name.lower().replace(" ", "_")
    fig.savefig(PLOT_DIR / f"{filename}_confusion_matrix.png", dpi=300)
    plt.close(fig)

# 3. ROC curves
fig, ax = plt.subplots(figsize=(8, 6))
roc_rows = []

for name, columns in models.items():
    scores = predictions[columns["score"]].astype(float).to_numpy()
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    auc = roc_auc_score(y_true, scores)

    ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
    roc_rows.append({
        "model": name,
        "roc_auc": auc,
    })

ax.plot([0, 1], [0, 1], linestyle="--", label="Chance")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — REM Model Comparison")
ax.legend()
fig.tight_layout()
fig.savefig(PLOT_DIR / "roc_curves.png", dpi=300)
plt.close(fig)

pd.DataFrame(roc_rows).to_csv(
    REPORT_DIR / "roc_auc_summary.csv",
    index=False,
)

# 4. Precision-recall curves
fig, ax = plt.subplots(figsize=(8, 6))
pr_rows = []

for name, columns in models.items():
    scores = predictions[columns["score"]].astype(float).to_numpy()
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    ap = average_precision_score(y_true, scores)

    ax.plot(recall, precision, label=f"{name} (AP = {ap:.3f})")
    pr_rows.append({
        "model": name,
        "average_precision": ap,
    })

ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision–Recall Curves — REM")
ax.legend()
fig.tight_layout()
fig.savefig(PLOT_DIR / "precision_recall_curves.png", dpi=300)
plt.close(fig)

pd.DataFrame(pr_rows).to_csv(
    REPORT_DIR / "precision_recall_summary.csv",
    index=False,
)

# 5. Markdown research report
report = [
    "# REM Hybrid Framework — Model Evaluation",
    "",
    "## Evaluation protocol",
    "",
    "All four models were compared on the same held-out test set of 319 "
    "unique API-call sequences. Decision thresholds were selected using "
    "validation data, not the held-out test set.",
    "",
    "## Performance comparison",
    "",
    comparison.to_markdown(index=False, floatfmt=".4f"),
    "",
    "## ROC-AUC summary",
    "",
    pd.DataFrame(roc_rows).to_markdown(index=False, floatfmt=".4f"),
    "",
    "## Precision–recall summary",
    "",
    pd.DataFrame(pr_rows).to_markdown(index=False, floatfmt=".4f"),
    "",
    "## Confusion matrices",
    "",
]

for name, columns in models.items():
    y_pred = predictions[columns["prediction"]].astype(int).to_numpy()
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    report.extend([
        f"### {name}",
        "",
        "| | Predicted benign | Predicted malicious |",
        "|---|---:|---:|",
        f"| Actual benign | {tn} | {fp} |",
        f"| Actual malicious | {fn} | {tp} |",
        "",
    ])

report.extend([
    "## Interpretation",
    "",
    "The hybrid model's observed metrics should be interpreted as results "
    "on this dataset and split, not as proof of production effectiveness. "
    "The comparison alone does not establish statistical significance, "
    "robustness to unseen malware families, or resistance to adversarial "
    "manipulation.",
    "",
    "## Limitations",
    "",
    "- The evaluation depends on the quality and representativeness of the labels.",
    "- Grouped sequence splitting reduces duplicate-sequence leakage, but does "
    "not guarantee separation by malware family or collection source.",
    "- OOF training scores and full-training base-model scores can have "
    "different distributions.",
    "- Federated learning and autonomous response require separate experiments "
    "before their effectiveness can be claimed.",
    "",
    "## Generated figures",
    "",
    "- `Figures/accuracy.png`",
    "- `Figures/roc_auc.png`",
    "- `Figures/malware_precision.png`",
    "- `Figures/malware_recall.png`",
    "- `Figures/malware_f1.png`",
    "- `Figures/roc_curves.png`",
    "- `Figures/precision_recall_curves.png`",
    "",
])

report_path = REPORT_DIR / "REM_evaluation_report.md"
report_path.write_text("\n".join(report), encoding="utf-8")

print("Evaluation report generated.")
print("Report:", report_path)
print("Figures:", PLOT_DIR)
print("ROC summary:", REPORT_DIR / "roc_auc_summary.csv")
print("PR summary:", REPORT_DIR / "precision_recall_summary.csv")
