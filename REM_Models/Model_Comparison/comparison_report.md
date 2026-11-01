# REM Model Comparison

Both models were evaluated on the same group-aware held-out test split.
Thresholds were selected using validation data, not test data.

| model         |   threshold |   accuracy |   roc_auc |   malware_precision |   malware_recall |   malware_f1 |   false_positives |   false_negatives |
|:--------------|------------:|-----------:|----------:|--------------------:|-----------------:|-------------:|------------------:|------------------:|
| Random Forest |      0.3800 |     0.9248 |    0.9742 |              0.8812 |           0.8812 |       0.8812 |                12 |                12 |
| LSTM          |      0.2500 |     0.9342 |    0.9576 |              0.9255 |           0.8614 |       0.8923 |                 7 |                14 |

## Limitations
- Results depend on the dataset and its labels.
- Grouping identical API sequences reduces one leakage risk.
- This evaluation does not establish generalization to unseen malware families.
- Thresholds differ between models; operating points are not identical.
- These baselines are experimental and are not production-ready detectors.