# REM Hybrid Framework — Model Evaluation

## Evaluation protocol

All four models were compared on the same held-out test set of 319 unique API-call sequences. Decision thresholds were selected using validation data, not the held-out test set.

## Performance comparison

| model            |   samples |   threshold |   accuracy |   roc_auc |   malware_precision |   malware_recall |   malware_f1 |   true_negative |   false_positive |   false_negative |   true_positive |
|:-----------------|----------:|------------:|-----------:|----------:|--------------------:|-----------------:|-------------:|----------------:|-----------------:|-----------------:|----------------:|
| Random Forest    |       319 |      0.3800 |     0.9248 |    0.9742 |              0.8812 |           0.8812 |       0.8812 |             206 |               12 |               12 |              89 |
| LSTM             |       319 |      0.2500 |     0.9342 |    0.9576 |              0.9255 |           0.8614 |       0.8923 |             211 |                7 |               14 |              87 |
| Isolation Forest |       319 |      0.3215 |     0.6426 |    0.6755 |              0.4575 |           0.6931 |       0.5512 |             135 |               83 |               31 |              70 |
| REM Hybrid       |       319 |      0.4968 |     0.9436 |    0.9797 |              0.9368 |           0.8812 |       0.9082 |             212 |                6 |               12 |              89 |

## ROC-AUC summary

| model            |   roc_auc |
|:-----------------|----------:|
| Random Forest    |    0.9742 |
| LSTM             |    0.9576 |
| Isolation Forest |    0.6755 |
| REM Hybrid       |    0.9797 |

## Precision–recall summary

| model            |   average_precision |
|:-----------------|--------------------:|
| Random Forest    |              0.9584 |
| LSTM             |              0.9468 |
| Isolation Forest |              0.4463 |
| REM Hybrid       |              0.9673 |

## Confusion matrices

### Random Forest

| | Predicted benign | Predicted malicious |
|---|---:|---:|
| Actual benign | 206 | 12 |
| Actual malicious | 12 | 89 |

### LSTM

| | Predicted benign | Predicted malicious |
|---|---:|---:|
| Actual benign | 211 | 7 |
| Actual malicious | 14 | 87 |

### Isolation Forest

| | Predicted benign | Predicted malicious |
|---|---:|---:|
| Actual benign | 135 | 83 |
| Actual malicious | 31 | 70 |

### REM Hybrid

| | Predicted benign | Predicted malicious |
|---|---:|---:|
| Actual benign | 212 | 6 |
| Actual malicious | 12 | 89 |

## Interpretation

The hybrid model's observed metrics should be interpreted as results on this dataset and split, not as proof of production effectiveness. The comparison alone does not establish statistical significance, robustness to unseen malware families, or resistance to adversarial manipulation.

## Limitations

- The evaluation depends on the quality and representativeness of the labels.
- Grouped sequence splitting reduces duplicate-sequence leakage, but does not guarantee separation by malware family or collection source.
- OOF training scores and full-training base-model scores can have different distributions.
- Federated learning and autonomous response require separate experiments before their effectiveness can be claimed.

## Generated figures

- `Figures/accuracy.png`
- `Figures/roc_auc.png`
- `Figures/malware_precision.png`
- `Figures/malware_recall.png`
- `Figures/malware_f1.png`
- `Figures/roc_curves.png`
- `Figures/precision_recall_curves.png`
