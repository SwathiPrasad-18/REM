# REM Nested Stacking Evaluation

- Mode: smoke_test
- Outer folds completed: 1
- Inner folds per outer fold: 4
- Fixed threshold: 0.5
- Total outer predictions: 207

## Per-fold results

| Fold | Accuracy | ROC-AUC | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.9227 | 0.9682 | 0.8947 | 0.8361 | 0.8644 |

## Mean and standard deviation across outer folds

| Metric | Mean | Standard deviation |
|---|---:|---:|
| accuracy | 0.9227 | 0.0000 |
| roc_auc | 0.9682 | 0.0000 |
| average_precision | 0.9468 | 0.0000 |
| malware_precision | 0.8947 | 0.0000 |
| malware_recall | 0.8361 | 0.0000 |
| malware_f1 | 0.8644 | 0.0000 |

## Pooled outer-fold predictions

- Accuracy: 0.9227
- ROC-AUC: 0.9682
- Average precision: 0.9468
- Malware precision: 0.8947
- Malware recall: 0.8361
- Malware F1: 0.8644
- TN: 140
- FP: 6
- FN: 10
- TP: 51

## Interpretation

These results estimate performance across outer folds drawn from the official training split. They do not replace the final evaluation on the untouched official test split. Report the per-fold variability alongside pooled metrics.
