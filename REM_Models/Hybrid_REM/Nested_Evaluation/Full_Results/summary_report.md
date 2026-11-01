# REM Nested Stacking Evaluation

- Mode: full_nested_cv
- Outer folds completed: 5
- Inner folds per outer fold: 4
- Fixed threshold: 0.5
- Total outer predictions: 1035

## Per-fold results

| Fold | Accuracy | ROC-AUC | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.9227 | 0.9682 | 0.8947 | 0.8361 | 0.8644 |
| 2 | 0.9034 | 0.9671 | 0.7941 | 0.9000 | 0.8438 |
| 3 | 0.9469 | 0.9859 | 0.9016 | 0.9167 | 0.9091 |
| 4 | 0.9130 | 0.9635 | 0.8525 | 0.8525 | 0.8525 |
| 5 | 0.9179 | 0.9678 | 0.8116 | 0.9333 | 0.8682 |

## Mean and standard deviation across outer folds

| Metric | Mean | Standard deviation |
|---|---:|---:|
| accuracy | 0.9208 | 0.0162 |
| roc_auc | 0.9705 | 0.0088 |
| average_precision | 0.9528 | 0.0127 |
| malware_precision | 0.8509 | 0.0481 |
| malware_recall | 0.8877 | 0.0418 |
| malware_f1 | 0.8676 | 0.0252 |

## Pooled outer-fold predictions

- Accuracy: 0.9208
- ROC-AUC: 0.9693
- Average precision: 0.9507
- Malware precision: 0.8481
- Malware recall: 0.8874
- Malware F1: 0.8673
- TN: 685
- FP: 48
- FN: 34
- TP: 268

## Interpretation

These results estimate performance across outer folds drawn from the official training split. They do not replace the final evaluation on the untouched official test split. Report the per-fold variability alongside pooled metrics.
