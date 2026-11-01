REM API-Sequence Random Forest Baseline

1. Copy train_api_sequence_baseline.py into:
   REM_Models\train_api_sequence_baseline.py

2. Run from the REM_Project root in PowerShell:
   python .\REM_Models\train_api_sequence_baseline.py `
     --data ".\REM_Datasets\04_Unified_Dataset\processed\public_labeled_candidate.csv" `
     --out-dir ".\REM_Models\API_Sequence_Baseline"

3. Outputs:
   - metrics.json
   - api_sequence_rf_pipeline.joblib
   - feature_columns.json
   - split_manifest.csv
   - feature_importance.csv

The script uses only API-call columns 0..152 and source_label. It excludes SHA256,
parser features, label metadata, and other columns from model inputs. It preserves
API-call order in text and uses unigram/bigram TF-IDF with Random Forest.

The default split is 65% train, 15% validation, 20% test. This is a first baseline,
not production detection. Confirm 0=benign and 1=malicious against the dataset
documentation before using results in a paper.
