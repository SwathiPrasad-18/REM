REM preprocessing package

Install:
  python -m pip install pandas

Run from REM_Project root (adjust paths if your filenames differ):
  python .\REM_Preprocessing\rem_preprocess.py `
    --public ".\REM_Datasets\04_Unified_Dataset\MalBehavD_V1_cleaned_deduplicated.csv" `
    --reports ".\REM_Parser_v5\output\report_features.csv" `
    --events ".\REM_Parser_v5\output\behavior_events.csv" `
    --manifest ".\REM_Datasets\02_VirusTotal_Reports\Auto_Label_Output\rem_auto_label_manifest.csv" `
    --out-dir ".\REM_Datasets\04_Unified_Dataset\processed"

Outputs:
- public_labeled_candidate.csv
- rem_reports_enriched.csv
- public_duplicate_hash_audit.csv
- preprocessing_summary.json

Caution:
- Numeric public labels are preserved as source_label until original documentation
  confirms their meanings.
- The script joins only on exact SHA-256 and does not assume all samples match.
- VirusTotal labels remain provisional and are never treated as ground truth.
- Split by sample identity before training. Do not fit scalers/encoders on all data.
