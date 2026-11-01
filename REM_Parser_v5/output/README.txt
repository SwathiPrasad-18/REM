REM Parser v5
Run: py parse_reports_v5.py Raw_Reports.zip -o REM_Parser_v5_Output
Dependency: py -m pip install beautifulsoup4
Outputs: report_features.csv, behavior_events.csv, extraction_log.csv, parse_errors.csv, quality_summary.json.
Ground-truth labels remain blank; report verdicts are not ground truth. Review source evidence before training.
