REM Parser v5 (experimental)

Run from this folder:
  py -m pip install beautifulsoup4
  py parse_reports_v5.py "..\REM_Datasets\02_VirusTotal_Reports\Raw_Reports.zip" -o output

Output files: report_features.csv, behavior_events.csv, extraction_log.csv, parse_errors.csv, quality_summary.json.

Important: v5 is experimental. It processed 40 reports with 0 parser exceptions, but that is not a correctness guarantee. Verdict extraction remains empty. Manually compare extracted events/signatures with original HTML. Ground-truth labels are blank; do not use report verdicts as ground truth. Runtime API calls are not inferred.
