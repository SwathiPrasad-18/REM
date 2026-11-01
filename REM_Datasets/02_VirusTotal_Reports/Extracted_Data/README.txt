REM raw report parser output

Files:
- report_features.csv: one row per HTML report; list-valued fields are JSON strings.
- behavior_events.csv: behavior evidence extracted from explicit HABO behavior tables and selected CAPE behavior summaries.
- extraction_log.csv: per-report extraction counts and status.
- quality_summary.json: aggregate extraction coverage and limitations.
- parse_reports.py: reproducible parser.

Run: python parse_reports.py Raw_Reports.zip -o REM_Parser_Output
Dependency: beautifulsoup4

Important: this is an extraction baseline, not a training-ready labeled dataset. Ground-truth labels are intentionally blank; VirusTotal/sandbox verdicts are not ground truth. Runtime API call sequences are intentionally left empty unless structured runtime API evidence can be reliably extracted. Review the reports and spot-check extracted rows before training.
