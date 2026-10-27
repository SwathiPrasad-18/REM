# REM Unified Dataset and Preprocessing Pipeline

## Run
Install dependencies:
```bash
pip install -r requirements.txt
```
Example (Windows PowerShell; adjust paths):
```powershell
python rem_pipeline.py --malbehav "C:\REM\MalBehavD-V1-dataset.csv" --claude "C:\REM\malware_behavior_dataset.csv" --reports "C:\REM\reports" --output "C:\REM\unified"
```
Optional verified labels file:
```powershell
python rem_pipeline.py --malbehav "C:\REM\MalBehavD-V1-dataset.csv" --claude "C:\REM\malware_behavior_dataset.csv" --reports "C:\REM\reports" --labels "C:\REM\verified_labels.csv" --output "C:\REM\unified"
```
`verified_labels.csv` must contain `sha256,label`; labels must be independently verified and documented (`benign`/`malware` or `0`/`1`).

## Outputs
- `samples.csv`: unified sample-level table and JSON feature columns.
- `sequences.jsonl`: sequence records intended for LSTM preprocessing.
- `report_events.csv`: event candidates extracted from HTML reports.
- `source_audit.csv`: source coverage and parser errors.
- `split_manifest.csv`: grouped train/validation/test split for externally verified labels only.
- `schema.json`: provenance and leakage policies.

## Important limitations
- The script parses CSV/HTML only; it never executes malware.
- Claude-generated labels are unverified by default; their numeric features may be useful for pipeline tests but must not be represented as real observed telemetry without evidence.
- MalBehavD-V1 label mapping must be verified against the original source before supervised use.
- Sandbox verdicts are never used as ground-truth labels.
- SHA-256 hashes are grouped to reduce leakage between reports of the same sample. Records without a valid SHA-256 cannot be reliably deduplicated.
- HTML parsing is heuristic and must be manually validated. In particular, API candidates from HTML rows are not guaranteed to be runtime API-call sequences; static imports and page text must not be used as observed behavior without validation.
- This preprocessing output is not yet a model-ready fixed numeric matrix. The Random Forest branch needs a validated fixed feature vector; the LSTM branch needs true ordered runtime sequences; Isolation Forest should be trained with a documented normal/benign reference set.
