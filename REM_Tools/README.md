# REM VirusTotal CAPE + Zenbox Parser

This parser converts saved VirusTotal CAPE and Zenbox HTML reports into one common CSV/JSON schema.

## Run

```bash
pip install -r requirements.txt
python rem_vt_parser.py --input ../02_VirusTotal_Reports/Raw_Reports --output ../02_VirusTotal_Reports/Extracted_Data
```

## Important labeling rule

The parser intentionally leaves `label` and `label_source` empty. Do not infer the REM ground-truth label from a VirusTotal detection count or sandbox verdict.

Populate the label from the original dataset/tracking sheet after matching by SHA-256.

## Important API-call rule

CAPE reports can contain static PE import/API listings. These are NOT automatically treated as runtime API calls. The parser only fills `api_calls` when a dedicated runtime API-call section is present.

## Common schema

The output includes file metadata, processes, file/registry operations, commands, mutexes/services, DNS/IP/network data, MITRE ATT&CK, behavior tags, signatures, and verdict information.

Multiple sandbox reports for the same SHA-256 should be merged later as observations of the same sample, not counted as separate malware samples.
