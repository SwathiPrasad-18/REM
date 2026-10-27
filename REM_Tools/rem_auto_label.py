#!/usr/bin/env python3
"""
REM automated label-assistance pipeline.

Labels samples only when there is explicit, reasonably strong evidence in the
source report. It does NOT infer ground truth from behavior events alone.
Unknown/ambiguous samples remain unknown.

Usage:
  python rem_auto_label.py --features "report_features(5).csv" --reports-zip "Raw_Reports.zip" --out-dir rem_autolabel_output

Optional VirusTotal API lookup:
  Windows PowerShell:
    $env:VT_API_KEY="YOUR_API_KEY"
    python rem_auto_label.py --features "report_features(5).csv" --reports-zip "Raw_Reports.zip" --use-vt-api --out-dir rem_autolabel_output

The public VirusTotal API has access/rate-limit restrictions; use only an API key
you are authorized to use. The script does not upload files or samples.
"""
import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    import pandas as pd
except ImportError:
    print("Missing dependency. Install with: python -m pip install pandas beautifulsoup4")
    raise SystemExit(2)

try:
    from bs4 import BeautifulSoup
except ImportError:
    print("Missing dependency. Install with: python -m pip install pandas beautifulsoup4")
    raise SystemExit(2)

SHA256_RE = re.compile(r"(?i)\b[a-f0-9]{64}\b")
# Common VirusTotal strings, including localized/older HTML renderings.
VT_RATIO_PATTERNS = [
    re.compile(r'(?i)(\d{1,3})\s*/\s*(\d{1,3})\s*(?:security\s+vendors?|engines?|detections?|engines?\s+detected)'),
    re.compile(r'(?i)(?:detected|detection ratio|malicious)\D{0,40}(\d{1,3})\s*/\s*(\d{1,3})'),
    re.compile(r'(?i)(\d{1,3})\s*/\s*(\d{1,3})'),
]

def norm_hash(value):
    value = str(value or "").strip().lower()
    return value if re.fullmatch(r"[a-f0-9]{64}", value) else ""

def html_text(raw):
    soup = BeautifulSoup(raw, "html.parser")
    return " ".join(soup.stripped_strings)

def detect_vt_ratio(text):
    # First patterns are contextual; generic ratio is accepted only if the
    # text appears to be a VirusTotal report.
    is_vt = "virustotal" in text.lower() or "virus total" in text.lower()
    patterns = VT_RATIO_PATTERNS if is_vt else VT_RATIO_PATTERNS[:2]
    for pat in patterns:
        for m in pat.finditer(text):
            positives, total = int(m.group(1)), int(m.group(2))
            if 0 <= positives <= total <= 200 and total >= 5:
                return positives, total
    return None

def get_vt_label(sha256, api_key, timeout=15):
    if not sha256 or not api_key:
        return None
    url = "https://www.virustotal.com/api/v3/files/" + sha256
    req = Request(url, headers={"x-apikey": api_key, "accept": "application/json"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
        stats = data.get("data", {}).get("attributes", {}).get("last_analysis_stats", {})
        malicious = int(stats.get("malicious", 0))
        suspicious = int(stats.get("suspicious", 0))
        harmless = int(stats.get("harmless", 0))
        undetected = int(stats.get("undetected", 0))
        total = malicious + suspicious + harmless + undetected
        if total == 0:
            return None
        return {"malicious": malicious, "suspicious": suspicious, "harmless": harmless,
                "undetected": undetected, "total": total}
    except HTTPError as e:
        return {"error": f"HTTP {e.code}"}
    except (URLError, TimeoutError, ValueError, KeyError) as e:
        return {"error": type(e).__name__}

def label_from_vt(stats):
    if not stats or "error" in stats:
        return None
    m, s, h, u, total = (stats[k] for k in ("malicious", "suspicious", "harmless", "undetected", "total"))
    # Conservative thresholds. Do not call 0 detections "benign" unless at least
    # 20 engines actually returned harmless/undetected results and none flagged.
    if total >= 10 and m >= 5:
        return ("malicious", "high", f"VirusTotal API: {m}/{total} malicious, {s} suspicious")
    if total >= 20 and m == 0 and s == 0 and h >= 10:
        return ("benign", "medium", f"VirusTotal API: 0 malicious, 0 suspicious, {h}/{total} harmless")
    return None

def label_from_html_ratio(ratio):
    if not ratio:
        return None
    p, total = ratio
    if total >= 10 and p >= 5:
        return ("malicious", "medium", f"VirusTotal HTML detection ratio: {p}/{total}")
    if total >= 20 and p == 0:
        # Zero detections are only weak evidence, not definitive benign ground truth.
        return ("unknown", "low", f"VirusTotal HTML detection ratio: 0/{total}; insufficient to establish benign ground truth")
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="Parser output report_features CSV")
    ap.add_argument("--reports-zip", required=True, help="ZIP containing original HTML reports")
    ap.add_argument("--out-dir", default="rem_autolabel_output")
    ap.add_argument("--use-vt-api", action="store_true", help="Query VirusTotal by hash; requires VT_API_KEY")
    ap.add_argument("--delay", type=float, default=16.0, help="Seconds between VT API requests (default 16)")
    args = ap.parse_args()

    features_path = Path(args.features)
    zip_path = Path(args.reports_zip)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not features_path.exists() or not zip_path.exists():
        print("ERROR: input file not found. Run this script from the folder containing both input files, or pass full paths.")
        raise SystemExit(2)

    df = pd.read_csv(features_path).fillna("")
    if "sha256" not in df.columns or "report_file" not in df.columns:
        print("ERROR: features CSV must contain 'sha256' and 'report_file' columns.")
        raise SystemExit(2)

    # Index source HTML by basename and text. No sample is executed or uploaded.
    html_by_name = {}
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.lower().endswith((".html", ".htm")):
                try:
                    raw = zf.read(name)
                    html_by_name[Path(name).name] = html_text(raw)
                except Exception:
                    pass

    api_key = os.environ.get("VT_API_KEY", "") if args.use_vt_api else ""
    if args.use_vt_api and not api_key:
        print("ERROR: --use-vt-api was set, but VT_API_KEY environment variable is not set.")
        raise SystemExit(2)

    rows = []
    cache = {}
    for i, row in df.iterrows():
        report_file = str(row.get("report_file", ""))
        sha = norm_hash(row.get("sha256", ""))
        text = html_by_name.get(Path(report_file).name, "")
        # Always start conservative.
        label, confidence, source, evidence, status = "unknown", "none", "", "", "pending"
        vt_stats = None

        if args.use_vt_api and sha:
            if sha not in cache:
                cache[sha] = get_vt_label(sha, api_key)
                time.sleep(max(0, args.delay))
            vt_stats = cache[sha]
            api_result = label_from_vt(vt_stats)
            if api_result:
                label, confidence, evidence = api_result
                source = "VirusTotal API v3"
                status = "auto_provisional"
            elif vt_stats and "error" in vt_stats:
                evidence = "VirusTotal API lookup unavailable: " + vt_stats["error"]

        # Use HTML ratio only if API did not provide a stronger label.
        if label == "unknown" and text:
            ratio = detect_vt_ratio(text)
            html_result = label_from_html_ratio(ratio)
            if html_result:
                label, confidence, evidence = html_result
                source = "VirusTotal HTML report"
                status = "auto_provisional" if label != "unknown" else "pending"

        # Existing parser verdicts can be recorded as clues, never silently
        # treated as ground truth.
        parsed_verdict = str(row.get("report_verdict", "")).strip()
        if label == "unknown" and parsed_verdict:
            evidence = f"Parser report_verdict present ({parsed_verdict}); needs independent confirmation"
            source = "Parsed report verdict (unverified)"
            status = "pending"

        if label in ("malicious", "benign") and status == "auto_provisional":
            status = "needs_review"  # automated labels are candidates, not ground truth

        rows.append({
            "sha256": sha,
            "report_file": report_file,
            "proposed_label": label,
            "confidence": confidence,
            "label_source": source,
            "label_evidence": evidence,
            "verification_status": status,
            "vt_malicious_count": (vt_stats or {}).get("malicious", "") if isinstance(vt_stats, dict) else "",
            "vt_suspicious_count": (vt_stats or {}).get("suspicious", "") if isinstance(vt_stats, dict) else "",
            "vt_total_engines": (vt_stats or {}).get("total", "") if isinstance(vt_stats, dict) else "",
            "reviewer_notes": "Automated label is provisional; do not treat as ground truth until reviewed."
        })

    manifest = pd.DataFrame(rows)
    manifest.to_csv(out_dir / "rem_auto_label_manifest.csv", index=False)
    summary = {
        "input_reports": int(len(df)),
        "reports_with_matching_html": int(sum(Path(str(x)).name in html_by_name for x in df["report_file"])),
        "unique_sha256": int(pd.Series([norm_hash(x) for x in df["sha256"] if norm_hash(x)]).nunique()),
        "labels": {str(k): int(v) for k, v in manifest["proposed_label"].value_counts(dropna=False).items()},
        "verification_status": {str(k): int(v) for k, v in manifest["verification_status"].value_counts(dropna=False).items()},
        "api_enabled": bool(args.use_vt_api),
        "note": "Automated labels are provisional. Unknown is safer than an unsupported benign/malicious label."
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nCreated: {out_dir / 'rem_auto_label_manifest.csv'}")
    print(f"Created: {out_dir / 'summary.json'}")

if __name__ == "__main__":
    main()
