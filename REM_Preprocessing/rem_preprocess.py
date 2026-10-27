#!/usr/bin/env python3
"""REM dataset preparation pipeline (safe, provenance-preserving).

This script:
- reads the cleaned MalBehavD-V1 CSV, REM report features, behavior events,
  and optional auto-label manifest;
- preserves the source dataset's numeric label as `source_label` (does not guess
  whether 0/1 means benign/malicious);
- aggregates behavior-event counts per report/hash;
- matches REM reports to the public dataset by SHA-256;
- emits separate public-training and REM-report tables;
- flags duplicate hashes and label conflicts.

It does NOT train models or promote VirusTotal suggestions to ground truth.
"""
import argparse
import json
from pathlib import Path
import re
import pandas as pd

def norm_hash(x):
    s = str(x or "").strip().lower()
    return s if re.fullmatch(r"[0-9a-f]{64}", s) else ""

def read_csv(path, required=True):
    p = Path(path)
    if not p.exists():
        if required:
            raise FileNotFoundError(f"Input not found: {p}")
        return pd.DataFrame()
    return pd.read_csv(p, low_memory=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--public", required=True, help="Cleaned MalBehavD-V1 CSV")
    ap.add_argument("--reports", required=True, help="Parser report_features CSV")
    ap.add_argument("--events", required=True, help="Parser behavior_events CSV")
    ap.add_argument("--manifest", default="", help="Optional REM auto-label manifest CSV")
    ap.add_argument("--out-dir", default="REM_Datasets/04_Unified_Dataset/processed")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    public = read_csv(args.public)
    reports = read_csv(args.reports)
    events = read_csv(args.events)
    manifest = read_csv(args.manifest, required=False) if args.manifest else pd.DataFrame()

    if "sha256" not in public.columns:
        raise ValueError("Public dataset must contain a sha256 column.")
    if "sha256" not in reports.columns:
        raise ValueError("Report features CSV must contain a sha256 column.")
    if "labels" not in public.columns:
        raise ValueError("Public dataset must contain a labels column; inspect schema before proceeding.")

    public["sha256"] = public["sha256"].map(norm_hash)
    reports["sha256"] = reports["sha256"].map(norm_hash)
    public = public[public["sha256"] != ""].copy()
    reports = reports.copy()

    # Audit duplicates before dropping exact repeated sample identities.
    public_dups = public[public.duplicated("sha256", keep=False)].sort_values("sha256")
    conflicts = public.groupby("sha256")["labels"].nunique(dropna=True)
    conflict_hashes = set(conflicts[conflicts > 1].index)
    public_unique = public.drop_duplicates("sha256", keep="first").copy()

    # Keep labels numeric and unmodified until label mapping has been documented.
    public_unique = public_unique.rename(columns={"labels": "source_label"})
    public_unique["label_source"] = "MalBehavD-V1"
    public_unique["label_status"] = "source_label_mapping_requires_documentation"

    # Aggregate event records by hash, with a report-name fallback for missing hashes.
    if not events.empty:
        for col in ["sha256", "report_file", "event_type", "detail"]:
            if col not in events.columns:
                events[col] = ""
        events["sha256"] = events["sha256"].map(norm_hash)
        events["event_type"] = events["event_type"].fillna("").astype(str).str.strip()
        events["detail"] = events["detail"].fillna("").astype(str)
        valid = events[events["sha256"] != ""].copy()
        if not valid.empty:
            counts = valid.groupby(["sha256", "event_type"]).size().unstack(fill_value=0)
            counts.columns = ["event_count__" + re.sub(r"[^A-Za-z0-9_]+", "_", str(c)).strip("_")[:60] for c in counts.columns]
            counts = counts.reset_index()
            total = valid.groupby("sha256").size().rename("behavior_event_total").reset_index()
            event_features = counts.merge(total, on="sha256", how="outer")
        else:
            event_features = pd.DataFrame(columns=["sha256", "behavior_event_total"])
    else:
        event_features = pd.DataFrame(columns=["sha256", "behavior_event_total"])

    # Optional manifest lookup; proposed labels remain explicitly provisional.
    if not manifest.empty and "sha256" in manifest.columns:
        manifest["sha256"] = manifest["sha256"].map(norm_hash)
        keep = [c for c in ["sha256", "proposed_label", "label_source", "label_evidence", "verification_status"] if c in manifest.columns]
        manifest = manifest[keep].drop_duplicates("sha256", keep="first")
        manifest = manifest.rename(columns={
            "proposed_label": "vt_proposed_label",
            "label_source": "vt_label_source",
            "label_evidence": "vt_label_evidence",
            "verification_status": "vt_verification_status"
        })

    # Public model-ready candidate: preserve original features, numeric labels, and provenance.
    public_train = public_unique.copy()
    if not event_features.empty:
        public_train = public_train.merge(event_features, on="sha256", how="left")
    public_train["hash_label_conflict"] = public_train["sha256"].isin(conflict_hashes)
    public_train["eligible_for_supervised_training"] = ~public_train["hash_label_conflict"]
    # Do not scale/encode automatically; downstream model pipeline should split first.

    # REM report table enriched from public dataset via exact hash only.
    label_lookup = public_unique[["sha256", "source_label", "label_source", "label_status"]].copy()
    rem = reports.merge(label_lookup, on="sha256", how="left", suffixes=("", "_public"))
    if not event_features.empty:
        rem = rem.merge(event_features, on="sha256", how="left")
    if not manifest.empty:
        rem = rem.merge(manifest, on="sha256", how="left")
    rem["label_status"] = rem.get("label_status", pd.Series(index=rem.index, dtype="object")).fillna("unknown_no_public_hash_match")
    rem["eligible_for_supervised_training"] = rem["source_label"].notna() & (rem["label_status"] != "unknown_no_public_hash_match")
    rem["vt_label_is_ground_truth"] = False

    public_train.to_csv(out / "public_labeled_candidate.csv", index=False)
    rem.to_csv(out / "rem_reports_enriched.csv", index=False)
    public_dups.to_csv(out / "public_duplicate_hash_audit.csv", index=False)

    matched = int(rem["source_label"].notna().sum()) if "source_label" in rem else 0
    summary = {
        "public_rows_input": int(len(public)),
        "public_unique_hashes": int(public["sha256"].nunique()),
        "public_rows_after_hash_deduplication": int(len(public_unique)),
        "public_duplicate_rows": int(len(public_dups)),
        "public_hashes_with_conflicting_labels": int(len(conflict_hashes)),
        "report_rows": int(len(reports)),
        "report_rows_matched_to_public_dataset": matched,
        "report_rows_without_public_label_match": int(len(reports) - matched),
        "behavior_event_rows": int(len(events)),
        "behavior_event_hashes_with_features": int(event_features["sha256"].nunique()) if "sha256" in event_features.columns else 0,
        "source_label_mapping": "preserved numeric values; not mapped to benign/malicious by this script",
        "VirusTotal_labels": "retained as provisional suggestions only; never treated as ground truth",
        "model_training": "not performed"
    }
    (out / "preprocessing_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nOutputs written to: {out.resolve()}")

if __name__ == "__main__":
    main()
