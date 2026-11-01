import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
REPORT_DIR = BASE_DIR / "reports"

APPROVED_PATH = DATA_DIR / "approved_training_feedback.csv"
EXCLUSIONS_PATH = DATA_DIR / "training_exclusions.csv"
EVIDENCE_REPORT_PATH = REPORT_DIR / "evidence_validation_report.json"
GATE_REPORT_PATH = REPORT_DIR / "training_gate_report.json"
REPORT_PATH = REPORT_DIR / "retraining_decision.json"

MIN_APPROVED_RECORDS = 50


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def load_json(path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def main():
    reasons = []
    feedback = pd.DataFrame()
    label_counts = {"0": 0, "1": 0}

    # Require current evidence validation to contain no invalid records.
    evidence_report = load_json(EVIDENCE_REPORT_PATH)
    if evidence_report is None:
        reasons.append("Evidence validation report missing or invalid")
    else:
        if evidence_report.get("invalid_records") != 0:
            reasons.append("Evidence validation has invalid records")
        if evidence_report.get("valid_records", 0) < 1:
            reasons.append("No valid evidence records in validation report")

    # Require a gate report with no rejected feedback and a consistent count.
    gate_report = load_json(GATE_REPORT_PATH)
    if gate_report is None:
        reasons.append("Training-data gate report missing or invalid")
    else:
        if gate_report.get("rejected_rows") != 0:
            reasons.append("Training-data gate rejected records")
        if gate_report.get("approved_rows", 0) < 1:
            reasons.append("Training-data gate approved no records")

    if not APPROVED_PATH.is_file():
        reasons.append("approved_training_feedback.csv does not exist")
    else:
        try:
            feedback = pd.read_csv(APPROVED_PATH, dtype=str).fillna("")
        except Exception as exc:
            reasons.append(f"Could not read approved feedback: {type(exc).__name__}")

    if feedback.empty:
        reasons.append("No approved feedback records are available")
    else:
        required = {
            "sample_id", "verified_label", "reviewer",
            "evidence_reference", "review_status", "eligible_for_training",
        }
        missing = required - set(feedback.columns)
        if missing:
            reasons.append("Approved feedback missing columns: " + ", ".join(sorted(missing)))
        else:
            ids = feedback["sample_id"].astype(str).str.strip()

            if ids.eq("").any():
                reasons.append("Missing sample IDs detected")
            if ids.duplicated().any():
                reasons.append("Duplicate sample IDs detected")

            labels = pd.to_numeric(feedback["verified_label"], errors="coerce")
            if not labels.isin([0, 1]).all():
                reasons.append("Invalid verified labels detected")

            label_counts = {
                "0": int((labels == 0).sum()),
                "1": int((labels == 1).sum()),
            }

            if not feedback["review_status"].str.strip().str.lower().eq("verified").all():
                reasons.append("Unverified records detected")

            eligible = feedback["eligible_for_training"].str.strip().str.lower()
            if not eligible.isin(["true", "1", "yes"]).all():
                reasons.append("Ineligible records detected")

            reviewers = feedback["reviewer"].str.strip()
            if reviewers.eq("").any() or reviewers.str.lower().eq("nan").any():
                reasons.append("Missing reviewer information")

            evidence = feedback["evidence_reference"].str.strip()
            if (
                evidence.eq("").any()
                or evidence.str.lower().eq("nan").any()
                or evidence.str.upper().eq("SYNTHETIC-TEST-ONLY").any()
            ):
                reasons.append("Missing or synthetic evidence references")

            if len(feedback) < MIN_APPROVED_RECORDS:
                reasons.append(
                    f"Need at least {MIN_APPROVED_RECORDS} approved records; found {len(feedback)}"
                )

            if not all(label_counts[label] > 0 for label in ("0", "1")):
                reasons.append("Both benign (0) and malicious (1) labels are required")

    # Always re-check exclusions independently.
    if not EXCLUSIONS_PATH.is_file():
        reasons.append("Training exclusion file is missing")
    else:
        try:
            exclusions = pd.read_csv(EXCLUSIONS_PATH, dtype=str).fillna("")
            if "sample_id" not in exclusions.columns:
                reasons.append("Exclusion file has no sample_id column")
            elif not feedback.empty and "sample_id" in feedback.columns:
                excluded_ids = set(exclusions["sample_id"].str.strip())
                overlap = set(feedback["sample_id"].astype(str).str.strip()) & excluded_ids
                if overlap:
                    reasons.append("Approved feedback contains excluded sample IDs")
        except Exception as exc:
            reasons.append(f"Could not read exclusions: {type(exc).__name__}")

    ready = len(reasons) == 0

    report = {
        "generated_at_utc": utc_now(),
        "status": "READY_FOR_REVIEW" if ready else "NOT_READY",
        "training_launched": False,
        "approved_record_count": int(len(feedback)),
        "minimum_required_records": MIN_APPROVED_RECORDS,
        "label_counts": label_counts,
        "evidence_validation_report": str(EVIDENCE_REPORT_PATH),
        "training_gate_report": str(GATE_REPORT_PATH),
        "reasons": reasons,
        "warning": (
            "This controller checks prerequisite reports and feedback fields. "
            "It does not authenticate reviewers or prove labels correct. "
            "It does not launch training."
        ),
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nDecision report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
