
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
EVIDENCE = (BASE / "evidence").resolve()
REPORTS = BASE / "reports"

REGISTRY = DATA / "evidence_registry.csv"
EXCLUSIONS = DATA / "training_exclusions.csv"
EVIDENCE_REPORT = REPORTS / "evidence_validation_report.json"
FEEDBACK = DATA / "verified_feedback.csv"
GATE_REPORT = REPORTS / "isolated_gate_report.json"
REJECTIONS = REPORTS / "isolated_gate_rejections.csv"


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_path(filename):
    path = (EVIDENCE / filename).resolve()
    path.relative_to(EVIDENCE)
    return path


def main():
    for path in (REGISTRY, EXCLUSIONS, EVIDENCE_REPORT, FEEDBACK):
        if not path.is_file():
            raise FileNotFoundError(f"Missing isolated test input: {path}")

    registry = read_csv(REGISTRY)
    exclusions = read_csv(EXCLUSIONS)
    feedback = read_csv(FEEDBACK)

    with EVIDENCE_REPORT.open("r", encoding="utf-8-sig") as f:
        report = json.load(f)

    invalid_count = report.get("integrity_invalid_records")
    valid_count = report.get("integrity_valid_records", 0)

    if invalid_count is None or not isinstance(invalid_count, int):
        raise ValueError("BLOCKED: evidence report has a missing or invalid integrity count.")

    if invalid_count != 0:
        results = report.get("results", [])
        failed_rows = [
            {
                "sample_id": row.get("sample_id", ""),
                "rejection_reasons": row.get("integrity_reasons") or "evidence_integrity_failure",
            }
            for row in results
            if row.get("integrity_valid") is not True
        ]

        with REJECTIONS.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["sample_id", "rejection_reasons"])
            writer.writeheader()
            writer.writerows(failed_rows)

        audit_report = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "isolated_synthetic_test",
            "input_feedback_rows": len(feedback),
            "approved_rows": 0,
            "rejected_rows": len(failed_rows),
            "rejections": failed_rows,
            "integrity_invalid_records": invalid_count,
            "integrity_valid_records": valid_count,
            "block_reason": "evidence report has integrity failures",
            "training_launched": False,
            "production_files_modified": False,
        }
        GATE_REPORT.write_text(
            json.dumps(audit_report, indent=2),
            encoding="utf-8",
        )
        raise ValueError("BLOCKED: evidence report has integrity failures.")

    if valid_count < 1:
        raise ValueError("BLOCKED: no integrity-valid evidence.")

    report_time = datetime.fromisoformat(
        report["generated_at_utc"].replace("Z", "+00:00")
    )

    if report_time.tzinfo is None:
        raise ValueError("BLOCKED: report timestamp has no timezone.")

    # Ensure the report was generated after registry and evidence changes.
    for path in [REGISTRY] + [
        safe_path(row["evidence_file"])
        for row in registry
        if row.get("evidence_file", "").strip()
        and safe_path(row["evidence_file"]).is_file()
    ]:
        modified = datetime.fromtimestamp(
            path.stat().st_mtime, tz=timezone.utc
        )
        if modified > report_time.astimezone(timezone.utc):
            raise ValueError(
                f"BLOCKED: stale evidence report; rerun validation after {path.name} changes."
            )

    report_by_id = {
        row["sample_id"]: row
        for row in report.get("results", [])
        if row.get("sample_id")
    }
    registry_by_id = {}
    duplicate_ids = set()

    for row in registry:
        sid = row.get("sample_id", "").strip()
        if not sid:
            continue
        if sid in registry_by_id:
            duplicate_ids.add(sid)
        else:
            registry_by_id[sid] = row

    excluded_ids = {
        row.get("sample_id", "").strip()
        for row in exclusions
    }
    excluded_ids.discard("")

    approved = []
    rejected = []
    seen = set()

    required_feedback = {
        "sample_id", "verified_label", "reviewer",
        "evidence_reference", "review_status", "eligible_for_training"
    }

    if feedback and not required_feedback.issubset(feedback[0].keys()):
        raise ValueError("BLOCKED: feedback CSV is missing required columns.")

    for row in feedback:
        sid = row.get("sample_id", "").strip()
        reasons = []

        if not sid:
            reasons.append("missing_sample_id")
        elif sid in seen:
            reasons.append("duplicate_feedback_sample_id")
        seen.add(sid)

        if sid in excluded_ids:
            reasons.append("explicitly_excluded")

        reviewer = row.get("reviewer", "").strip()
        if not reviewer:
            reasons.append("missing_reviewer")

        try:
            label = int(row.get("verified_label", ""))
            if label not in (0, 1):
                reasons.append("invalid_label")
        except ValueError:
            label = None
            reasons.append("invalid_label")

        if row.get("review_status", "").strip().lower() != "verified":
            reasons.append("not_verified")

        if row.get("eligible_for_training", "").strip().lower() not in {
            "true", "1", "yes"
        }:
            reasons.append("not_training_eligible")

        evidence_ref = row.get("evidence_reference", "").strip()
        if not evidence_ref or evidence_ref.upper() == "SYNTHETIC-TEST-ONLY":
            reasons.append("missing_or_synthetic_evidence")

        if sid in duplicate_ids:
            reasons.append("duplicate_registry_id")

        reg = registry_by_id.get(sid)
        if reg is None:
            reasons.append("evidence_not_registered")
        else:
            if reg.get("reviewer", "").strip() != reviewer:
                reasons.append("reviewer_mismatch")

            try:
                reg_label = int(reg.get("verified_label", ""))
                if label is None or reg_label != label or reg_label not in (0, 1):
                    reasons.append("label_mismatch")
            except ValueError:
                reasons.append("invalid_registry_label")

            report_item = report_by_id.get(sid)
            if report_item is None:
                reasons.append("missing_validation_record")
            else:
                if report_item.get("integrity_valid") is not True:
                    reasons.append("evidence_integrity_failed")

                try:
                    path = safe_path(reg.get("evidence_file", ""))
                except (ValueError, OSError):
                    path = None
                    reasons.append("unsafe_evidence_path")

                if path is None or not path.is_file():
                    reasons.append("evidence_file_missing")
                else:
                    actual = sha256_file(path)
                    expected = reg.get("sha256", "").strip().lower()

                    if actual != expected:
                        reasons.append("sha256_mismatch")

                    if report_item.get("actual_sha256", "").lower() != actual:
                        reasons.append("report_hash_mismatch")

                    if report_item.get("evidence_file") != reg.get("evidence_file"):
                        reasons.append("report_filename_mismatch")

        if reasons:
            rejected.append({
                "sample_id": sid,
                "rejection_reasons": ";".join(sorted(set(reasons)))
            })
        else:
            approved.append(row)

    REPORTS.mkdir(parents=True, exist_ok=True)

    with REJECTIONS.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["sample_id", "rejection_reasons"]
        )
        writer.writeheader()
        writer.writerows(rejected)

    # Synthetic tests must never produce an approved training file.
    if approved:
        raise RuntimeError(
            "Test safety check failed: synthetic feedback was approved. "
            "No training file was written."
        )

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "isolated_synthetic_test",
        "input_feedback_rows": len(feedback),
        "approved_rows": 0,
        "rejected_rows": len(rejected),
        "rejections": rejected,
        "training_launched": False,
        "production_files_modified": False
    }

    GATE_REPORT.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
