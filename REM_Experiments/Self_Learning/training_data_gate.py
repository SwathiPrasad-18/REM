
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


# =========================================================
# Paths and configuration
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
EVIDENCE_DIR = (BASE_DIR / "evidence").resolve()
REPORT_DIR = BASE_DIR / "reports"

FEEDBACK_PATH = DATA_DIR / "verified_feedback.csv"
EXCLUSIONS_PATH = DATA_DIR / "training_exclusions.csv"
REGISTRY_PATH = DATA_DIR / "evidence_registry.csv"
OUTPUT_PATH = DATA_DIR / "approved_training_feedback.csv"

EVIDENCE_REPORT_PATH = REPORT_DIR / "evidence_validation_report.json"
REPORT_PATH = REPORT_DIR / "training_gate_report.json"
REJECTIONS_PATH = REPORT_DIR / "training_gate_rejections.csv"

FEEDBACK_COLUMNS = {
    "sample_id",
    "verified_label",
    "reviewer",
    "evidence_reference",
    "review_status",
    "eligible_for_training",
}

REGISTRY_COLUMNS = {
    "sample_id",
    "evidence_file",
    "sha256",
    "reviewer",
    "verified_label",
}


# =========================================================
# Utilities
# =========================================================

def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def safe_evidence_path(filename):
    """Resolve evidence paths without permitting directory traversal."""

    if not filename:
        raise ValueError("missing_evidence_filename")

    path = (EVIDENCE_DIR / filename).resolve()

    try:
        path.relative_to(EVIDENCE_DIR)
    except ValueError as exc:
        raise ValueError(
            "evidence_path_outside_evidence_directory"
        ) from exc

    return path


def load_evidence_report():
    """
    Read the updated validator report.

    Integrity failures block the gate. Training exclusions are checked
    separately against the exclusion list and feedback records.
    """

    if not EVIDENCE_REPORT_PATH.is_file():
        raise FileNotFoundError(
            f"Evidence validation report missing: {EVIDENCE_REPORT_PATH}"
        )

    try:
        report = json.loads(
            EVIDENCE_REPORT_PATH.read_text(encoding="utf-8")
        )
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(
            "Evidence validation report is unreadable or invalid."
        ) from exc

    if not isinstance(report, dict):
        raise ValueError("Evidence report must be a JSON object.")

    required_fields = {
        "generated_at_utc",
        "integrity_valid_records",
        "integrity_invalid_records",
        "results",
    }

    missing = required_fields - set(report)

    if missing:
        raise ValueError(
            f"Evidence report missing fields: {sorted(missing)}. "
            "Rerun validate_evidence.py."
        )

    if not isinstance(report["results"], list):
        raise ValueError("Evidence report results must be a list.")

    valid_count = report["integrity_valid_records"]
    invalid_count = report["integrity_invalid_records"]

    if (
        not isinstance(valid_count, int)
        or isinstance(valid_count, bool)
        or not isinstance(invalid_count, int)
        or isinstance(invalid_count, bool)
        or valid_count < 0
        or invalid_count < 0
    ):
        raise ValueError("Evidence report record counts are invalid.")

    if invalid_count != 0:
        raise ValueError(
            "Evidence integrity failures detected. Training gate blocked."
        )

    if valid_count < 1:
        raise ValueError(
            "No integrity-valid evidence records. Training gate blocked."
        )

    # Independently reconcile report counts with the reported results.
    results = report["results"]

    actual_valid = sum(
        item.get("integrity_valid") is True
        for item in results
        if isinstance(item, dict)
    )

    actual_invalid = sum(
        item.get("integrity_valid") is False
        for item in results
        if isinstance(item, dict)
    )

    if (
        len(results) != valid_count + invalid_count
        or actual_valid != valid_count
        or actual_invalid != invalid_count
    ):
        raise ValueError(
            "Evidence report counts do not match its results."
        )

    return report


def validate_report_freshness(evidence_report):
    """
    Require the evidence report to postdate the registry and each
    registered evidence file. Files changed after validation block use.
    """

    timestamp_text = evidence_report.get("generated_at_utc")

    if not timestamp_text:
        raise ValueError(
            "Evidence report timestamp missing. "
            "Rerun validate_evidence.py."
        )

    try:
        report_time = datetime.fromisoformat(
            timestamp_text.replace("Z", "+00:00")
        )

        if report_time.tzinfo is None:
            raise ValueError("Timestamp must include a timezone.")

        report_time = report_time.astimezone(timezone.utc)

    except (ValueError, TypeError) as exc:
        raise ValueError(
            "Evidence report timestamp is invalid."
        ) from exc

    paths_to_check = [REGISTRY_PATH]

    registry = pd.read_csv(
        REGISTRY_PATH, dtype=str
    ).fillna("")

    for _, row in registry.iterrows():
        filename = row["evidence_file"].strip()

        if not filename:
            continue

        try:
            path = safe_evidence_path(filename)
        except ValueError:
            # The record-level checks will reject unsafe paths.
            continue

        if path.is_file():
            paths_to_check.append(path)

    for path in paths_to_check:
        modified = datetime.fromtimestamp(
            path.stat().st_mtime,
            tz=timezone.utc,
        )

        if modified > report_time:
            raise ValueError(
                f"Evidence report is stale: {path.name} changed after "
                "validation. Rerun validate_evidence.py."
            )


# =========================================================
# Main training-data gate
# =========================================================

def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------
    # Load required inputs
    # -----------------------------------------------------

    for path in (
        FEEDBACK_PATH,
        EXCLUSIONS_PATH,
        REGISTRY_PATH,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required file missing: {path}"
            )

    feedback = pd.read_csv(
        FEEDBACK_PATH, dtype=str
    ).fillna("")

    exclusions = pd.read_csv(
        EXCLUSIONS_PATH, dtype=str
    ).fillna("")

    registry = pd.read_csv(
        REGISTRY_PATH, dtype=str
    ).fillna("")

    missing_feedback = FEEDBACK_COLUMNS - set(feedback.columns)
    missing_registry = REGISTRY_COLUMNS - set(registry.columns)

    if missing_feedback:
        raise ValueError(
            f"Feedback file missing columns: {sorted(missing_feedback)}"
        )

    if missing_registry:
        raise ValueError(
            f"Evidence registry missing columns: {sorted(missing_registry)}"
        )

    if "sample_id" not in exclusions.columns:
        raise ValueError(
            "Exclusion list must contain sample_id."
        )

    # -----------------------------------------------------
    # Validate report and freshness before approving data
    # -----------------------------------------------------

    evidence_report = load_evidence_report()
    validate_report_freshness(evidence_report)

    # Every evidence report entry must explicitly be integrity-valid.
    # This also prevents a malformed report from silently passing.
    report_by_id = {}

    for item in evidence_report["results"]:
        if not isinstance(item, dict):
            raise ValueError("Malformed evidence report result.")

        sid = str(item.get("sample_id", "")).strip()

        if not sid or sid in report_by_id:
            raise ValueError(
                "Evidence report has missing or duplicate sample IDs."
            )

        if item.get("integrity_valid") is not True:
            raise ValueError(
                f"Evidence integrity not confirmed for sample {sid}."
            )

        report_by_id[sid] = item

    # -----------------------------------------------------
    # Prepare exclusions and registry index
    # -----------------------------------------------------

    excluded_ids = set(
        exclusions["sample_id"].astype(str).str.strip()
    )
    excluded_ids.discard("")

    registry_by_id = {}
    duplicate_registry_ids = set()

    for _, row in registry.iterrows():
        sid = row["sample_id"].strip()

        if not sid:
            continue

        if sid in registry_by_id:
            duplicate_registry_ids.add(sid)
        else:
            registry_by_id[sid] = row

    # -----------------------------------------------------
    # Validate each feedback record
    # -----------------------------------------------------

    seen_feedback_ids = set()
    approved = []
    rejected = []

    for _, row in feedback.iterrows():
        sid = row["sample_id"].strip()
        reviewer = row["reviewer"].strip()
        evidence_ref = row["evidence_reference"].strip()

        reasons = []

        # IDs and duplicates
        if not sid:
            reasons.append("missing_sample_id")
        elif sid in seen_feedback_ids:
            reasons.append("duplicate_feedback_sample_id")
        else:
            seen_feedback_ids.add(sid)

        # Training exclusions are independent of file integrity.
        if sid in excluded_ids:
            reasons.append("explicitly_excluded")

        # Reviewer
        if not reviewer or reviewer.lower() == "nan":
            reasons.append("missing_reviewer")

        # Feedback label
        label = None

        try:
            label = int(row["verified_label"])

            if label not in (0, 1):
                reasons.append("invalid_verified_label")
                label = None

        except (TypeError, ValueError):
            reasons.append("invalid_verified_label")

        # Review status and eligibility flag
        if row["review_status"].strip().lower() != "verified":
            reasons.append("not_verified")

        if row["eligible_for_training"].strip().lower() not in {
            "true", "1", "yes"
        }:
            reasons.append("not_training_eligible")

        # Never accept a synthetic placeholder as evidence.
        if (
            not evidence_ref
            or evidence_ref.lower() == "nan"
            or evidence_ref.upper() == "SYNTHETIC-TEST-ONLY"
        ):
            reasons.append("missing_or_synthetic_evidence")

        # Registry checks
        if sid in duplicate_registry_ids:
            reasons.append("duplicate_evidence_registry_entry")

        registry_row = registry_by_id.get(sid)

        if registry_row is None:
            reasons.append("evidence_not_registered")

        else:
            reg_reviewer = registry_row["reviewer"].strip()
            reg_label_text = registry_row["verified_label"].strip()

            if reg_reviewer != reviewer:
                reasons.append("reviewer_mismatch")

            try:
                reg_label = int(reg_label_text)

                if (
                    reg_label not in (0, 1)
                    or label is None
                    or reg_label != label
                ):
                    reasons.append("verified_label_mismatch")

            except (TypeError, ValueError):
                reasons.append("invalid_registry_label")

            # Evidence must appear in the latest integrity report.
            report_item = report_by_id.get(sid)

            if report_item is None:
                reasons.append("evidence_missing_from_validation_report")
            else:
                if report_item.get("reviewer") != reg_reviewer:
                    reasons.append("report_reviewer_mismatch")

                if (
                    str(report_item.get("verified_label", "")).strip()
                    != reg_label_text
                ):
                    reasons.append("report_label_mismatch")

                if (
                    report_item.get("evidence_file")
                    != registry_row["evidence_file"]
                ):
                    reasons.append("report_evidence_filename_mismatch")

            evidence_name = registry_row["evidence_file"].strip()
            expected_hash = registry_row["sha256"].strip().lower()

            try:
                evidence_path = safe_evidence_path(evidence_name)

            except ValueError as exc:
                reasons.append(str(exc))
                evidence_path = None

            if evidence_path is not None:
                if not evidence_path.is_file():
                    reasons.append("evidence_file_not_found")

                else:
                    actual_hash = sha256_file(evidence_path)

                    hash_is_valid = (
                        len(expected_hash) == 64
                        and all(
                            char in "0123456789abcdef"
                            for char in expected_hash
                        )
                    )

                    if not hash_is_valid:
                        reasons.append("invalid_registry_sha256")

                    elif actual_hash != expected_hash:
                        reasons.append("sha256_mismatch")

                    if (
                        report_item is not None
                        and report_item.get("actual_sha256", "").lower()
                        != actual_hash
                    ):
                        reasons.append("report_hash_mismatch")

        # Approve only if every check passed.
        if reasons:
            rejected.append({
                "sample_id": sid,
                "rejection_reasons": ";".join(
                    sorted(set(reasons))
                ),
            })
        else:
            approved.append(row.to_dict())

    # -----------------------------------------------------
    # Write rejection and approval outputs
    # -----------------------------------------------------

    rejected_df = pd.DataFrame(
        rejected,
        columns=["sample_id", "rejection_reasons"],
    )

    rejected_df.to_csv(REJECTIONS_PATH, index=False)

    approved_df = pd.DataFrame(
        approved,
        columns=feedback.columns,
    )

    # Remove stale approval output if no records pass this run.
    if approved_df.empty:
        if OUTPUT_PATH.exists():
            OUTPUT_PATH.unlink()

        approved_output_created = False

    else:
        approved_df.to_csv(OUTPUT_PATH, index=False)
        approved_output_created = True

    # -----------------------------------------------------
    # Summary report
    # -----------------------------------------------------

    summary = {
        "generated_at_utc": utc_now(),
        "input_feedback_rows": int(len(feedback)),
        "approved_rows": int(len(approved_df)),
        "rejected_rows": int(len(rejected_df)),
        "excluded_sample_ids": sorted(
            set(feedback["sample_id"].str.strip())
            & excluded_ids
        ),
        "evidence_validation": {
            "report_path": str(EVIDENCE_REPORT_PATH),
            "integrity_valid_records": evidence_report[
                "integrity_valid_records"
            ],
            "integrity_invalid_records": evidence_report[
                "integrity_invalid_records"
            ],
            "freshness_check": "passed",
        },
        "approved_output_created": approved_output_created,
        "approved_output_path": (
            str(OUTPUT_PATH) if approved_output_created else None
        ),
        "rejection_report_path": str(REJECTIONS_PATH),
        "training_launched": False,
        "warning": (
            "Integrity checks do not prove evidence authenticity, reviewer "
            "identity, or label correctness. Production use requires "
            "authenticated reviewers and independent label-quality controls."
        ),
    }

    write_json(REPORT_PATH, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    try:
        main()

    except Exception as exc:
        print(
            json.dumps(
                {
                    "generated_at_utc": utc_now(),
                    "status": "BLOCKED",
                    "training_launched": False,
                    "error": str(exc),
                },
                indent=2,
            )
        )
        raise SystemExit(1)
