
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
EVIDENCE_DIR = (BASE_DIR / "evidence").resolve()
REPORT_DIR = BASE_DIR / "reports"

REGISTRY_PATH = DATA_DIR / "evidence_registry.csv"
EXCLUSIONS_PATH = DATA_DIR / "training_exclusions.csv"
REPORT_PATH = REPORT_DIR / "evidence_validation_report.json"

REQUIRED_COLUMNS = {
    "sample_id",
    "evidence_file",
    "sha256",
    "reviewer",
    "verified_label",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def main():
    if not REGISTRY_PATH.is_file():
        raise FileNotFoundError(f"Registry not found: {REGISTRY_PATH}")

    if not EXCLUSIONS_PATH.is_file():
        raise FileNotFoundError(f"Exclusion list not found: {EXCLUSIONS_PATH}")

    registry = pd.read_csv(REGISTRY_PATH, dtype=str).fillna("")
    exclusions = pd.read_csv(EXCLUSIONS_PATH, dtype=str).fillna("")

    missing = REQUIRED_COLUMNS - set(registry.columns)
    if missing:
        raise ValueError(f"Registry missing columns: {sorted(missing)}")

    if "sample_id" not in exclusions.columns:
        raise ValueError("Exclusion list must contain sample_id.")

    excluded_ids = set(exclusions["sample_id"].str.strip())
    excluded_ids.discard("")

    results = []
    seen_ids = set()

    for _, row in registry.iterrows():
        sample_id = row["sample_id"].strip()
        evidence_file = row["evidence_file"].strip()
        expected_hash = row["sha256"].strip().lower()
        reviewer = row["reviewer"].strip()
        label_text = row["verified_label"].strip()

        integrity_reasons = []
        metadata_reasons = []

        if not sample_id:
            metadata_reasons.append("missing_sample_id")
        elif sample_id in seen_ids:
            metadata_reasons.append("duplicate_sample_id")
        else:
            seen_ids.add(sample_id)

        if not reviewer:
            metadata_reasons.append("missing_reviewer")

        try:
            label = int(label_text)
            if label not in (0, 1):
                metadata_reasons.append("invalid_verified_label")
        except (TypeError, ValueError):
            metadata_reasons.append("invalid_verified_label")

        if (
            len(expected_hash) != 64
            or any(c not in "0123456789abcdef" for c in expected_hash)
        ):
            integrity_reasons.append("invalid_sha256_format")

        actual_hash = ""
        evidence_path = (EVIDENCE_DIR / evidence_file).resolve()

        try:
            evidence_path.relative_to(EVIDENCE_DIR)
            path_is_safe = True
        except ValueError:
            path_is_safe = False
            integrity_reasons.append(
                "evidence_path_outside_evidence_directory"
            )

        if path_is_safe:
            if not evidence_file:
                integrity_reasons.append("missing_evidence_file")
            elif not evidence_path.is_file():
                integrity_reasons.append("evidence_file_not_found")
            else:
                actual_hash = sha256_file(evidence_path)
                if (
                    len(expected_hash) == 64
                    and all(c in "0123456789abcdef" for c in expected_hash)
                    and actual_hash != expected_hash
                ):
                    integrity_reasons.append("sha256_mismatch")

        integrity_valid = not integrity_reasons
        metadata_valid = not metadata_reasons
        excluded = sample_id in excluded_ids

        results.append({
            "sample_id": sample_id,
            "integrity_valid": integrity_valid,
            "integrity_reasons": ";".join(sorted(set(integrity_reasons))),
            "metadata_valid": metadata_valid,
            "metadata_reasons": ";".join(sorted(set(metadata_reasons))),
            "excluded_from_training": excluded,
            "training_eligible": (
                integrity_valid and metadata_valid and not excluded
            ),
            "reviewer": reviewer,
            "verified_label": label_text,
            "evidence_file": evidence_file,
            "expected_sha256": expected_hash,
            "actual_sha256": actual_hash,
        })

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "generated_at_utc": utc_now(),
        "registry_rows": len(registry),
        "integrity_valid_records": sum(
            item["integrity_valid"] for item in results
        ),
        "integrity_invalid_records": sum(
            not item["integrity_valid"] for item in results
        ),
        "metadata_valid_records": sum(
            item["metadata_valid"] for item in results
        ),
        "training_eligible_records": sum(
            item["training_eligible"] for item in results
        ),
        "results": results,
        "warning": (
            "Matching hashes confirm file integrity against the registry, "
            "not evidence authenticity, reviewer identity, or label correctness. "
            "Training eligibility additionally requires valid metadata and "
            "absence from the exclusion list."
        ),
    }

    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))
    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
    