
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


BASE_DIR = Path("REM_Experiments/Self_Learning")
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"

QUEUE_PATH = DATA_DIR / "prediction_queue.csv"
VERIFIED_PATH = DATA_DIR / "verified_feedback.csv"
AUDIT_PATH = LOG_DIR / "feedback_audit.jsonl"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def record_verified_feedback(
    sample_id,
    verified_label,
    reviewer,
    evidence_reference,
):
    """
    Record analyst-verified feedback.

    This local prototype records the reviewer identity supplied by the caller.
    It does not authenticate the reviewer; production authentication is needed.
    """
    if verified_label not in (0, 1):
        raise ValueError("verified_label must be 0 (benign) or 1 (malicious).")

    if not str(sample_id).strip():
        raise ValueError("sample_id cannot be empty.")

    if not str(reviewer).strip():
        raise ValueError("A reviewer identity is required.")

    if not str(evidence_reference).strip():
        raise ValueError("An evidence reference is required.")

    if not QUEUE_PATH.exists():
        raise FileNotFoundError(
            f"Prediction queue not found: {QUEUE_PATH}"
        )

    queue = pd.read_csv(QUEUE_PATH, dtype={"sample_id": str})
    matches = queue.index[queue["sample_id"] == str(sample_id)].tolist()

    if len(matches) != 1:
        raise ValueError(
            "sample_id must match exactly one prediction in the queue."
        )

    existing = pd.read_csv(VERIFIED_PATH) if VERIFIED_PATH.exists() else pd.DataFrame()

    if not existing.empty:
        prior = existing[existing["sample_id"].astype(str) == str(sample_id)]

        if not prior.empty:
            labels = set(pd.to_numeric(prior["verified_label"]).astype(int))
            if labels != {int(verified_label)}:
                raise ValueError(
                    "Conflicting verified label detected. "
                    "Do not overwrite; escalate for adjudication."
                )

            raise ValueError(
                "This sample already has verified feedback. "
                "Use an explicit review workflow for corrections."
            )

    queue_row = queue.loc[matches[0]]
    record = {
        "sample_id": str(sample_id),
        "verified_label": int(verified_label),
        "reviewer": str(reviewer).strip(),
        "evidence_reference": str(evidence_reference).strip(),
        "verified_at_utc": utc_now(),
        "review_status": "verified",
        "eligible_for_training": True,
        "source_model_version": str(queue_row.get("model_version", "unknown")),
        "original_prediction": int(queue_row["predicted_label"]),
        "original_confidence": float(queue_row["confidence"]),
    }

    # Write feedback first, then update the prediction queue.
    # The verified-feedback CSV is append-only in this prototype.
    new_row = pd.DataFrame([record])
    if VERIFIED_PATH.exists():
        new_row.to_csv(
            VERIFIED_PATH, mode="a", header=False, index=False
        )
    else:
        new_row.to_csv(VERIFIED_PATH, index=False)

    queue.loc[matches[0], "verified_label"] = int(verified_label)
    queue.loc[matches[0], "review_status"] = "verified"
    queue.loc[matches[0], "eligible_for_training"] = True
    queue.to_csv(QUEUE_PATH, index=False)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with AUDIT_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")

    return record


if __name__ == "__main__":
    print("Feedback manager loaded.")
    print("No labels were changed.")
    print("Use record_verified_feedback(...) after an actual review.")
