import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"

QUEUE_PATH = DATA_DIR / "prediction_queue.csv"
AUDIT_PATH = LOG_DIR / "confidence_router_audit.jsonl"

HIGH_CONFIDENCE_THRESHOLD = 0.90
MANUAL_REVIEW_THRESHOLD = 0.60


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def route_prediction(probability, sample_id=None, model_version="unknown"):
    """Route a prediction for review; confidence alone never verifies a label."""
    probability = float(probability)

    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between 0 and 1.")

    predicted_label = int(probability >= 0.5)
    confidence = max(probability, 1.0 - probability)

    if confidence >= HIGH_CONFIDENCE_THRESHOLD:
        queue_name = "high_confidence_review"
    elif confidence >= MANUAL_REVIEW_THRESHOLD:
        queue_name = "manual_review"
    else:
        queue_name = "uncertain"

    return {
        "sample_id": str(sample_id or uuid.uuid4()),
        "created_at_utc": utc_now(),
        "model_version": str(model_version),
        "malicious_probability": probability,
        "predicted_label": predicted_label,
        "confidence": confidence,
        "queue": queue_name,
        "review_status": "pending",
        "verified_label": None,
        "eligible_for_training": False,
    }


def record_prediction(probability, sample_id=None, model_version="unknown"):
    """Save a routed prediction to the queue and append an audit event."""
    record = route_prediction(probability, sample_id, model_version)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if QUEUE_PATH.exists():
        queue = pd.read_csv(QUEUE_PATH, dtype={"sample_id": str})
    else:
        queue = pd.DataFrame()

    if not queue.empty and queue["sample_id"].astype(str).eq(
        record["sample_id"]
    ).any():
        raise ValueError("sample_id already exists in the prediction queue.")

    pd.DataFrame([record]).to_csv(
        QUEUE_PATH,
        mode="a" if QUEUE_PATH.exists() else "w",
        header=not QUEUE_PATH.exists(),
        index=False,
    )

    with AUDIT_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")

    return record


if __name__ == "__main__":
    examples = [0.97, 0.74, 0.52, 0.12]

    for index, probability in enumerate(examples, start=1):
        result = route_prediction(probability)
        print(
            f"Example {index}: probability={probability:.2f}, "
            f"confidence={result['confidence']:.2f}, "
            f"queue={result['queue']}, "
            f"training_eligible={result['eligible_for_training']}"
        )
