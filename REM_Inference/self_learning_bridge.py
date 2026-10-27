from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from REM_Inference.inference import REMInference
from REM_Experiments.Self_Learning.confidence_router import record_prediction


MODEL_VERSION = "REM-hybrid-experimental-v0.1"


def infer_and_queue(api_calls, sample_id):
    if not str(sample_id).strip():
        raise ValueError("A unique sample_id is required.")

    detector = REMInference()
    result = detector.predict_one(api_calls)

    hybrid = result["models"]["hybrid"]
    if hybrid.get("status") != (
        "experimental; validate live score distribution before operational use"
    ):
        raise RuntimeError("Unexpected hybrid status; refusing to queue.")

    # The hybrid score is experimental. This queue entry is for integration
    # testing and analyst review, not an autonomous security decision.
    record = record_prediction(
        probability=hybrid["score"],
        sample_id=sample_id,
        model_version=MODEL_VERSION,
    )

    return {
        "inference": result,
        "queued_prediction": record,
        "training_enabled": False,
        "autonomous_response_enabled": False,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--api-calls", nargs="+", required=True)
    args = parser.parse_args()

    output = infer_and_queue(args.api_calls, args.sample_id)
    print(json.dumps(output, indent=2))
