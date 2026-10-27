"""REM inference adapter: model scoring only; no training or response actions."""
from pathlib import Path
import json

import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "REM_Models"

RF_PATH = MODELS / "API_Sequence_Grouped" / "grouped_api_sequence_rf.joblib"
RF_THRESHOLD_PATH = MODELS / "API_Sequence_Grouped" / "Threshold_Tuning" / "threshold_selection.json"
LSTM_PATH = MODELS / "LSTM_Baseline" / "rem_lstm.keras"
VOCAB_PATH = MODELS / "LSTM_Baseline" / "vocabulary.json"
LSTM_METRICS_PATH = MODELS / "LSTM_Baseline" / "metrics.json"
IF_PATH = MODELS / "Isolation_Forest_Baseline" / "isolation_forest.joblib"
IF_METRICS_PATH = MODELS / "Isolation_Forest_Baseline" / "metrics.json"
META_PATH = MODELS / "Hybrid_REM" / "Artifacts" / "rem_meta_classifier.joblib"
HYBRID_METRICS_PATH = MODELS / "Hybrid_REM" / "Reports" / "hybrid_metrics.json"


def read_json(path):
    with path.open(encoding="utf-8-sig") as f:
        return json.load(f)


def extract_threshold(data, fallback=None):
    """Read common threshold keys without silently guessing."""
    for key in ("selected_threshold", "threshold", "best_threshold"):
        value = data.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    if fallback is not None:
        return float(fallback)
    raise ValueError("No recognized numeric threshold found in metrics JSON.")


def normalize_tokens(api_calls):
    if isinstance(api_calls, str):
        tokens = api_calls.split()
    else:
        tokens = list(api_calls)
    return [
        str(token).strip()
        for token in tokens
        if token is not None and str(token).strip().lower() not in
        {"", "nan", "none"}
    ]


def isolation_features(tokens):
    features = {
        "sequence_length": len(tokens),
        "unique_api_count": len(set(tokens)),
        "unique_api_ratio": len(set(tokens)) / max(len(tokens), 1),
    }

    for token in tokens:
        key = f"api_freq={token}"
        features[key] = features.get(key, 0) + 1

    for left, right in zip(tokens, tokens[1:]):
        key = f"transition={left}->{right}"
        features[key] = features.get(key, 0) + 1

    return features


class REMInference:
    def __init__(self):
        required = [
            RF_PATH, RF_THRESHOLD_PATH, LSTM_PATH, VOCAB_PATH,
            LSTM_METRICS_PATH, IF_PATH, IF_METRICS_PATH,
            META_PATH, HYBRID_METRICS_PATH,
        ]
        missing = [str(p) for p in required if not p.is_file()]
        if missing:
            raise FileNotFoundError("Missing REM artifacts:\n" + "\n".join(missing))

        self.rf = joblib.load(RF_PATH)
        self.rf_threshold = extract_threshold(read_json(RF_THRESHOLD_PATH))

        self.lstm = tf.keras.models.load_model(LSTM_PATH, compile=False)
        vocab_data = read_json(VOCAB_PATH)
        self.vocab = vocab_data.get("word_index", vocab_data)
        self.lstm_threshold = extract_threshold(read_json(LSTM_METRICS_PATH), 0.25)
        self.max_len = int(self.lstm.input_shape[1])

        self.if_artifact = joblib.load(IF_PATH)
        self.if_vectorizer = self.if_artifact["vectorizer"]
        self.if_model = self.if_artifact["model"]
        self.if_threshold = extract_threshold(read_json(IF_METRICS_PATH))

        self.meta = joblib.load(META_PATH)
        self.hybrid_metrics = read_json(HYBRID_METRICS_PATH)
        self.hybrid_threshold = extract_threshold(self.hybrid_metrics)

        expected = {"rf_oof_score", "lstm_oof_score", "if_oof_anomaly_score"}
        actual = set(getattr(self.meta, "feature_names_in_", []))
        if actual and actual != expected:
            raise ValueError(
                f"Unexpected meta-classifier features: {sorted(actual)}"
            )

    def predict_one(self, api_calls):
        tokens = normalize_tokens(api_calls)
        if not tokens:
            raise ValueError("Provide at least one non-empty API-call token.")

        sequence_text = " ".join(tokens)

        # Random Forest: use the fitted pipeline's original text vectorizer.
        rf_probability = float(self.rf.predict_proba([sequence_text])[0, 1])

        # LSTM: preserve the saved vocabulary, unknown ID, and padding.
        encoded = [int(self.vocab.get(token, 1)) for token in tokens]
        padded = pad_sequences(
            [encoded], maxlen=self.max_len, padding="post",
            truncating="post", value=0
        )
        lstm_probability = float(
            np.asarray(self.lstm.predict(padded, verbose=0)).reshape(-1)[0]
        )

        # Isolation Forest: same feature naming convention as training.
        features = isolation_features(tokens)
        X_if = self.if_vectorizer.transform([features])
        if_anomaly_score = float((-self.if_model.score_samples(X_if))[0])

        import pandas as pd

        base_features = pd.DataFrame([{
            "rf_oof_score": rf_probability,
            "lstm_oof_score": lstm_probability,
            "if_oof_anomaly_score": if_anomaly_score,
        }], columns=[
            "rf_oof_score",
            "lstm_oof_score",
            "if_oof_anomaly_score",
        ])
        hybrid_score = float(self.meta.predict_proba(base_features)[0, 1])

        return {
            "api_call_count": len(tokens),
            "models": {
                "random_forest": {
                    "score": rf_probability,
                    "score_type": "malicious-class probability from RF",
                    "threshold": self.rf_threshold,
                    "decision": "malicious" if rf_probability >= self.rf_threshold else "benign",
                },
                "lstm": {
                    "score": lstm_probability,
                    "score_type": "model output probability",
                    "threshold": self.lstm_threshold,
                    "decision": "malicious" if lstm_probability >= self.lstm_threshold else "benign",
                },
                "isolation_forest": {
                    "score": if_anomaly_score,
                    "score_type": "anomaly score; not a probability",
                    "threshold": self.if_threshold,
                    "decision": "anomalous" if if_anomaly_score >= self.if_threshold else "not_anomalous",
                },
                "hybrid": {
                    "score": hybrid_score,
                    "threshold": self.hybrid_threshold,
                    "decision": "malicious" if hybrid_score >= self.hybrid_threshold else "benign",
                    "status": "experimental; validate live score distribution before operational use",
                },
            },
            "actions_taken": [],
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Score an API-call sequence with REM.")
    parser.add_argument("--api-calls", nargs="+", required=True,
                        help="API-call tokens, separated by spaces")
    args = parser.parse_args()

    detector = REMInference()
    print(json.dumps(detector.predict_one(args.api_calls), indent=2))
