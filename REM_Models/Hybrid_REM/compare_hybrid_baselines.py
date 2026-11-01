
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

ROOT = Path.cwd()
BASE = ROOT / "REM_Models"

DATA_PATH = (
    ROOT / "REM_Datasets/04_Unified_Dataset/processed/"
    "public_labeled_candidate.csv"
)
MANIFEST_PATH = BASE / "API_Sequence_Grouped/grouped_split_manifest.csv"

RF_PATH = BASE / "API_Sequence_Grouped/grouped_api_sequence_rf.joblib"
RF_THRESHOLD_PATH = (
    BASE / "API_Sequence_Grouped/Threshold_Tuning/threshold_selection.json"
)
LSTM_PATH = BASE / "LSTM_Baseline/rem_lstm.keras"
VOCAB_PATH = BASE / "LSTM_Baseline/vocabulary.json"
IF_PATH = BASE / "Isolation_Forest_Baseline/isolation_forest.joblib"

HYBRID_PATH = BASE / "Hybrid_REM/Reports/test_hybrid_predictions.csv"
HYBRID_METRICS_PATH = BASE / "Hybrid_REM/Reports/hybrid_metrics.json"
OUT_DIR = BASE / "Hybrid_REM/Reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data and use the existing official split.
df = pd.read_csv(DATA_PATH, low_memory=False)
manifest = pd.read_csv(MANIFEST_PATH)

api_cols = [str(i) for i in range(153)]

df["sha256"] = df["sha256"].astype(str)
manifest["sha256"] = manifest["sha256"].astype(str)
df["source_label"] = pd.to_numeric(
    df["source_label"], errors="coerce"
)

def clean_tokens(row):
    ignored = {"", "nan", "none", "null"}
    tokens = []
    for value in row:
        if pd.isna(value):
            continue
        token = str(value).strip()
        if token.lower() not in ignored:
            tokens.append(token)
    return tokens

df = df[df["source_label"].isin([0, 1])].copy()
df["source_label"] = df["source_label"].astype(int)
df["tokens"] = df[api_cols].apply(clean_tokens, axis=1)
df["sequence_text"] = df["tokens"].apply(lambda x: " ".join(x))

# Join to the exact split assignments used for model evaluation.
df = df.merge(
    manifest[["sha256", "sequence_hash", "split"]],
    on="sha256",
    how="inner",
    validate="one_to_one",
)

# Preserve one sample per sequence, as in the grouped evaluation.
df = df.drop_duplicates("sequence_hash", keep="first").copy()
test = df[df["split"] == "test"].copy().reset_index(drop=True)

if len(test) != 319:
    raise ValueError(
        f"Expected 319 grouped test sequences; found {len(test)}."
    )

y = test["source_label"].to_numpy(dtype=int)
results = []
predictions = test[
    ["sha256", "sequence_hash", "source_label"]
].copy()

def add_result(name, scores, threshold, score_direction="high"):
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if len(scores) != len(y):
        raise ValueError(f"{name}: prediction count does not match test set.")
    if not np.isfinite(scores).all():
        raise ValueError(f"{name}: scores contain NaN or infinite values.")

    # All models here use higher score = more likely malicious.
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    row = {
        "model": name,
        "samples": len(y),
        "threshold": float(threshold),
        "accuracy": accuracy_score(y, pred),
        "roc_auc": roc_auc_score(y, scores),
        "malware_precision": precision_score(y, pred, zero_division=0),
        "malware_recall": recall_score(y, pred, zero_division=0),
        "malware_f1": f1_score(y, pred, zero_division=0),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }
    results.append(row)
    predictions[name.replace(" ", "_").lower() + "_score"] = scores
    predictions[name.replace(" ", "_").lower() + "_pred"] = pred

# Random Forest
print("Scoring Random Forest...")
rf = joblib.load(RF_PATH)
rf_scores = rf.predict_proba(test["sequence_text"].tolist())[:, 1]

with open(RF_THRESHOLD_PATH, encoding="utf-8") as f:
    rf_threshold = float(json.load(f)["selected_threshold"])

add_result("Random Forest", rf_scores, rf_threshold)

# LSTM
print("Scoring LSTM...")
import tensorflow as tf

lstm = tf.keras.models.load_model(LSTM_PATH, compile=False)
with open(VOCAB_PATH, encoding="utf-8") as f:
    vocab_data = json.load(f)

if isinstance(vocab_data, dict) and "word_index" in vocab_data:
    vocab = vocab_data["word_index"]
else:
    vocab = vocab_data

vocab = {str(k): int(v) for k, v in vocab.items()}
shape = lstm.input_shape
if isinstance(shape, list):
    shape = shape[0]
max_len = int(shape[1] or 153)

def encode(tokens):
    ids = [vocab.get(token, 1) for token in tokens[:max_len]]
    return ids + [0] * (max_len - len(ids))

X_lstm = np.asarray(
    [encode(tokens) for tokens in test["tokens"]],
    dtype=np.int32,
)
lstm_scores = lstm.predict(X_lstm, verbose=0).reshape(-1)

# Use the already-selected validation threshold from the LSTM baseline.
lstm_metrics_path = BASE / "LSTM_Baseline/metrics.json"
with open(lstm_metrics_path, encoding="utf-8") as f:
    lstm_metrics = json.load(f)

lstm_threshold = float(lstm_metrics.get("selected_threshold", 0.25))
add_result("LSTM", lstm_scores, lstm_threshold)

# Isolation Forest
print("Scoring Isolation Forest...")
if_artifact = joblib.load(IF_PATH)
if_vectorizer = if_artifact["vectorizer"]
if_model = if_artifact["model"]

from collections import Counter

def if_features(tokens):
    features = {}
    for token in tokens:
        key = "api_freq=" + token
        features[key] = features.get(key, 0) + 1

    n = len(tokens)
    unique_count = len(set(tokens))
    features["sequence_length"] = n
    features["unique_api_count"] = unique_count
    features["unique_api_ratio"] = unique_count / n if n else 0.0

    for a, b in zip(tokens, tokens[1:]):
        key = "transition=" + a + "->" + b
        features[key] = features.get(key, 0) + 1
    return features

X_if = if_vectorizer.transform(
    [if_features(tokens) for tokens in test["tokens"]]
)
if_scores = -if_model.score_samples(X_if)

if_threshold_path = BASE / "Isolation_Forest_Baseline/metrics.json"
with open(if_threshold_path, encoding="utf-8") as f:
    if_metrics = json.load(f)

if_threshold = float(if_metrics["selected_threshold"])
add_result("Isolation Forest", if_scores, if_threshold)

# Hybrid predictions are saved from the same grouped test split.
print("Loading hybrid scores...")
hybrid = pd.read_csv(HYBRID_PATH)
hybrid["sha256"] = hybrid["sha256"].astype(str)

# Align by SHA256, not by row order.
aligned = test[["sha256", "source_label"]].merge(
    hybrid[["sha256", "source_label", "hybrid_score"]],
    on="sha256",
    how="left",
    suffixes=("_test", "_hybrid"),
    validate="one_to_one",
)

if aligned["hybrid_score"].isna().any():
    raise ValueError("Some test samples have no hybrid prediction.")

if not (
    aligned["source_label_test"].to_numpy()
    == aligned["source_label_hybrid"].to_numpy()
).all():
    raise ValueError("Hybrid labels do not match the test dataset.")

with open(HYBRID_METRICS_PATH, encoding="utf-8") as f:
    hybrid_metrics = json.load(f)

hybrid_threshold = float(hybrid_metrics["selected_threshold"])
add_result(
    "REM Hybrid",
    aligned["hybrid_score"].to_numpy(),
    hybrid_threshold,
)

# Save the comparison and per-sample predictions.
comparison = pd.DataFrame(results)
comparison.to_csv(OUT_DIR / "hybrid_vs_baselines.csv", index=False)
predictions.to_csv(OUT_DIR / "all_models_test_predictions.csv", index=False)

print("\n=== REM MODEL COMPARISON: SAME TEST SET ===")
print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nTest samples:", len(test))
print("Saved comparison:", OUT_DIR / "hybrid_vs_baselines.csv")
print("Saved predictions:", OUT_DIR / "all_models_test_predictions.csv")
