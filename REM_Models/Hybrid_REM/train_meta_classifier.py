
import json
import warnings
from pathlib import Path
from collections import Counter

import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path.cwd()

DATA_PATH = ROOT / "REM_Datasets/04_Unified_Dataset/processed/public_labeled_candidate.csv"
MANIFEST_PATH = ROOT / "REM_Models/API_Sequence_Grouped/grouped_split_manifest.csv"
CONFLICT_PATH = ROOT / "REM_Models/API_Sequence_Grouped/quarantined_conflicting_sequences.csv"

HYBRID_DIR = ROOT / "REM_Models/Hybrid_REM"
OOF_DIR = HYBRID_DIR / "OOF_Predictions"
ARTIFACT_DIR = HYBRID_DIR / "Artifacts"
REPORT_DIR = HYBRID_DIR / "Reports"

RF_PATH = ROOT / "REM_Models/API_Sequence_Grouped/grouped_api_sequence_rf.joblib"
LSTM_PATH = ROOT / "REM_Models/LSTM_Baseline/rem_lstm.keras"
VOCAB_PATH = ROOT / "REM_Models/LSTM_Baseline/vocabulary.json"
IF_PATH = ROOT / "REM_Models/Isolation_Forest_Baseline/isolation_forest.joblib"

for directory in [ARTIFACT_DIR, REPORT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# 1. Check required files
# ---------------------------------------------------------

required_files = [
    DATA_PATH,
    MANIFEST_PATH,
    CONFLICT_PATH,
    OOF_DIR / "train_oof_rf_if.csv",
    OOF_DIR / "train_oof_lstm.csv",
    RF_PATH,
    LSTM_PATH,
    VOCAB_PATH,
    IF_PATH,
]

missing = [str(path) for path in required_files if not path.exists()]
if missing:
    raise FileNotFoundError(
        "Required files are missing:\n" + "\n".join(missing)
    )

print("[1/8] Loading dataset and split manifest...")

# ---------------------------------------------------------
# 2. Load dataset and construct API sequences
# ---------------------------------------------------------

df = pd.read_csv(DATA_PATH, low_memory=False)
manifest = pd.read_csv(MANIFEST_PATH)

api_cols = [str(i) for i in range(153)]

required_columns = {"sha256", "source_label", *api_cols}
missing_columns = required_columns - set(df.columns)
if missing_columns:
    raise ValueError(f"Dataset columns missing: {sorted(missing_columns)}")

manifest_columns = {"sha256", "source_label", "sequence_hash", "split"}
if not manifest_columns.issubset(manifest.columns):
    raise ValueError(
        f"Manifest must contain {sorted(manifest_columns)}"
    )

df["sha256"] = df["sha256"].astype(str).str.strip()
manifest["sha256"] = manifest["sha256"].astype(str).str.strip()

df["source_label"] = pd.to_numeric(
    df["source_label"], errors="coerce"
)
df = df[df["source_label"].isin([0, 1])].copy()
df["source_label"] = df["source_label"].astype(int)

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

df["tokens"] = df[api_cols].apply(clean_tokens, axis=1)
df["sequence_text"] = df["tokens"].apply(lambda x: " ".join(x))

# Recompute the sequence hash in the same manner as the baseline scripts.
import hashlib

df["sequence_hash"] = df["tokens"].map(
    lambda tokens: hashlib.sha256(
        " ".join(tokens).encode("utf-8")
    ).hexdigest()
)

# Verify sequence hashes against the saved split manifest.
hash_check = df.merge(
    manifest[["sha256", "sequence_hash", "split"]],
    on="sha256",
    how="inner",
    suffixes=("_data", "_manifest"),
    validate="one_to_one",
)

mismatches = hash_check[
    hash_check["sequence_hash_data"]
    != hash_check["sequence_hash_manifest"]
]

if len(mismatches):
    raise ValueError(
        f"{len(mismatches)} dataset sequences do not match the manifest. "
        "Stop and investigate the sequence preprocessing."
    )

# Remove conflicting sequence groups.
conflicts = pd.read_csv(CONFLICT_PATH)
if "sequence_hash" in conflicts.columns:
    conflict_hashes = set(
        conflicts["sequence_hash"].dropna().astype(str)
    )
elif "sequence_hash" in df.columns:
    label_counts = df.groupby("sequence_hash")["source_label"].nunique()
    conflict_hashes = set(label_counts[label_counts > 1].index)
else:
    conflict_hashes = set()

df = df[~df["sequence_hash"].isin(conflict_hashes)].copy()

# Attach split assignments, preserving only the official manifest membership.
df = df.drop(columns=["sequence_hash"]).merge(
    manifest[["sha256", "sequence_hash", "split"]],
    on="sha256",
    how="inner",
    validate="one_to_one",
)

# Ensure there is one row per sequence.
df = df.drop_duplicates("sequence_hash", keep="first").copy()

if df["sequence_hash"].duplicated().any():
    raise ValueError("Duplicate sequences remain after deduplication.")

for split_name in ["train", "validation", "test"]:
    count = int((df["split"] == split_name).sum())
    print(f"    {split_name}: {count} unique sequences")

# ---------------------------------------------------------
# 3. Load and validate OOF predictions
# ---------------------------------------------------------

print("[2/8] Loading out-of-fold predictions...")

oof_rf_if = pd.read_csv(OOF_DIR / "train_oof_rf_if.csv")
oof_lstm = pd.read_csv(OOF_DIR / "train_oof_lstm.csv")

required_rf_if = {
    "sha256", "sequence_hash", "label",
    "rf_oof_score", "if_oof_anomaly_score",
}
required_lstm = {
    "sha256", "sequence_hash", "label", "lstm_oof_score",
}

if not required_rf_if.issubset(oof_rf_if.columns):
    raise ValueError(
        f"RF/IF OOF columns missing: "
        f"{sorted(required_rf_if - set(oof_rf_if.columns))}"
    )

if not required_lstm.issubset(oof_lstm.columns):
    raise ValueError(
        f"LSTM OOF columns missing: "
        f"{sorted(required_lstm - set(oof_lstm.columns))}"
    )

oof = oof_rf_if.merge(
    oof_lstm[
        ["sha256", "sequence_hash", "label", "lstm_oof_score"]
    ],
    on=["sha256", "sequence_hash", "label"],
    how="inner",
    validate="one_to_one",
)

oof["label"] = pd.to_numeric(oof["label"], errors="raise").astype(int)

train_hashes = set(
    df.loc[df["split"] == "train", "sequence_hash"].astype(str)
)

if not set(oof["sequence_hash"].astype(str)).issubset(train_hashes):
    raise ValueError("OOF predictions contain sequences outside training.")

if oof["sequence_hash"].duplicated().any():
    raise ValueError("Duplicate sequences found in OOF predictions.")

oof_features = [
    "rf_oof_score",
    "lstm_oof_score",
    "if_oof_anomaly_score",
]

oof = oof.dropna(subset=oof_features + ["label"]).copy()

if not oof["label"].isin([0, 1]).all():
    raise ValueError("OOF labels must be 0 or 1.")

# Ensure labels agree with the source dataset.
label_lookup = df.set_index("sequence_hash")["source_label"].to_dict()
for row in oof[["sequence_hash", "label"]].itertuples(index=False):
    expected = label_lookup.get(str(row.sequence_hash))
    if expected is None or int(expected) != int(row.label):
        raise ValueError("OOF label mismatch with the dataset.")

X_oof = oof[oof_features].astype(float)
y_oof = oof["label"].astype(int)

if y_oof.nunique() != 2:
    raise ValueError("OOF training set must contain both classes.")

# ---------------------------------------------------------
# 4. Train the meta-classifier on OOF scores only
# ---------------------------------------------------------

print("[3/8] Training the meta-classifier...")

meta_model = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
    ),
)

meta_model.fit(X_oof, y_oof)

# ---------------------------------------------------------
# 5. Load base models
# ---------------------------------------------------------

print("[4/8] Loading trained base models...")

rf_model = joblib.load(RF_PATH)

if_artifact = joblib.load(IF_PATH)
if not isinstance(if_artifact, dict):
    raise ValueError("Unexpected Isolation Forest artifact format.")

if "vectorizer" not in if_artifact or "model" not in if_artifact:
    raise ValueError(
        "Isolation Forest artifact must contain 'vectorizer' and 'model'."
    )

if_vectorizer = if_artifact["vectorizer"]
if_model = if_artifact["model"]

with open(VOCAB_PATH, "r", encoding="utf-8") as file:
    vocab_data = json.load(file)

if isinstance(vocab_data, dict) and "word_index" in vocab_data:
    vocabulary = vocab_data["word_index"]
elif isinstance(vocab_data, dict):
    vocabulary = vocab_data
else:
    raise ValueError("Unsupported vocabulary.json format.")

vocabulary = {
    str(token): int(index)
    for token, index in vocabulary.items()
}

import tensorflow as tf

lstm_model = tf.keras.models.load_model(
    LSTM_PATH,
    compile=False,
)

# Infer the required sequence length from the model's input shape.
input_shape = lstm_model.input_shape
if isinstance(input_shape, list):
    input_shape = input_shape[0]

MAX_LEN = input_shape[1]
if MAX_LEN is None:
    MAX_LEN = 153
MAX_LEN = int(MAX_LEN)

print("    LSTM sequence length:", MAX_LEN)
print("    Vocabulary entries:", len(vocabulary))

def encode_tokens(tokens):
    # Keras Tokenizer-style OOV ID is usually 1.
    oov_id = 1
    pad_id = 0

    ids = [
        vocabulary.get(token, oov_id)
        for token in tokens[:MAX_LEN]
    ]

    ids += [pad_id] * (MAX_LEN - len(ids))
    return ids

# ---------------------------------------------------------
# 6. Match the Isolation Forest feature extraction exactly
# ---------------------------------------------------------

def extract_if_features(tokens):
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

# ---------------------------------------------------------
# 7. Score validation and test sets
# ---------------------------------------------------------

print("[5/8] Scoring validation and test sets...")

def score_split(part):
    if part.empty:
        raise ValueError("A required split is empty.")

    tokens = part["tokens"].tolist()
    texts = [" ".join(sequence) for sequence in tokens]

    # RF probability for class 1.
    rf_scores = rf_model.predict_proba(texts)[:, 1]

    # LSTM malware probability.
    encoded = np.asarray(
        [encode_tokens(sequence) for sequence in tokens],
        dtype=np.int32,
    )
    lstm_scores = lstm_model.predict(
        encoded, verbose=0
    ).reshape(-1)

    # Higher IF score means more anomalous.
    feature_dicts = [extract_if_features(sequence) for sequence in tokens]
    X_if = if_vectorizer.transform(feature_dicts)
    if_scores = -if_model.score_samples(X_if)

    base_scores = pd.DataFrame({
        "rf_oof_score": rf_scores,
        "lstm_oof_score": lstm_scores,
        "if_oof_anomaly_score": if_scores,
    })

    hybrid_scores = meta_model.predict_proba(base_scores)[:, 1]

    result = part[
        ["sha256", "sequence_hash", "source_label", "split"]
    ].copy()

    result["rf_score"] = rf_scores
    result["lstm_score"] = lstm_scores
    result["if_anomaly_score"] = if_scores
    result["hybrid_score"] = hybrid_scores

    return result

validation = score_split(
    df[df["split"] == "validation"].copy()
)
test = score_split(
    df[df["split"] == "test"].copy()
)

# ---------------------------------------------------------
# 8. Select threshold on validation and evaluate test once
# ---------------------------------------------------------

print("[6/8] Selecting threshold using validation data...")

def choose_threshold(y_true, scores, minimum_precision=0.80):
    rows = []

    candidates = np.unique(
        np.concatenate([
            np.linspace(0.01, 0.99, 199),
            np.asarray(scores, dtype=float),
        ])
    )

    for threshold in candidates:
        predictions = (scores >= threshold).astype(int)

        rows.append({
            "threshold": float(threshold),
            "precision": precision_score(
                y_true, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_true, predictions, zero_division=0
            ),
            "f1": f1_score(
                y_true, predictions, zero_division=0
            ),
        })

    threshold_table = pd.DataFrame(rows)
    eligible = threshold_table[
        threshold_table["precision"] >= minimum_precision
    ]

    if not eligible.empty:
        best = eligible.sort_values(
            ["recall", "f1", "threshold"],
            ascending=[False, False, False],
        ).iloc[0]
        rule = "maximum validation recall with precision >= 0.80"
    else:
        best = threshold_table.sort_values(
            ["f1", "precision", "threshold"],
            ascending=[False, False, False],
        ).iloc[0]
        rule = "precision target unmet; best validation F1"

    return float(best["threshold"]), threshold_table, rule

threshold, threshold_table, threshold_rule = choose_threshold(
    validation["source_label"].to_numpy(),
    validation["hybrid_score"].to_numpy(),
)

def evaluate(part, selected_threshold):
    y_true = part["source_label"].astype(int).to_numpy()
    scores = part["hybrid_score"].to_numpy()
    predictions = (scores >= selected_threshold).astype(int)

    return {
        "samples": int(len(y_true)),
        "threshold": float(selected_threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "malware_precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "malware_recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "malware_f1": float(
            f1_score(y_true, predictions, zero_division=0)
        ),
        "confusion_matrix_labels_0_1": confusion_matrix(
            y_true, predictions, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            y_true,
            predictions,
            labels=[0, 1],
            target_names=["benign", "malicious"],
            output_dict=True,
            zero_division=0,
        ),
    }

validation_metrics = evaluate(validation, threshold)

# Freeze the threshold before calculating test metrics.
test_metrics = evaluate(test, threshold)

print("[7/8] Saving models, predictions, and reports...")

joblib.dump(
    meta_model,
    ARTIFACT_DIR / "rem_meta_classifier.joblib",
)

oof.to_csv(
    REPORT_DIR / "meta_training_oof.csv",
    index=False,
)
validation.to_csv(
    REPORT_DIR / "validation_hybrid_predictions.csv",
    index=False,
)
test.to_csv(
    REPORT_DIR / "test_hybrid_predictions.csv",
    index=False,
)
threshold_table.to_csv(
    REPORT_DIR / "validation_threshold_metrics.csv",
    index=False,
)

report = {
    "model": "REM Hybrid Stacking Baseline",
    "base_models": [
        "Grouped API-sequence Random Forest",
        "LSTM",
        "Isolation Forest",
    ],
    "meta_model": "Imputer + StandardScaler + LogisticRegression",
    "oof_training_samples": int(len(oof)),
    "validation_samples": int(len(validation)),
    "test_samples": int(len(test)),
    "oof_label_counts": {
        str(label): int(count)
        for label, count in y_oof.value_counts().sort_index().items()
    },
    "threshold_selection_rule": threshold_rule,
    "selected_threshold": threshold,
    "validation_metrics": validation_metrics,
    "test_metrics": test_metrics,
    "caveats": [
        "Base model artifacts must have been trained using the official training split.",
        "Threshold selection uses validation data only.",
        "Isolation Forest scores are anomaly scores, not calibrated probabilities.",
        "The LSTM vocabulary and padding must match its original training configuration.",
        "OOF and full-training base-model score distributions can differ.",
    ],
}

with open(
    REPORT_DIR / "hybrid_metrics.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(report, file, indent=2)

print("[8/8] Completed.")

print("\n========== REM HYBRID RESULTS ==========")
print("OOF training samples:", len(oof))
print("Validation samples:", len(validation))
print("Test samples:", len(test))
print("Selected threshold:", round(threshold, 4))
print("Threshold rule:", threshold_rule)

print("\nVALIDATION METRICS")
print(json.dumps(validation_metrics, indent=2))

print("\nTEST METRICS")
print(json.dumps(test_metrics, indent=2))

print("\nSaved artifacts:", ARTIFACT_DIR)
print("Saved reports:", REPORT_DIR)
