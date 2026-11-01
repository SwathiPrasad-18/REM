from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
import warnings
from collections import Counter

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# REM NESTED STACKING EVALUATION
#
# Smoke test:
#   python .\REM_Models\Hybrid_REM\Nested_Evaluation\nested_stacking_evaluation.py --smoke-test
#
# Full nested evaluation:
#   python .\REM_Models\Hybrid_REM\Nested_Evaluation\nested_stacking_evaluation.py --full
#
# Five outer folds, four inner folds.
# Primary evaluation threshold: 0.50.
# ============================================================

SEED = 42
OUTER_FOLDS = 5
INNER_FOLDS = 4
MAX_LEN = 153
LSTM_MAX_EPOCHS = 12
LSTM_BATCH_SIZE = 32
THRESHOLD = 0.50

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)

DATASET_PATH = os.path.join(
    PROJECT_ROOT,
    "REM_Datasets",
    "04_Unified_Dataset",
    "processed",
    "public_labeled_candidate.csv",
)

MANIFEST_PATH = os.path.join(
    PROJECT_ROOT,
    "REM_Models",
    "API_Sequence_Grouped",
    "grouped_split_manifest.csv",
)

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__),
    "Full_Results",
)

SMOKE_DIR = os.path.join(
    os.path.dirname(__file__),
    "Smoke_Test_Results",
)

FEATURE_COLUMNS = ["rf_score", "lstm_score", "if_score"]


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
    except ImportError:
        pass


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def get_api_columns(df: pd.DataFrame) -> list[str]:
    """Return API-call sequence columns named 0, 1, ..., 152."""
    columns = []
    for i in range(MAX_LEN):
        if str(i) in df.columns:
            columns.append(str(i))

    if not columns:
        raise ValueError(
            "No API sequence columns named '0', '1', ... were found."
        )

    return columns


def tokenize_row(row, api_columns: list[str]) -> list[str]:
    tokens = []

    for column in api_columns:
        value = row[column]

        if pd.isna(value):
            continue

        token = str(value).strip()

        if not token or token.lower() in {"nan", "none", "null"}:
            continue

        tokens.append(token)

    return tokens


def sequence_hash(tokens: list[str]) -> str:
    sequence_text = " ".join(tokens)
    return hashlib.sha256(sequence_text.encode("utf-8")).hexdigest()


def api_feature_dict(tokens: list[str]) -> dict:
    """Build sparse-safe behavioral features for Isolation Forest."""
    counts = Counter(tokens)
    unique_count = len(counts)
    length = len(tokens)

    features = {
        f"api_freq::{api}": count
        for api, count in counts.items()
    }

    features["sequence_length"] = length
    features["unique_api_count"] = unique_count
    features["unique_api_ratio"] = (
        unique_count / length if length else 0.0
    )

    transition_counts = Counter(
        f"{tokens[i]}->{tokens[i + 1]}"
        for i in range(len(tokens) - 1)
    )

    for transition, count in transition_counts.items():
        features[f"transition::{transition}"] = count

    return features


def encode_sequences(
    token_lists: list[list[str]],
    vocabulary: dict[str, int],
) -> np.ndarray:
    encoded = np.zeros((len(token_lists), MAX_LEN), dtype=np.int32)

    for i, tokens in enumerate(token_lists):
        for j, token in enumerate(tokens[:MAX_LEN]):
            encoded[i, j] = vocabulary.get(token, 1)

    return encoded


def build_vocabulary(token_lists: list[list[str]]) -> dict[str, int]:
    counter = Counter(
        token
        for tokens in token_lists
        for token in tokens
    )

    # 0 = padding; 1 = unknown
    vocabulary = {"<PAD>": 0, "<UNK>": 1}

    for token, _ in counter.most_common():
        if token not in vocabulary:
            vocabulary[token] = len(vocabulary)

    return vocabulary


def build_lstm(vocab_size: int):
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(MAX_LEN,)),
            tf.keras.layers.Embedding(
                input_dim=vocab_size,
                output_dim=64,
                mask_zero=True,
            ),
            tf.keras.layers.SpatialDropout1D(0.20),
            tf.keras.layers.LSTM(64, dropout=0.20),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dropout(0.30),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    return model


def fit_base_models(
    train_df: pd.DataFrame,
    score_df: pd.DataFrame,
    seed: int,
    context: str = "",
) -> pd.DataFrame:
    """
    Fit RF, Isolation Forest, and LSTM using train_df only.
    Generate scores for score_df.
    All vocabulary/vectorizer fitting is confined to train_df.
    """
    import tensorflow as tf

    set_seed(seed)

    train_tokens = train_df["tokens"].tolist()
    score_tokens = score_df["tokens"].tolist()

    y_train = train_df["source_label"].astype(int).to_numpy()

    if len(np.unique(y_train)) < 2:
        raise ValueError(
            f"{context}: training fold contains only one class."
        )

    # --------------------------------------------------------
    # 1. Random Forest: fold-local TF-IDF
    # --------------------------------------------------------
    train_texts = [" ".join(tokens) for tokens in train_tokens]
    score_texts = [" ".join(tokens) for tokens in score_tokens]

    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        max_features=50000,
        dtype=np.float32,
    )

    X_train_rf = tfidf.fit_transform(train_texts)
    X_score_rf = tfidf.transform(score_texts)

    rf = RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1,
    )

    rf.fit(X_train_rf, y_train)

    rf_classes = list(rf.classes_)
    positive_index = rf_classes.index(1)
    rf_score = rf.predict_proba(X_score_rf)[:, positive_index]

    # --------------------------------------------------------
    # 2. Isolation Forest: fit on presumed-benign fold samples
    # --------------------------------------------------------
    train_feature_dicts = [
        api_feature_dict(tokens) for tokens in train_tokens
    ]
    score_feature_dicts = [
        api_feature_dict(tokens) for tokens in score_tokens
    ]

    vectorizer = DictVectorizer(sparse=True)

    X_train_if = vectorizer.fit_transform(train_feature_dicts)
    X_score_if = vectorizer.transform(score_feature_dicts)

    benign_mask = y_train == 0

    if benign_mask.sum() < 2:
        raise ValueError(
            f"{context}: insufficient benign samples for Isolation Forest."
        )

    isolation_forest = IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=seed,
        n_jobs=-1,
    )

    isolation_forest.fit(X_train_if[benign_mask])

    # Larger scores indicate more anomalous behavior.
    if_score = -isolation_forest.decision_function(X_score_if)

    # --------------------------------------------------------
    # 3. LSTM: vocabulary learned from fold training samples only
    # --------------------------------------------------------
    vocabulary = build_vocabulary(train_tokens)

    X_train_lstm = encode_sequences(train_tokens, vocabulary)
    X_score_lstm = encode_sequences(score_tokens, vocabulary)

    tf.keras.backend.clear_session()
    set_seed(seed)

    lstm = build_lstm(len(vocabulary))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
        )
    ]

    lstm.fit(
        X_train_lstm,
        y_train,
        epochs=LSTM_MAX_EPOCHS,
        batch_size=LSTM_BATCH_SIZE,
        validation_split=0.15,
        callbacks=callbacks,
        verbose=0,
        shuffle=True,
    )

    lstm_score = lstm.predict(
        X_score_lstm,
        batch_size=LSTM_BATCH_SIZE,
        verbose=0,
    ).reshape(-1)

    result = pd.DataFrame(
        {
            "rf_score": rf_score,
            "lstm_score": lstm_score,
            "if_score": if_score,
        },
        index=score_df.index,
    )

    return result


def calculate_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float = THRESHOLD,
) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    predictions = (scores >= threshold).astype(int)

    cm = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "threshold": float(threshold),
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(
            average_precision_score(y_true, scores)
        ),
        "malware_precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "malware_recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "malware_f1": float(
            f1_score(y_true, predictions, zero_division=0)
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "confusion_matrix": cm.tolist(),
    }

    return metrics


def load_development_data() -> pd.DataFrame:
    if not os.path.isfile(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    if not os.path.isfile(MANIFEST_PATH):
        raise FileNotFoundError(f"Split manifest not found: {MANIFEST_PATH}")

    print("Loading dataset...")
    df = pd.read_csv(DATASET_PATH, low_memory=False)
    manifest = pd.read_csv(MANIFEST_PATH, low_memory=False)

    if "source_label" not in df.columns:
        raise ValueError("Dataset is missing 'source_label'.")

    if "sha256" not in df.columns:
        raise ValueError("Dataset is missing 'sha256'.")

    required_manifest = {"sha256", "split"}
    missing = required_manifest - set(manifest.columns)

    if missing:
        raise ValueError(
            f"Manifest is missing required columns: {sorted(missing)}"
        )

    api_columns = get_api_columns(df)

    df["source_label"] = pd.to_numeric(
        df["source_label"], errors="coerce"
    )
    df = df[df["source_label"].isin([0, 1])].copy()
    df["source_label"] = df["source_label"].astype(int)

    # Normalize SHA-256 identifiers before joining.
    df["sha256"] = df["sha256"].astype(str).str.strip().str.lower()
    manifest["sha256"] = (
        manifest["sha256"].astype(str).str.strip().str.lower()
    )

    if manifest["sha256"].duplicated().any():
        raise ValueError(
            "The split manifest contains duplicate SHA-256 identifiers."
        )

    # Join the existing official split assignments.
    df = df.merge(
        manifest[["sha256", "split"]],
        on="sha256",
        how="inner",
        validate="many_to_one",
    )

    print(f"Rows matched to manifest: {len(df)}")

    # Tokenize and compute hashes consistently with the prior OOF scripts.
    print("Tokenizing API sequences...")
    df["tokens"] = df.apply(
        lambda row: tokenize_row(row, api_columns),
        axis=1,
    )
    df["sequence_hash"] = df["tokens"].apply(sequence_hash)

    # Exclude sequences with conflicting labels.
    label_counts = df.groupby("sequence_hash")["source_label"].nunique()
    conflicting_hashes = set(label_counts[label_counts > 1].index)

    if conflicting_hashes:
        print(
            "Excluding conflicting sequence groups:",
            len(conflicting_hashes),
        )
        df = df[
            ~df["sequence_hash"].isin(conflicting_hashes)
        ].copy()

    # A sequence must not appear more than once in this evaluation pool.
    df = df.drop_duplicates(
        subset="sequence_hash",
        keep="first",
    ).copy()

    # Development pool only: official training split.
    # Official validation/test splits remain untouched.
    development = df[df["split"].astype(str).str.lower() == "train"].copy()

    if development.empty:
        raise ValueError(
            "No rows found for split='train'. Check manifest split names."
        )

    development = development.reset_index(drop=True)

    if development["sequence_hash"].duplicated().any():
        raise AssertionError(
            "Duplicate sequences remain after deduplication."
        )

    print(f"Development sequences: {len(development)}")
    print("Class counts:")
    print(development["source_label"].value_counts().sort_index())

    return development


def run_inner_oof(
    outer_train: pd.DataFrame,
    inner_splits: int,
    outer_fold_number: int,
) -> pd.DataFrame:
    """Generate inner-fold OOF scores for fitting the meta-classifier."""
    y = outer_train["source_label"].to_numpy()
    groups = outer_train["sequence_hash"].to_numpy()

    splitter = StratifiedGroupKFold(
        n_splits=inner_splits,
        shuffle=True,
        random_state=SEED + outer_fold_number,
    )

    oof_scores = pd.DataFrame(
        np.nan,
        index=outer_train.index,
        columns=FEATURE_COLUMNS,
    )

    for inner_fold, (fit_positions, hold_positions) in enumerate(
        splitter.split(
            np.zeros(len(outer_train)),
            y,
            groups,
        ),
        start=1,
    ):
        fit_df = outer_train.iloc[fit_positions].copy()
        hold_df = outer_train.iloc[hold_positions].copy()

        print(
            f"  Outer {outer_fold_number}, inner {inner_fold}/{inner_splits}: "
            f"fit={len(fit_df)}, score={len(hold_df)}",
            flush=True,
        )

        scores = fit_base_models(
            fit_df,
            hold_df,
            seed=SEED + outer_fold_number * 100 + inner_fold,
            context=f"outer-{outer_fold_number}-inner-{inner_fold}",
        )

        oof_scores.loc[hold_df.index, FEATURE_COLUMNS] = (
            scores[FEATURE_COLUMNS].to_numpy()
        )

    if oof_scores.isna().any().any():
        raise AssertionError(
            "Inner OOF generation left missing scores."
        )

    return oof_scores


def evaluate_outer_fold(
    development: pd.DataFrame,
    train_positions: np.ndarray,
    test_positions: np.ndarray,
    fold_number: int,
    output_dir: str,
) -> tuple[dict, pd.DataFrame]:
    outer_train = development.iloc[train_positions].copy()
    outer_test = development.iloc[test_positions].copy()

    print("\n" + "=" * 72)
    print(
        f"OUTER FOLD {fold_number}: "
        f"train={len(outer_train)}, held-out={len(outer_test)}"
    )
    print("=" * 72)

    # Step 1: inner OOF scores to train the meta-classifier.
    inner_oof = run_inner_oof(
        outer_train,
        inner_splits=INNER_FOLDS,
        outer_fold_number=fold_number,
    )

    y_meta = outer_train["source_label"].astype(int).to_numpy()

    # Keep feature names consistent during fit and predict.
    meta_model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "logistic_regression",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=SEED,
                ),
            ),
        ]
    )

    meta_model.fit(
        inner_oof[FEATURE_COLUMNS],
        y_meta,
    )

    # Step 2: refit all base models on the complete outer training fold.
    # The outer held-out fold is used only for prediction/evaluation.
    outer_scores = fit_base_models(
        outer_train,
        outer_test,
        seed=SEED + fold_number * 10000,
        context=f"outer-{fold_number}-refit",
    )

    hybrid_scores = meta_model.predict_proba(
        outer_scores[FEATURE_COLUMNS]
    )[:, 1]

    y_test = outer_test["source_label"].astype(int).to_numpy()

    metrics = calculate_metrics(
        y_test,
        hybrid_scores,
        threshold=THRESHOLD,
    )
    metrics["outer_fold"] = fold_number
    metrics["outer_train_samples"] = len(outer_train)
    metrics["outer_test_samples"] = len(outer_test)

    predictions = pd.DataFrame(
        {
            "outer_fold": fold_number,
            "sha256": outer_test["sha256"].to_numpy(),
            "sequence_hash": outer_test["sequence_hash"].to_numpy(),
            "y_true": y_test,
            "rf_score": outer_scores["rf_score"].to_numpy(),
            "lstm_score": outer_scores["lstm_score"].to_numpy(),
            "if_score": outer_scores["if_score"].to_numpy(),
            "hybrid_score": hybrid_scores,
            "y_pred": (hybrid_scores >= THRESHOLD).astype(int),
        }
    )

    fold_prefix = os.path.join(
        output_dir,
        f"outer_fold_{fold_number}",
    )

    predictions.to_csv(
        f"{fold_prefix}_predictions.csv",
        index=False,
    )

    with open(
        f"{fold_prefix}_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(json_safe(metrics), file, indent=2)

    print(
        f"Fold {fold_number} results: "
        f"accuracy={metrics['accuracy']:.4f}, "
        f"AUC={metrics['roc_auc']:.4f}, "
        f"precision={metrics['malware_precision']:.4f}, "
        f"recall={metrics['malware_recall']:.4f}, "
        f"F1={metrics['malware_f1']:.4f}",
        flush=True,
    )

    return metrics, predictions


def write_summary(
    metrics_list: list[dict],
    all_predictions: pd.DataFrame,
    output_dir: str,
    mode: str,
) -> None:
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(
        os.path.join(output_dir, "fold_metrics.csv"),
        index=False,
    )

    all_predictions.to_csv(
        os.path.join(output_dir, "all_outer_predictions.csv"),
        index=False,
    )

    y_true = all_predictions["y_true"].to_numpy()
    scores = all_predictions["hybrid_score"].to_numpy()

    pooled_metrics = calculate_metrics(
        y_true,
        scores,
        threshold=THRESHOLD,
    )

    metric_names = [
        "accuracy",
        "roc_auc",
        "average_precision",
        "malware_precision",
        "malware_recall",
        "malware_f1",
    ]

    fold_summary = {}
    for metric_name in metric_names:
        values = metrics_df[metric_name].astype(float)
        fold_summary[metric_name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }

    aggregate = {
        "evaluation_mode": mode,
        "outer_folds": len(metrics_list),
        "inner_folds_per_outer_fold": INNER_FOLDS,
        "fixed_threshold": THRESHOLD,
        "fold_metric_mean_and_std": fold_summary,
        "pooled_out_of_fold_metrics": pooled_metrics,
        "methodology_notes": [
            "All base-model fitting and meta-classifier training occurred "
            "inside each outer training fold.",
            "The outer held-out fold was not used for model fitting or "
            "threshold selection.",
            "The evaluation pool is the official training split only.",
            "The official validation and test splits were not included in "
            "this nested cross-validation.",
            "Isolation Forest anomaly scores are inputs to the meta-model; "
            "they are not standalone malware probabilities.",
            "The fixed 0.50 threshold was used for primary outer-fold "
            "evaluation.",
        ],
    }

    with open(
        os.path.join(output_dir, "aggregate_metrics.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(json_safe(aggregate), file, indent=2)

    lines = [
        "# REM Nested Stacking Evaluation",
        "",
        f"- Mode: {mode}",
        f"- Outer folds completed: {len(metrics_list)}",
        f"- Inner folds per outer fold: {INNER_FOLDS}",
        f"- Fixed threshold: {THRESHOLD}",
        f"- Total outer predictions: {len(all_predictions)}",
        "",
        "## Per-fold results",
        "",
        "| Fold | Accuracy | ROC-AUC | Precision | Recall | F1 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]

    for item in metrics_list:
        lines.append(
            f"| {item['outer_fold']} "
            f"| {item['accuracy']:.4f} "
            f"| {item['roc_auc']:.4f} "
            f"| {item['malware_precision']:.4f} "
            f"| {item['malware_recall']:.4f} "
            f"| {item['malware_f1']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Mean and standard deviation across outer folds",
            "",
            "| Metric | Mean | Standard deviation |",
            "|---|---:|---:|",
        ]
    )

    for metric_name in metric_names:
        item = fold_summary[metric_name]
        lines.append(
            f"| {metric_name} | {item['mean']:.4f} "
            f"| {item['std']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Pooled outer-fold predictions",
            "",
            f"- Accuracy: {pooled_metrics['accuracy']:.4f}",
            f"- ROC-AUC: {pooled_metrics['roc_auc']:.4f}",
            f"- Average precision: "
            f"{pooled_metrics['average_precision']:.4f}",
            f"- Malware precision: "
            f"{pooled_metrics['malware_precision']:.4f}",
            f"- Malware recall: {pooled_metrics['malware_recall']:.4f}",
            f"- Malware F1: {pooled_metrics['malware_f1']:.4f}",
            f"- TN: {pooled_metrics['tn']}",
            f"- FP: {pooled_metrics['fp']}",
            f"- FN: {pooled_metrics['fn']}",
            f"- TP: {pooled_metrics['tp']}",
            "",
            "## Interpretation",
            "",
            "These results estimate performance across outer folds drawn "
            "from the official training split. They do not replace the "
            "final evaluation on the untouched official test split. "
            "Report the per-fold variability alongside pooled metrics.",
            "",
        ]
    )

    with open(
        os.path.join(output_dir, "summary_report.md"),
        "w",
        encoding="utf-8",
    ) as file:
        file.write("\n".join(lines))

    print("\nAggregate results:")
    print(json.dumps(json_safe(aggregate), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="REM nested stacking evaluation."
    )

    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one outer fold to verify the complete pipeline.",
    )
    modes.add_argument(
        "--full",
        action="store_true",
        help="Run all five outer folds with four inner folds each.",
    )

    args = parser.parse_args()

    set_seed(SEED)

    if args.smoke_test:
        output_dir = SMOKE_DIR
        mode = "smoke_test"
        outer_fold_count = 1
    else:
        output_dir = OUTPUT_DIR
        mode = "full_nested_cv"
        outer_fold_count = OUTER_FOLDS

    os.makedirs(output_dir, exist_ok=True)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Output directory: {output_dir}")
    print(f"Evaluation mode: {mode}")
    print(f"Fixed threshold: {THRESHOLD}")

    development = load_development_data()

    y = development["source_label"].to_numpy()
    groups = development["sequence_hash"].to_numpy()

    outer_splitter = StratifiedGroupKFold(
        n_splits=OUTER_FOLDS,
        shuffle=True,
        random_state=SEED,
    )

    all_outer_splits = list(
        outer_splitter.split(
            np.zeros(len(development)),
            y,
            groups,
        )
    )

    selected_splits = all_outer_splits[:outer_fold_count]

    print(
        f"Outer folds selected: {len(selected_splits)} "
        f"of {OUTER_FOLDS}"
    )

    metrics_list = []
    prediction_frames = []
    start_time = time.time()

    for fold_number, (train_positions, test_positions) in enumerate(
        selected_splits,
        start=1,
    ):
        metrics, predictions = evaluate_outer_fold(
            development=development,
            train_positions=train_positions,
            test_positions=test_positions,
            fold_number=fold_number,
            output_dir=output_dir,
        )

        metrics_list.append(metrics)
        prediction_frames.append(predictions)

        elapsed = (time.time() - start_time) / 60
        print(f"Elapsed time: {elapsed:.1f} minutes")

    all_predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    if all_predictions["sequence_hash"].duplicated().any():
        raise AssertionError(
            "A sequence was evaluated in multiple outer test folds."
        )

    write_summary(
        metrics_list=metrics_list,
        all_predictions=all_predictions,
        output_dir=output_dir,
        mode=mode,
    )

    print("\nEvaluation completed.")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()