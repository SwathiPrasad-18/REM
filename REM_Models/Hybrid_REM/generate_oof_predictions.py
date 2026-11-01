
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import StratifiedGroupKFold


SEED = 42
N_SPLITS = 5


def tokens_from_row(row):
    ignored = {"", "nan", "none", "null"}
    return [
        token for value in row
        if (token := str(value).strip()).lower() not in ignored
    ]


def sequence_text(tokens):
    return " ".join(tokens)


def anomaly_features(tokens):
    features = {}
    for token in tokens:
        key = "api_freq=" + token
        features[key] = features.get(key, 0) + 1

    n = len(tokens)
    features["sequence_length"] = n
    features["unique_api_count"] = len(set(tokens))
    features["unique_api_ratio"] = len(set(tokens)) / n if n else 0.0

    for a, b in zip(tokens, tokens[1:]):
        key = "transition=" + a + "->" + b
        features[key] = features.get(key, 0) + 1

    return features


def main():
    root = Path(".")
    data_path = root / "REM_Datasets/04_Unified_Dataset/processed/public_labeled_candidate.csv"
    manifest_path = root / "REM_Models/API_Sequence_Grouped/grouped_split_manifest.csv"
    out_dir = root / "REM_Models/Hybrid_REM/OOF_Predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path, low_memory=False)
    manifest = pd.read_csv(manifest_path)

    api_cols = [str(i) for i in range(153)]
    df["sha256"] = df["sha256"].astype(str)
    manifest["sha256"] = manifest["sha256"].astype(str)
    df["source_label"] = pd.to_numeric(df["source_label"], errors="raise").astype(int)

    df["tokens"] = df[api_cols].apply(tokens_from_row, axis=1)
    df["sequence_hash"] = df["tokens"].map(
        lambda t: hashlib.sha256(sequence_text(t).encode()).hexdigest()
    )

    # Exclude sequences with contradictory labels.
    label_counts = df.groupby("sequence_hash")["source_label"].nunique()
    conflicts = set(label_counts[label_counts > 1].index)
    df = df[~df["sequence_hash"].isin(conflicts)].copy()

    # Preserve the previously defined train/validation/test partition.
    df = df.merge(
        manifest[["sha256", "split"]],
        on="sha256",
        how="inner",
        validate="one_to_one",
    )
    df = df.drop_duplicates("sequence_hash").copy()

    train = df[df["split"] == "train"].reset_index(drop=True)
    y = train["source_label"].to_numpy()
    groups = train["sequence_hash"].to_numpy()

    if len(np.unique(y)) != 2:
        raise ValueError("Training data must contain both labels.")

    # Stratified group folds: identical API sequences stay together.
    cv = StratifiedGroupKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=SEED
    )

    oof_rf = np.full(len(train), np.nan)
    oof_if = np.full(len(train), np.nan)
    fold_ids = np.full(len(train), -1)

    for fold, (fit_idx, hold_idx) in enumerate(
        cv.split(np.zeros(len(train)), y, groups), start=1
    ):
        print(f"Fold {fold}/{N_SPLITS}")

        fit = train.iloc[fit_idx]
        hold = train.iloc[hold_idx]

        # Random Forest: fit TF-IDF on fold-training sequences only.
        tfidf = TfidfVectorizer(
            lowercase=False,
            token_pattern=r"(?u)\b[\w.-]+\b",
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )
        X_fit = tfidf.fit_transform(fit["tokens"].map(sequence_text))
        X_hold = tfidf.transform(hold["tokens"].map(sequence_text))

        rf = RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            random_state=SEED + fold,
            n_jobs=-1,
        )
        rf.fit(X_fit, fit["source_label"])
        positive_index = list(rf.classes_).index(1)
        oof_rf[hold_idx] = rf.predict_proba(X_hold)[:, positive_index]

        # Isolation Forest: fit only on presumed-benign fold-training samples.
        fit_dicts = fit["tokens"].map(anomaly_features).tolist()
        hold_dicts = hold["tokens"].map(anomaly_features).tolist()

        dv = DictVectorizer(sparse=True)
        X_fit_if = dv.fit_transform(fit_dicts)
        X_hold_if = dv.transform(hold_dicts)

        benign_mask = fit["source_label"].to_numpy() == 0
        if benign_mask.sum() < 10:
            raise ValueError(f"Too few benign samples in fold {fold}")

        iso = IsolationForest(
            n_estimators=300,
            contamination="auto",
            random_state=SEED + fold,
            n_jobs=-1,
        )
        iso.fit(X_fit_if[benign_mask])
        oof_if[hold_idx] = -iso.score_samples(X_hold_if)
        fold_ids[hold_idx] = fold

    if np.isnan(oof_rf).any() or np.isnan(oof_if).any():
        raise RuntimeError("Some OOF predictions are missing.")

    result = pd.DataFrame({
        "sha256": train["sha256"],
        "sequence_hash": train["sequence_hash"],
        "label": y,
        "fold": fold_ids,
        "rf_oof_score": oof_rf,
        "if_oof_anomaly_score": oof_if,
    })

    result.to_csv(out_dir / "train_oof_rf_if.csv", index=False)

    metadata = {
        "training_unique_sequences": int(len(train)),
        "folds": N_SPLITS,
        "conflicting_sequence_groups_excluded": len(conflicts),
        "missing_rf_scores": int(result["rf_oof_score"].isna().sum()),
        "missing_if_scores": int(result["if_oof_anomaly_score"].isna().sum()),
        "fold_counts": {
            str(k): int(v)
            for k, v in result["fold"].value_counts().sort_index().items()
        },
        "note": (
            "RF and Isolation Forest scores are out-of-fold training predictions. "
            "LSTM OOF predictions have not yet been generated."
        ),
    }

    with open(out_dir / "oof_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
