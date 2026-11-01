
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import StratifiedGroupKFold
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input, Embedding, LSTM, Dense, Dropout, SpatialDropout1D
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences


SEED = 42
N_SPLITS = 5
MAX_LEN = 153
MAX_EPOCHS = 12
BATCH_SIZE = 32


def tokens_from_row(row):
    ignored = {"", "nan", "none", "null"}
    return [
        token for value in row
        if (token := str(value).strip()).lower() not in ignored
    ]


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    data_path = Path(
        "REM_Datasets/04_Unified_Dataset/processed/"
        "public_labeled_candidate.csv"
    )
    manifest_path = Path(
        "REM_Models/API_Sequence_Grouped/grouped_split_manifest.csv"
    )
    out_dir = Path("REM_Models/Hybrid_REM/OOF_Predictions")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path, low_memory=False)
    manifest = pd.read_csv(manifest_path)

    api_cols = [str(i) for i in range(153)]
    df["sha256"] = df["sha256"].astype(str)
    manifest["sha256"] = manifest["sha256"].astype(str)
    df["source_label"] = pd.to_numeric(
        df["source_label"], errors="raise"
    ).astype(int)

    df["tokens"] = df[api_cols].apply(tokens_from_row, axis=1)
    df["sequence_hash"] = df["tokens"].map(
        lambda t: hashlib.sha256(
            " ".join(t).encode("utf-8")
        ).hexdigest()
    )

    # Quarantine sequence groups with conflicting labels.
    label_counts = df.groupby("sequence_hash")["source_label"].nunique()
    conflicts = set(label_counts[label_counts > 1].index)
    df = df[~df["sequence_hash"].isin(conflicts)].copy()

    df = df.merge(
        manifest[["sha256", "split"]],
        on="sha256",
        how="inner",
        validate="one_to_one",
    )

    # Same training population as the RF/Isolation Forest OOF stage.
    train = (
        df[df["split"] == "train"]
        .drop_duplicates("sequence_hash")
        .reset_index(drop=True)
    )

    y = train["source_label"].to_numpy()
    groups = train["sequence_hash"].to_numpy()

    oof_scores = np.full(len(train), np.nan)
    fold_ids = np.full(len(train), -1)
    fold_metadata = []

    cv = StratifiedGroupKFold(
        n_splits=N_SPLITS, shuffle=True, random_state=SEED
    )

    for fold, (fit_idx, hold_idx) in enumerate(
        cv.split(np.zeros(len(train)), y, groups), start=1
    ):
        print(f"\nLSTM OOF fold {fold}/{N_SPLITS}")

        fit = train.iloc[fit_idx]
        hold = train.iloc[hold_idx]

        # Fit vocabulary using this fold's training samples only.
        vocab = {"<PAD>": 0, "<UNK>": 1}
        for tokens in fit["tokens"]:
            for token in tokens:
                if token not in vocab:
                    vocab[token] = len(vocab)

        def encode(tokens):
            return [vocab.get(token, 1) for token in tokens]

        X_fit = pad_sequences(
            [encode(t) for t in fit["tokens"]],
            maxlen=MAX_LEN, padding="post", truncating="post"
        )
        X_hold = pad_sequences(
            [encode(t) for t in hold["tokens"]],
            maxlen=MAX_LEN, padding="post", truncating="post"
        )

        y_fit = fit["source_label"].to_numpy()

        # Reset model weights and optimizer for each fold.
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(SEED + fold)

        model = Sequential([
            Input(shape=(MAX_LEN,)),
            Embedding(
                input_dim=len(vocab),
                output_dim=64,
                mask_zero=True,
            ),
            SpatialDropout1D(0.20),
            LSTM(64, dropout=0.20),
            Dense(32, activation="relu"),
            Dropout(0.30),
            Dense(1, activation="sigmoid"),
        ])

        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        callback = EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
        )

        # A small subset of fold-training data is used for early stopping.
        # The held-out OOF fold is never used for fitting or early stopping.
        history = model.fit(
            X_fit,
            y_fit,
            validation_split=0.15,
            epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[callback],
            verbose=1,
            shuffle=True,
        )

        scores = model.predict(X_hold, verbose=0).ravel()
        oof_scores[hold_idx] = scores
        fold_ids[hold_idx] = fold

        fold_metadata.append({
            "fold": fold,
            "fit_samples": int(len(fit)),
            "held_out_samples": int(len(hold)),
            "vocabulary_size": int(len(vocab)),
            "epochs_trained": len(history.history["loss"]),
        })

        del model
        tf.keras.backend.clear_session()

    if np.isnan(oof_scores).any():
        raise RuntimeError("Some LSTM OOF predictions are missing.")

    result = pd.DataFrame({
        "sha256": train["sha256"],
        "sequence_hash": train["sequence_hash"],
        "label": y,
        "fold": fold_ids,
        "lstm_oof_score": oof_scores,
    })

    result.to_csv(out_dir / "train_oof_lstm.csv", index=False)

    metadata = {
        "training_unique_sequences": int(len(train)),
        "folds": N_SPLITS,
        "missing_lstm_scores": int(
            result["lstm_oof_score"].isna().sum()
        ),
        "fold_counts": {
            str(k): int(v)
            for k, v in result["fold"].value_counts().sort_index().items()
        },
        "fold_metadata": fold_metadata,
        "note": (
            "OOF predictions generated by fold-specific LSTMs. "
            "Vocabulary fitted independently within each fold."
        ),
    }

    with open(out_dir / "lstm_oof_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nOOF summary:")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
