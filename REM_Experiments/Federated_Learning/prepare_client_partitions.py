import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
CLIENTS = 3

DATA_PATH = Path(
    "REM_Datasets/04_Unified_Dataset/processed/public_labeled_candidate.csv"
)
MANIFEST_PATH = Path(
    "REM_Models/API_Sequence_Grouped/grouped_split_manifest.csv"
)
OUT_DIR = Path("REM_Experiments/Federated_Learning")
API_COLS = [str(i) for i in range(153)]


def make_sequence(row):
    ignored = {"", "nan", "none", "null"}
    return [
        token for value in row
        if (token := str(value).strip()).lower() not in ignored
    ]


def main():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    manifest = pd.read_csv(MANIFEST_PATH)

    required = {
        "sha256", "source_label", "eligible_for_supervised_training",
        "hash_label_conflict", *API_COLS
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    df["sha256"] = df["sha256"].astype(str).str.strip().str.lower()
    manifest["sha256"] = (
        manifest["sha256"].astype(str).str.strip().str.lower()
    )

    df["api_tokens"] = df[API_COLS].apply(make_sequence, axis=1)
    df["sequence_hash"] = df["api_tokens"].map(
        lambda tokens: hashlib.sha256(
            " ".join(tokens).encode("utf-8")
        ).hexdigest()
    )

    label_counts = df.groupby("sequence_hash")["source_label"].nunique()
    conflicts = set(label_counts[label_counts > 1].index)
    df = df[~df["sequence_hash"].isin(conflicts)].copy()

    df = df.merge(
        manifest[["sha256", "split"]],
        on="sha256",
        how="inner",
        validate="one_to_one",
    )

    df = df[
        (df["split"] == "train")
        & (df["eligible_for_supervised_training"] == True)
        & (df["hash_label_conflict"] == False)
    ].copy()

    df = df.drop_duplicates("sequence_hash").reset_index(drop=True)

    if df["sequence_hash"].duplicated().any():
        raise ValueError("Duplicate sequence hashes remain.")
    if not set(df["source_label"].astype(int).unique()).issubset({0, 1}):
        raise ValueError("Unexpected class labels.")
    if len(df) == 0 or df["source_label"].nunique() != 2:
        raise ValueError("Training data is empty or missing a class.")

    # Deterministic, label-stratified client partitioning.
    # Grouping by sequence_hash prevents identical sequences being split.
    rng = np.random.default_rng(SEED)
    assignments = {}

    for label, group in df.groupby("source_label"):
        hashes = group["sequence_hash"].drop_duplicates().to_numpy().copy()
        rng.shuffle(hashes)
        for index, seq_hash in enumerate(hashes):
            client_id = index % CLIENTS
            assignments[seq_hash] = client_id

    df["client_id"] = df["sequence_hash"].map(assignments)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audit_rows = []
    for client_id in range(CLIENTS):
        client = df[df["client_id"] == client_id].copy()
        audit_rows.append({
            "client_id": client_id,
            "samples": len(client),
            "benign": int((client["source_label"] == 0).sum()),
            "malicious": int((client["source_label"] == 1).sum()),
            "unique_sequences": client["sequence_hash"].nunique(),
        })

    audit = pd.DataFrame(audit_rows)
    audit.to_csv(OUT_DIR / "client_partition_audit.csv", index=False)

    # Save metadata only; client datasets are created after audit approval.
    metadata = {
        "seed": SEED,
        "clients": CLIENTS,
        "partition_strategy": "deterministic label-stratified sequence groups",
        "training_samples": len(df),
        "unique_sequence_hashes": int(df["sequence_hash"].nunique()),
        "validation_test_included": False,
        "model_training_performed": False,
    }
    (OUT_DIR / "partition_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("Partition audit (no model training performed):")
    print(audit.to_string(index=False))
    print("\nTotal samples:", int(audit["samples"].sum()))
    print("Unique sequences across clients:", int(df["sequence_hash"].nunique()))
    print("Sequence hashes assigned to multiple clients:",
          int(df.groupby("sequence_hash")["client_id"].nunique().gt(1).sum()))
    print("Output directory:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()
