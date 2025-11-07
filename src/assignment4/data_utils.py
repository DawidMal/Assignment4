"""Utilities for loading and preparing the ransomware datasets."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset

from .config import AssignmentConfig, ModelConfig


class TokenisedDataset(Dataset):
    """Simple ``torch.utils.data.Dataset`` wrapper for tokenised samples."""

    def __init__(self, encodings: Dict[str, torch.Tensor], labels: Iterable[int]):
        self.encodings = encodings
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:  # pragma: no cover - trivial
        item = {key: tensor[idx] for key, tensor in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


def load_dataframe(csv_path: Path) -> pd.DataFrame:
    """Load a CSV file with columns ``text`` and ``label``."""

    df = pd.read_csv(csv_path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(
            f"Expected columns 'text' and 'label' in {csv_path}, got {df.columns.tolist()}"
        )
    return df


def stratified_split(
    df: pd.DataFrame, config: AssignmentConfig
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the dataframe into train/validation/test splits with stratification."""

    train_df, test_df = train_test_split(
        df,
        test_size=config.test_size,
        stratify=df["label"],
        random_state=config.random_seed,
    )
    train_df, val_df = train_test_split(
        train_df,
        test_size=config.validation_size,
        stratify=train_df["label"],
        random_state=config.random_seed,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def tokenise_texts(
    tokenizer,
    texts: Iterable[str],
    model_config: ModelConfig,
) -> Dict[str, torch.Tensor]:
    """Tokenise raw texts using the provided Hugging Face tokenizer."""

    return tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=model_config.max_length,
        return_tensors="pt",
    )


def build_datasets(
    tokenizer,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_config: ModelConfig,
) -> Tuple[TokenisedDataset, TokenisedDataset, TokenisedDataset]:
    """Create tokenised datasets for each split."""

    train_enc = tokenise_texts(tokenizer, train_df["text"].tolist(), model_config)
    val_enc = tokenise_texts(tokenizer, val_df["text"].tolist(), model_config)
    test_enc = tokenise_texts(tokenizer, test_df["text"].tolist(), model_config)

    train_ds = TokenisedDataset(train_enc, train_df["label"].tolist())
    val_ds = TokenisedDataset(val_enc, val_df["label"].tolist())
    test_ds = TokenisedDataset(test_enc, test_df["label"].tolist())
    return train_ds, val_ds, test_ds


def save_split_metadata(
    output_dir: Path,
    dataset_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: AssignmentConfig,
) -> None:
    """Persist summary statistics about the split for traceability."""

    meta = {
        "dataset": dataset_name,
        "num_train": len(train_df),
        "num_val": len(val_df),
        "num_test": len(test_df),
        "class_distribution_train": train_df["label"].value_counts(normalize=True).to_dict(),
        "class_distribution_val": val_df["label"].value_counts(normalize=True).to_dict(),
        "class_distribution_test": test_df["label"].value_counts(normalize=True).to_dict(),
        "config": asdict(config),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_file = output_dir / f"{dataset_name}_split_metadata.json"
    metadata_file.write_text(json.dumps(meta, indent=2))
