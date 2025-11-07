"""Model creation helpers for Assignment 4."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import ModelConfig


@dataclass
class ModelComponents:
    """Container holding model and tokenizer."""

    model: AutoModelForSequenceClassification
    tokenizer: AutoTokenizer


def load_model_components(
    model_config: ModelConfig,
    num_labels: int,
    cache_dir: str | None = None,
) -> ModelComponents:
    """Instantiate tokenizer and sequence classification model."""

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.pretrained_model_name,
        cache_dir=cache_dir,
        use_fast=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config.pretrained_model_name,
        cache_dir=cache_dir,
        num_labels=num_labels,
        output_attentions=True,
    )
    return ModelComponents(model=model, tokenizer=tokenizer)
