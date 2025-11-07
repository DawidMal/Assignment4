"""Configuration dataclasses for Assignment 4 models and training."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ModelConfig:
    """Hyper-parameters and identifiers for a single transformer model."""

    name: str
    pretrained_model_name: str
    max_length: int = 256
    learning_rate: float = 2e-5
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 1
    fp16: bool = True


@dataclass
class AssignmentConfig:
    """Top level configuration shared across datasets and models."""

    random_seed: int = 42
    test_size: float = 0.2
    validation_size: float = 0.1
    metrics: List[str] = field(
        default_factory=lambda: [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
        ]
    )
    num_workers: int = 2
    shap_sample_size: int = 20
    lime_sample_size: int = 5
    attention_plot_examples: int = 3
    output_dir: str = "artifacts"
    model_cache_dir: str | None = None
    label_names: Dict[int, str] = field(
        default_factory=lambda: {0: "benign", 1: "ransomware"}
    )


DEFAULT_MODELS: List[ModelConfig] = [
    ModelConfig(name="bert", pretrained_model_name="bert-base-uncased"),
    ModelConfig(name="roberta", pretrained_model_name="roberta-base"),
    ModelConfig(name="deberta", pretrained_model_name="microsoft/deberta-base"),
]
