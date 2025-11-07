"""Training helpers using Hugging Face Trainer."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from transformers import Trainer, TrainingArguments, set_seed

from .config import AssignmentConfig, ModelConfig
from .metrics import compute_metrics


@dataclass
class TrainingResult:
    """Holds the trained trainer and evaluation artefacts."""

    trainer: Trainer
    eval_metrics: Dict[str, float]
    train_history: Dict[str, list]


def prepare_training_arguments(
    output_dir: Path,
    model_config: ModelConfig,
    dataset_name: str,
    assignment_config: AssignmentConfig,
) -> TrainingArguments:
    """Construct ``TrainingArguments`` with sensible defaults."""

    return TrainingArguments(
        output_dir=str(output_dir / f"{dataset_name}_{model_config.name}"),
        num_train_epochs=model_config.num_train_epochs,
        learning_rate=model_config.learning_rate,
        per_device_train_batch_size=model_config.per_device_train_batch_size,
        per_device_eval_batch_size=model_config.per_device_eval_batch_size,
        weight_decay=model_config.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        warmup_ratio=model_config.warmup_ratio,
        gradient_accumulation_steps=model_config.gradient_accumulation_steps,
        logging_strategy="steps",
        logging_steps=25,
        save_total_limit=2,
        seed=assignment_config.random_seed,
        dataloader_num_workers=assignment_config.num_workers,
        fp16=model_config.fp16,
    )


def train_model(
    model_components,
    train_dataset,
    val_dataset,
    test_dataset,
    dataset_name: str,
    model_config: ModelConfig,
    assignment_config: AssignmentConfig,
    output_dir: Path,
) -> TrainingResult:
    """Train a transformer model and evaluate it on the test set."""

    set_seed(assignment_config.random_seed)

    args = prepare_training_arguments(output_dir, model_config, dataset_name, assignment_config)
    trainer = Trainer(
        model=model_components.model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=model_components.tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_metrics = trainer.evaluate(test_dataset)

    history = trainer.state.log_history
    losses = {
        "step": [],
        "loss": [],
        "eval_step": [],
        "eval_loss": [],
    }
    for record in history:
        if "loss" in record:
            losses["step"].append(record.get("step", len(losses["step"])) )
            losses["loss"].append(record["loss"])
        if "eval_loss" in record:
            losses["eval_step"].append(record.get("step", len(losses["eval_step"])) )
            losses["eval_loss"].append(record["eval_loss"])

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = output_dir / f"{dataset_name}_{model_config.name}_metrics.json"
    metrics_file.write_text(json.dumps(eval_metrics, indent=2))
    history_file = output_dir / f"{dataset_name}_{model_config.name}_history.json"
    history_file.write_text(json.dumps(losses, indent=2))

    return TrainingResult(trainer=trainer, eval_metrics=eval_metrics, train_history=losses)
