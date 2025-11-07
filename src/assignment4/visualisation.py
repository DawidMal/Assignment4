"""Visualisation helpers for training diagnostics and attention analysis."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import seaborn as sns
import torch

from .config import ModelConfig


plt.style.use("seaborn-v0_8")


def plot_training_history(
    history: dict,
    output_dir: Path,
    dataset_name: str,
    model_name: str,
) -> Path:
    """Plot training and evaluation loss curves."""

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    if history.get("step") and history.get("loss"):
        ax.plot(history["step"], history["loss"], label="train loss")
    if history.get("eval_step") and history.get("eval_loss"):
        ax.plot(history["eval_step"], history["eval_loss"], label="validation loss")
    ax.set_title(f"Loss curve: {dataset_name} - {model_name}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.legend()
    fig.tight_layout()
    path = output_dir / f"{dataset_name}_{model_name}_loss.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_attention_heatmap(
    model,
    tokenizer,
    texts: Iterable[str],
    output_dir: Path,
    dataset_name: str,
    model_config: ModelConfig,
    max_examples: int = 3,
) -> List[Path]:
    """Plot averaged attention weights for a few representative examples."""

    model.eval()
    device = next(model.parameters()).device
    saved_paths: List[Path] = []
    for idx, text in enumerate(texts):
        if idx >= max_examples:
            break
        encoding = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=model_config.max_length,
        )
        encoding = {k: v.to(device) for k, v in encoding.items()}
        with torch.no_grad():
            outputs = model(**encoding, output_attentions=True)
        attentions = outputs.attentions  # tuple of (num_layers, batch, num_heads, seq_len, seq_len)
        stacked = torch.stack(attentions)  # (layers, batch, heads, seq_len, seq_len)
        mean_attention = stacked.mean(dim=(0, 2)).squeeze(0).cpu().numpy()
        tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            mean_attention[: len(tokens), : len(tokens)],
            xticklabels=tokens,
            yticklabels=tokens,
            cmap="mako",
            ax=ax,
        )
        ax.set_title(
            f"Attention heatmap - {dataset_name} - {model_config.name} - sample {idx+1}"
        )
        ax.set_xlabel("Key tokens")
        ax.set_ylabel("Query tokens")
        fig.tight_layout()
        path = output_dir / f"{dataset_name}_{model_config.name}_attention_{idx+1}.png"
        fig.savefig(path)
        plt.close(fig)
        saved_paths.append(path)
    return saved_paths
