"""Explainability utilities using SHAP and LIME."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import shap
import torch
from lime.lime_text import LimeTextExplainer

from .config import ModelConfig


def _prepare_prediction_function(model, tokenizer, model_config: ModelConfig):
    device = next(model.parameters()).device

    def predict(texts: Iterable[str]) -> np.ndarray:
        encoded = tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=model_config.max_length,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return probabilities.detach().cpu().numpy()

    return predict


def _normalise_shap_explanation(explanation: shap.Explanation) -> shap.Explanation:
    """Reduce multi-class explanations to a single binary class for plotting."""

    if explanation.values.ndim == 3:
        # Choose the positive class (index 1) when available.
        class_index = 1 if explanation.values.shape[-1] > 1 else 0
        values = explanation.values[..., class_index]
        base_values = explanation.base_values[..., class_index]
        data = explanation.data
        feature_names = explanation.feature_names
        return shap.Explanation(
            values=values,
            base_values=base_values,
            data=data,
            feature_names=feature_names,
        )
    return explanation


def generate_shap_plots(
    model,
    tokenizer,
    texts: List[str],
    output_dir: Path,
    dataset_name: str,
    model_config: ModelConfig,
    label_names: List[str],
) -> Path:
    """Generate SHAP explanation bar plot for a subset of texts."""

    if not texts:
        raise ValueError("No texts provided for SHAP analysis")

    predict_fn = _prepare_prediction_function(model, tokenizer, model_config)
    try:
        masker = shap.maskers.Text(tokenizer)
        explainer = shap.Explainer(predict_fn, masker)
    except AttributeError:
        # Fallback for older SHAP versions without text masker support.
        background = texts[: min(5, len(texts))]
        explainer = shap.KernelExplainer(predict_fn, background)

    explanation = explainer(texts)
    if isinstance(explanation, list):
        explanation = explanation[0]
    explanation = _normalise_shap_explanation(explanation)

    plt.figure(figsize=(10, 6))
    shap.plots.bar(explanation, max_display=20, show=False)
    plt.title(f"SHAP feature importance: {dataset_name} - {model_config.name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{dataset_name}_{model_config.name}_shap.png"
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    raw_values = {
        "texts": texts,
        "shap_values": np.asarray(explanation.values).tolist(),
        "base_values": np.asarray(explanation.base_values).tolist(),
        "label_names": label_names,
    }
    (output_dir / f"{dataset_name}_{model_config.name}_shap.json").write_text(
        json.dumps(raw_values, indent=2)
    )
    return path


def generate_lime_reports(
    model,
    tokenizer,
    texts: List[str],
    output_dir: Path,
    dataset_name: str,
    model_config: ModelConfig,
    label_names: List[str],
) -> List[Path]:
    """Generate LIME explanations and persist them as HTML files."""

    if not texts:
        raise ValueError("No texts provided for LIME analysis")

    predict_fn = _prepare_prediction_function(model, tokenizer, model_config)
    explainer = LimeTextExplainer(class_names=label_names)

    output_dir.mkdir(parents=True, exist_ok=True)
    report_paths: List[Path] = []
    for idx, text in enumerate(texts, start=1):
        explanation = explainer.explain_instance(text, predict_fn, num_features=20)
        path = output_dir / f"{dataset_name}_{model_config.name}_lime_{idx}.html"
        explanation.save_to_file(str(path))
        report_paths.append(path)
    return report_paths
