"""Metric computation utilities."""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn import metrics as sk_metrics


def compute_metrics(prediction_output) -> Dict[str, float]:
    """Compute accuracy, precision, recall, F1 and ROC-AUC."""

    logits = prediction_output.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    labels = prediction_output.label_ids
    probabilities = softmax(logits)
    preds = probabilities.argmax(axis=-1)

    metric_values = {
        "accuracy": sk_metrics.accuracy_score(labels, preds),
        "precision": sk_metrics.precision_score(labels, preds, zero_division=0),
        "recall": sk_metrics.recall_score(labels, preds, zero_division=0),
        "f1": sk_metrics.f1_score(labels, preds, zero_division=0),
    }
    try:
        metric_values["roc_auc"] = sk_metrics.roc_auc_score(labels, probabilities[:, 1])
    except ValueError:
        metric_values["roc_auc"] = float("nan")
    return metric_values


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""

    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)
