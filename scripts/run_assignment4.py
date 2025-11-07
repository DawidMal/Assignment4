"""Entry point for Assignment 4 coding tasks.

This script orchestrates the full workflow:
1. Load the UGR and PM ransomware datasets.
2. Split them into train/validation/test subsets with stratification.
3. Fine-tune BERT, RoBERTa and DeBERTa models.
4. Evaluate each model and persist metrics, loss curves and attention plots.
5. Generate SHAP and LIME explanations for qualitative analysis.

The script assumes the following Python dependencies are available:
    - torch
    - transformers
    - pandas
    - scikit-learn
    - matplotlib & seaborn
    - shap
    - lime

Due to the size of the underlying transformer checkpoints, the first run can
be slow while models are downloaded. Subsequent executions will use the cached
artifacts.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List

# Ensure the ``src`` directory is on the Python path so the ``assignment4``
# package can be imported without installation.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from assignment4.config import AssignmentConfig, DEFAULT_MODELS, ModelConfig
from assignment4.data_utils import (
    build_datasets,
    load_dataframe,
    save_split_metadata,
    stratified_split,
)
from assignment4.explainability import generate_lime_reports, generate_shap_plots
from assignment4.modeling import load_model_components
from assignment4.training import train_model
from assignment4.visualisation import plot_attention_heatmap, plot_training_history


DATASETS: Dict[str, Path] = {
    "ugr": Path("UGR_text.csv"),
    "pm": Path("PM_text.csv"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Assignment 4 pipeline")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory where models, metrics and plots will be written.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=[cfg.name for cfg in DEFAULT_MODELS],
        help="Subset of models to run (bert, roberta, deberta).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=AssignmentConfig().random_seed,
        help="Random seed used for splits and training.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit on number of samples per dataset for quick experiments.",
    )
    return parser.parse_args()


def filter_models(selected: Iterable[str]) -> List[ModelConfig]:
    selected = {name.lower() for name in selected}
    configs = [cfg for cfg in DEFAULT_MODELS if cfg.name in selected]
    if not configs:
        raise ValueError(f"No valid models selected from: {selected}")
    return configs


def sample_texts(texts: List[str], k: int, seed: int) -> List[str]:
    if k is None or k <= 0:
        return texts
    rng = random.Random(seed)
    if len(texts) <= k:
        return texts
    return rng.sample(texts, k)


def run_pipeline(args: argparse.Namespace) -> None:
    assignment_cfg = AssignmentConfig(random_seed=args.seed, output_dir=str(args.output_dir))
    models_to_run = filter_models(args.models)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Dict[str, float]] = {}

    for dataset_name, csv_path in DATASETS.items():
        df = load_dataframe(csv_path)
        if args.max_samples:
            df = df.sample(n=min(args.max_samples, len(df)), random_state=args.seed)
        train_df, val_df, test_df = stratified_split(df, assignment_cfg)
        save_split_metadata(args.output_dir, dataset_name, train_df, val_df, test_df, assignment_cfg)

        for model_cfg in models_to_run:
            components = load_model_components(
                model_cfg,
                num_labels=len(assignment_cfg.label_names),
                cache_dir=assignment_cfg.model_cache_dir,
            )
            train_ds, val_ds, test_ds = build_datasets(
                components.tokenizer,
                train_df,
                val_df,
                test_df,
                model_cfg,
            )
            training_result = train_model(
                components,
                train_ds,
                val_ds,
                test_ds,
                dataset_name,
                model_cfg,
                assignment_cfg,
                args.output_dir,
            )
            summary_key = f"{dataset_name}_{model_cfg.name}"
            summary[summary_key] = training_result.eval_metrics

            plot_training_history(
                training_result.train_history,
                args.output_dir,
                dataset_name,
                model_cfg.name,
            )

            attention_examples = sample_texts(
                test_df["text"].tolist(),
                assignment_cfg.attention_plot_examples,
                seed=args.seed,
            )
            plot_attention_heatmap(
                training_result.trainer.model,
                components.tokenizer,
                attention_examples,
                args.output_dir,
                dataset_name,
                model_cfg,
                max_examples=assignment_cfg.attention_plot_examples,
            )

            shap_texts = sample_texts(
                test_df["text"].tolist(),
                assignment_cfg.shap_sample_size,
                seed=args.seed,
            )
            generate_shap_plots(
                training_result.trainer.model,
                components.tokenizer,
                shap_texts,
                args.output_dir,
                dataset_name,
                model_cfg,
                [assignment_cfg.label_names[i] for i in sorted(assignment_cfg.label_names)],
            )

            lime_texts = sample_texts(
                test_df["text"].tolist(),
                assignment_cfg.lime_sample_size,
                seed=args.seed,
            )
            generate_lime_reports(
                training_result.trainer.model,
                components.tokenizer,
                lime_texts,
                args.output_dir,
                dataset_name,
                model_cfg,
                [assignment_cfg.label_names[i] for i in sorted(assignment_cfg.label_names)],
            )

    summary_file = args.output_dir / "summary_metrics.json"
    summary_file.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run_pipeline(parse_args())
