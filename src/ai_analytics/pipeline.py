from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data_loader import load_dataset, split_features_target
from .model import ModelConfig, train_and_evaluate


def run_pipeline(
    data_path: str | Path,
    target: str,
    test_size: float = 0.2,
    task: str = "classification",
) -> dict[str, float]:
    """Load data, train the model, and return evaluation metrics."""
    dataset = load_dataset(data_path)
    features, labels = split_features_target(dataset, target=target)

    config = ModelConfig(test_size=test_size, task=task)
    _, metrics = train_and_evaluate(features, labels, config=config)
    return metrics


def save_metrics(metrics: dict[str, float], output_path: str | Path) -> None:
    """Persist metrics as JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a simple analytics model on tabular data."
    )
    parser.add_argument(
        "--data",
        dest="data_path",
        default="data/sample/iris_small.csv",
        help="Path to CSV dataset.",
    )
    parser.add_argument(
        "--target",
        default="target",
        help="Name of target column in the dataset.",
    )
    parser.add_argument(
        "--task",
        choices=["classification", "regression"],
        default="classification",
        help="Model type to train.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data reserved for evaluation.",
    )
    parser.add_argument(
        "--save-metrics",
        dest="save_metrics_path",
        default=None,
        help="Optional path to write metrics JSON (e.g., artifacts/metrics.json).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    metrics = run_pipeline(
        args.data_path,
        args.target,
        test_size=args.test_size,
        task=args.task,
    )
    for name, value in metrics.items():
        print(f"{name}: {value:.3f}")

    if args.save_metrics_path:
        save_metrics(metrics, args.save_metrics_path)


if __name__ == "__main__":
    main()
