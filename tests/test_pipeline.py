from pathlib import Path

from ai_analytics.data_loader import load_dataset, split_features_target
from ai_analytics.model import ModelConfig, train_and_evaluate
from ai_analytics.pipeline import run_pipeline, save_metrics

SAMPLE_DATA = Path("data/sample/iris_small.csv")


def test_sample_dataset_loads() -> None:
    df = load_dataset(SAMPLE_DATA)
    assert not df.empty
    assert "target" in df.columns


def test_split_features_target() -> None:
    df = load_dataset(SAMPLE_DATA)
    features, target = split_features_target(df, target="target")
    assert target.name == "target"
    assert features.shape[0] == target.shape[0]
    assert "target" not in features.columns


def test_train_and_evaluate_produces_metrics() -> None:
    df = load_dataset(SAMPLE_DATA)
    features, labels = split_features_target(df, target="target")

    _, metrics = train_and_evaluate(features, labels, config=ModelConfig(test_size=0.4))
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1_macro"] <= 1.0


def test_cli_pipeline_runs_end_to_end(tmp_path: Path) -> None:
    metrics = run_pipeline(SAMPLE_DATA, target="target", test_size=0.3)
    assert set(metrics.keys()) == {"accuracy", "f1_macro"}

    metrics_path = tmp_path / "metrics.json"
    save_metrics(metrics, metrics_path)
    assert metrics_path.exists()
