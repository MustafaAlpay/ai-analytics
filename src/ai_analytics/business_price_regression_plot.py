from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

from ai_analytics.data_loader import load_dataset, split_features_target
from ai_analytics.model import (
    ModelConfig,
    build_regressor_model,
    evaluate_regression_predictions,
)


DATA_PATH = Path("data/sample/business_sales_demo.csv")
TARGET = "unit_price"
TEST_SIZE = 0.2
OUTPUT_FIG = Path("artifacts/price_regression.png")


def main() -> None:
    """Train a regression model on unit_price and save a prediction plot."""
    data = load_dataset(DATA_PATH)
    features, labels = split_features_target(data, target=TARGET)

    cfg = ModelConfig(task="regression", test_size=TEST_SIZE)
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=None,
    )

    numeric_cols = [
        col for col in features.columns if pd.api.types.is_numeric_dtype(features[col])
    ]
    categorical_cols = [col for col in features.columns if col not in numeric_cols]

    model = build_regressor_model(numeric_cols, categorical_cols, cfg)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = evaluate_regression_predictions(y_test, preds)

    print(f"Data: {DATA_PATH}")
    print(f"Target: {TARGET}")
    for name, value in metrics.items():
        print(f"{name}: {value:.3f}")

    OUTPUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    ax.scatter(y_test, preds, alpha=0.75, edgecolor="white", linewidth=0.5)
    min_val = min(y_test.min(), preds.min())
    max_val = max(y_test.max(), preds.max())
    ax.plot([min_val, max_val], [min_val, max_val], color="#4c78a8", linestyle="--")
    ax.set_xlabel("Actual price")
    ax.set_ylabel("Predicted price")
    ax.set_title("Price regression: actual vs predicted")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUTPUT_FIG, dpi=200)
    print(f"Saved plot to {OUTPUT_FIG}")


if __name__ == "__main__":
    main()
