from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class ModelConfig:
    """Configuration for model training and evaluation."""

    task: Literal["classification", "regression"] = "classification"
    test_size: float = 0.2
    random_state: int = 42
    max_iter: int = 300
    n_estimators: int = 200


def build_preprocessor(
    numeric_cols: Iterable[str], categorical_cols: Iterable[str]
) -> ColumnTransformer:
    """Create preprocessing transformer for numeric and categorical features."""
    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())],
    )

    categorical_transformer = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ],
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, list(numeric_cols)),
            ("cat", categorical_transformer, list(categorical_cols)),
        ],
        remainder="drop",
    )
    return preprocessor


def build_classifier_model(
    numeric_cols: Iterable[str], categorical_cols: Iterable[str], config: ModelConfig
) -> Pipeline:
    """Assemble preprocessing + classifier pipeline."""
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    classifier = LogisticRegression(
        max_iter=config.max_iter,
        n_jobs=None,
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", classifier),
        ]
    )
    return model


def build_regressor_model(
    numeric_cols: Iterable[str], categorical_cols: Iterable[str], config: ModelConfig
) -> Pipeline:
    """Assemble preprocessing + regressor pipeline."""
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    regressor = RandomForestRegressor(
        n_estimators=config.n_estimators,
        random_state=config.random_state,
        n_jobs=None,
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", regressor),
        ]
    )
    return model


def train_and_evaluate(
    features: pd.DataFrame, target: pd.Series, config: ModelConfig | None = None
) -> tuple[Pipeline, dict[str, float]]:
    """Train the model pipeline and return metrics."""
    cfg = config or ModelConfig()
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=target if cfg.task == "classification" else None,
    )

    numeric_cols = [
        col for col in features.columns if pd.api.types.is_numeric_dtype(features[col])
    ]
    categorical_cols = [col for col in features.columns if col not in numeric_cols]

    if cfg.task == "classification":
        pipeline = build_classifier_model(numeric_cols, categorical_cols, cfg)
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        metrics = evaluate_classification_predictions(y_test, predictions)
    else:
        pipeline = build_regressor_model(numeric_cols, categorical_cols, cfg)
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        metrics = evaluate_regression_predictions(y_test, predictions)
    return pipeline, metrics


def evaluate_classification_predictions(
    y_true: pd.Series, y_pred: np.ndarray
) -> dict[str, float]:
    """Compute simple classification metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }


def evaluate_regression_predictions(
    y_true: pd.Series, y_pred: np.ndarray
) -> dict[str, float]:
    """Compute regression metrics."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }
