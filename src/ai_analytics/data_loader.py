from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a CSV dataset into a DataFrame with basic validation."""
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    df = pd.read_csv(dataset_path)
    if df.empty:
        raise ValueError("Dataset is empty; provide at least one row.")

    return df


def split_features_target(
    df: pd.DataFrame, target: str
) -> tuple[pd.DataFrame, pd.Series]:
    """Split DataFrame into feature matrix and target vector."""
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found. Available: {list(df.columns)}")

    features = df.drop(columns=[target])
    y = df[target]
    if features.empty:
        raise ValueError("No feature columns remain after removing target.")

    return features, y


def detect_feature_types(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return numeric and categorical columns for preprocessing."""
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    for col in features.columns:
        series = features[col]
        if pd.api.types.is_numeric_dtype(series):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    return numeric_cols, categorical_cols
