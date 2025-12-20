from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from ai_analytics.pipeline import run_pipeline


DEFAULT_DATA_PATH = "data/sample/business_sales_demo.csv"
DEFAULT_TARGET = "price_bucket"


@st.cache_data(show_spinner=False)
def load_data(path: str | Path) -> pd.DataFrame:
    """Read a CSV into a DataFrame."""
    return pd.read_csv(path)


def render_preview(data: pd.DataFrame, target: str) -> None:
    st.subheader("Data preview")
    st.dataframe(data.head())

    if target in data.columns:
        st.subheader("Target distribution")
        if pd.api.types.is_numeric_dtype(data[target]):
            st.bar_chart(data[target])
        else:
            st.bar_chart(data[target].value_counts())

    numeric_cols = [
        col for col in data.select_dtypes(include="number").columns if col != target
    ]
    if not numeric_cols:
        return

    st.subheader("Feature histograms")
    for col in numeric_cols:
        fig, ax = plt.subplots()
        ax.hist(data[col].dropna(), bins=15, color="#4c78a8", edgecolor="white")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.grid(True, linestyle="--", alpha=0.5)
        st.pyplot(fig)


def main() -> None:
    st.set_page_config(page_title="Tabular Model Metrics", page_icon="📊")
    st.title("Tabular Model Metrics")
    st.caption("Point to a CSV, set the target, choose task, and evaluate the model.")

    data_path = st.text_input("Data CSV", DEFAULT_DATA_PATH)
    target = st.text_input("Target column", DEFAULT_TARGET)
    task = st.selectbox("Task", ["regression", "classification"], index=0)
    test_size = st.slider("Test size", 0.1, 0.5, 0.2, 0.05)

    data = None
    if data_path:
        try:
            data = load_data(data_path)
        except FileNotFoundError:
            st.warning(f"Could not find file at {data_path}.")
        except Exception as exc:  # pylint: disable=broad-except
            st.error(f"Failed to load data: {exc}")

    if data is not None:
        render_preview(data, target)

    if st.button("Run pipeline", type="primary"):
        with st.spinner("Training and evaluating model..."):
            try:
                metrics = run_pipeline(
                    data_path, target=target, test_size=test_size, task=task
                )
            except Exception as exc:  # pylint: disable=broad-except
                st.error(f"Pipeline failed: {exc}")
            else:
                st.success("Done")
                st.json(metrics)


if __name__ == "__main__":
    main()
