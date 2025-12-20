from __future__ import annotations

from pathlib import Path

from ai_analytics.pipeline import run_pipeline


DATA_PATH = Path("data/sample/business_sales_demo.csv")
TARGET = "unit_price"
TEST_SIZE = 0.2


def main() -> None:
    """Run regression on the business sales demo data."""
    metrics = run_pipeline(
        DATA_PATH,
        target=TARGET,
        test_size=TEST_SIZE,
        task="regression",
    )
    print(f"Data: {DATA_PATH}")
    print(f"Target: {TARGET}")
    for name, value in metrics.items():
        print(f"{name}: {value:.3f}")


if __name__ == "__main__":
    main()
