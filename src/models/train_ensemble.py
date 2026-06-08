"""
train_ensemble.py
-----------------
Main training script. Runs the full pipeline:
  1. Load processed features
  2. Train ensemble per district
  3. Evaluate and print comparison table
  4. Save models

Usage:
    python src/models/train_ensemble.py [--district Harare]
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

FEATURE_PATH = Path("data/processed/features.csv")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
Path("reports/figures").mkdir(exist_ok=True)


def load_features() -> pd.DataFrame:
    """Load feature-engineered dataset."""
    if not FEATURE_PATH.exists():
        logger.error(
            f"Features file not found at {FEATURE_PATH}. "
            "Run: python src/data/etl.py"
        )
        sys.exit(1)
    df = pd.read_csv(FEATURE_PATH, parse_dates=["date"])
    logger.info(f"Loaded features: {df.shape}")
    return df


def train_district(df: pd.DataFrame, district: str) -> dict:
    """Train ensemble for a single district and return results."""
    from src.models.ensemble import EnsembleForecaster
    from src.models.evaluate import build_comparison_table, plot_forecast

    logger.info(f"\n{'='*50}")
    logger.info(f"Training ensemble for: {district}")
    logger.info(f"{'='*50}")

    district_df = df[df["district"] == district]
    if len(district_df) < 50:
        logger.warning(f"Insufficient data for {district}. Skipping.")
        return {}

    # Split: last 20% for test
    split = int(len(district_df) * 0.8)
    train_df = df.copy()
    test_df = district_df.iloc[split:].copy()

    # Train
    ensemble = EnsembleForecaster()
    ensemble.fit(train_df, district)

    # Predict on test
    preds_df = ensemble.predict(df, district)
    preds_df = preds_df[preds_df["date"].isin(test_df["date"])]

    # Evaluate
    actual = test_df["cases"].values[:len(preds_df)]
    from src.models.evaluate import mape, rmse, mae
    results = {
        "district": district,
        "mape": round(mape(actual, preds_df["yhat"].values), 2),
        "rmse": round(rmse(actual, preds_df["yhat"].values), 2),
        "mae": round(mae(actual, preds_df["yhat"].values), 2),
        "n_test": len(actual),
    }
    logger.info(f"Results for {district}: MAPE={results['mape']}% RMSE={results['rmse']}")

    # Save model
    model_path = ensemble.save()
    results["model_path"] = model_path

    # Save forecast plot
    actual_series = pd.Series(
        district_df["cases"].values,
        index=pd.to_datetime(district_df["date"].values),
    )
    full_preds = ensemble.predict(df, district)
    plot_forecast(
        actual_series, full_preds, district,
        save_path=f"reports/figures/forecast_{district.lower()}.png",
    )

    return results


def main():
    parser = argparse.ArgumentParser(description="Train cholera ensemble models")
    parser.add_argument("--district", type=str, default=None,
                        help="Single district to train (default: all)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: fewer LSTM epochs for testing")
    args = parser.parse_args()

    df = load_features()
    districts = [args.district] if args.district else df["district"].unique().tolist()

    all_results = []
    for district in districts:
        try:
            result = train_district(df, district)
            if result:
                all_results.append(result)
        except Exception as e:
            logger.error(f"Failed to train {district}: {e}")
            continue

    # Summary table
    if all_results:
        summary = pd.DataFrame(all_results)
        print("\n" + "="*60)
        print("📊 Training Summary")
        print("="*60)
        print(summary.to_string(index=False))
        print(f"\nMean MAPE across districts: {summary['mape'].mean():.2f}%")
        summary.to_csv(REPORTS_DIR / "training_summary.csv", index=False)
        logger.info(f"Summary saved → {REPORTS_DIR / 'training_summary.csv'}")


if __name__ == "__main__":
    main()
