"""
evaluate.py
-----------
Model evaluation: MAPE, RMSE, MAE, baseline comparison,
SHAP analysis, and comparative results table generation.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

logger = logging.getLogger(__name__)


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Percentage Error (handles zeros via +1 smoothing)."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs(actual - predicted) / (actual + 1)) * 100)


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(actual - predicted)))


def evaluate_model(
    actual: np.ndarray,
    predicted: np.ndarray,
    model_name: str = "Model",
) -> Dict[str, float]:
    """Compute all evaluation metrics for a single model."""
    mask = ~np.isnan(predicted) & ~np.isnan(actual)
    a, p = actual[mask], predicted[mask]
    return {
        "model": model_name,
        "mape": round(mape(a, p), 3),
        "rmse": round(rmse(a, p), 3),
        "mae": round(mae(a, p), 3),
        "n_samples": int(mask.sum()),
    }


def naive_baseline(series: np.ndarray, lag: int = 1) -> np.ndarray:
    """Naive lag-k baseline: predict today's value = value k steps ago."""
    preds = np.roll(series, lag).astype(float)
    preds[:lag] = np.nan
    return preds


def arima_baseline(series: pd.Series) -> np.ndarray:
    """Simple ARIMA(1,1,1) baseline using statsmodels."""
    try:
        from statsmodels.tsa.arima.model import ARIMA
        model = ARIMA(series.fillna(0), order=(1, 1, 1))
        res = model.fit()
        return res.fittedvalues.clip(0).values
    except Exception as e:
        logger.warning(f"ARIMA failed: {e}. Using naive baseline.")
        return naive_baseline(series.values)


def build_comparison_table(
    df: pd.DataFrame,
    district: str,
    model_predictions: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """
    Build a model comparison table with MAPE/RMSE/MAE for all models.

    Args:
        df: Full features dataframe.
        district: District name.
        model_predictions: Dict of {model_name: predictions_array}.

    Returns:
        Sorted comparison DataFrame.
    """
    district_df = df[df["district"] == district].sort_values("date")
    actual = district_df["cases"].values

    # Add baselines
    all_preds = {
        "Naive (lag-1)": naive_baseline(actual, lag=1),
        "ARIMA(1,1,1)": arima_baseline(district_df["cases"]),
        **model_predictions,
    }

    rows = [evaluate_model(actual, preds, name) for name, preds in all_preds.items()]
    results = pd.DataFrame(rows).sort_values("mape")
    results["mape_improvement_vs_arima"] = (
        (results.loc[results["model"] == "ARIMA(1,1,1)", "mape"].values[0]
         - results["mape"])
        / results.loc[results["model"] == "ARIMA(1,1,1)", "mape"].values[0] * 100
    ).round(1)
    return results


def plot_forecast(
    actual: pd.Series,
    ensemble_df: pd.DataFrame,
    district: str,
    save_path: Optional[str] = None,
) -> None:
    """Generate forecast vs actual plot with confidence intervals."""
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    dates = ensemble_df["date"]
    ax.fill_between(
        dates, ensemble_df["yhat_lower"], ensemble_df["yhat_upper"],
        alpha=0.25, color="#00c9ff", label="95% Interval"
    )
    ax.plot(dates, ensemble_df["yhat"], color="#00c9ff", lw=2, label="Ensemble Forecast")
    ax.plot(actual.index, actual.values, color="#ff6b6b", lw=1.5,
            alpha=0.9, label="Actual Cases")

    ax.set_title(f"Cholera Forecast — {district}", color="white", fontsize=14, pad=12)
    ax.set_xlabel("Date", color="#888")
    ax.set_ylabel("Weekly Cases", color="#888")
    ax.tick_params(colors="#888")
    ax.spines[:].set_color("#333")
    ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        logger.info(f"Forecast plot saved → {save_path}")
    plt.close()


def plot_shap_summary(shap_df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """Plot mean absolute SHAP values (feature importance bar chart)."""
    mean_shap = shap_df.abs().mean().sort_values(ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    bars = ax.barh(mean_shap.index, mean_shap.values, color="#00c9ff", alpha=0.85)
    ax.set_title("SHAP Feature Importance (Mean |SHAP|)", color="white", fontsize=12)
    ax.tick_params(colors="#aaa", labelsize=9)
    ax.spines[:].set_color("#333")
    ax.set_xlabel("Mean |SHAP value|", color="#888")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        logger.info(f"SHAP plot saved → {save_path}")
    plt.close()
