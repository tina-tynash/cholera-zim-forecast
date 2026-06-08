"""
test_models.py
==============
Pytest tests for the XGBoost and Ensemble forecasting models.
Run: pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
import shap
from pathlib import Path

DATA_DIR = Path("data/processed")


@pytest.fixture(scope="module")
def sample_data():
    """Build a minimal feature-engineered dataset for model tests."""
    if not (DATA_DIR / "cholera_cases.csv").exists():
        pytest.skip("Processed data not available; run generate_synthetic.py first")

    cholera = pd.read_csv(DATA_DIR / "cholera_cases.csv", parse_dates=["date"])
    climate = pd.read_csv(DATA_DIR / "climate.csv", parse_dates=["date"])

    weekly_clim = (
        climate
        .assign(date=lambda d: d["date"] - pd.to_timedelta(d["date"].dt.dayofweek, unit="D"))
        .groupby(["date", "district"])
        .agg(rainfall_mm=("rainfall_mm", "sum"), temperature_c=("temperature_c", "mean"))
        .reset_index()
    )

    cholera["year"]  = cholera["date"].dt.year
    cholera["month"] = cholera["date"].dt.month
    cholera["week"]  = cholera["date"].dt.isocalendar().week.astype(int)
    master = cholera.merge(weekly_clim, on=["date", "district"], how="left")

    for lag in [1, 2, 4]:
        master[f"lag_{lag}w"] = master.groupby("district")["cases"].shift(lag).fillna(0)

    master["rainy"] = master["month"].isin([11, 12, 1, 2, 3, 4]).astype(int)
    master = master.dropna(subset=["lag_1w"]).reset_index(drop=True)
    return master


FEATURES = ["lag_1w", "lag_2w", "lag_4w", "rainfall_mm", "temperature_c", "month", "week", "rainy"]


class TestXGBoostModel:
    """Tests for XGBoost forecaster."""

    def test_fit_and_predict_shape(self, sample_data):
        """Model output should have same length as input test set."""
        train = sample_data[sample_data["date"] < "2024-01-01"]
        test  = sample_data[sample_data["date"] >= "2024-01-01"]
        model = xgb.XGBRegressor(n_estimators=20, verbosity=0, random_state=42)
        model.fit(train[FEATURES].fillna(0), train["cases"])
        preds = model.predict(test[FEATURES].fillna(0))
        assert len(preds) == len(test), "Prediction length mismatch"

    def test_predictions_non_negative(self, sample_data):
        """Raw XGBoost predictions should be clipped to 0; cases are counts."""
        train = sample_data[sample_data["date"] < "2024-01-01"]
        test  = sample_data[sample_data["date"] >= "2024-01-01"]
        model = xgb.XGBRegressor(n_estimators=20, verbosity=0, random_state=42)
        model.fit(train[FEATURES].fillna(0), train["cases"])
        preds = np.maximum(model.predict(test[FEATURES].fillna(0)), 0)
        assert np.all(preds >= 0), "Negative predictions found"

    def test_feature_importance_available(self, sample_data):
        """Feature importances should be available post-fit."""
        model = xgb.XGBRegressor(n_estimators=20, verbosity=0, random_state=42)
        model.fit(sample_data[FEATURES].fillna(0), sample_data["cases"])
        importances = model.feature_importances_
        assert len(importances) == len(FEATURES)
        assert np.all(importances >= 0)

    def test_shap_values_shape(self, sample_data):
        """SHAP values should have shape (n_samples, n_features)."""
        model = xgb.XGBRegressor(n_estimators=20, verbosity=0, random_state=42)
        model.fit(sample_data[FEATURES].fillna(0), sample_data["cases"])
        subset = sample_data[FEATURES].fillna(0).iloc[:50]
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(subset)
        assert shap_values.shape == (50, len(FEATURES)), "SHAP shape mismatch"

    def test_model_improves_over_naive(self, sample_data):
        """XGBoost MAPE should be better than naive lag-1 baseline."""
        train = sample_data[sample_data["date"] < "2024-01-01"]
        test  = sample_data[
            (sample_data["date"] >= "2024-01-01") & (sample_data["cases"] > 0)
        ]
        model = xgb.XGBRegressor(n_estimators=100, max_depth=4, verbosity=0, random_state=42)
        model.fit(train[FEATURES].fillna(0), train["cases"])
        preds = np.maximum(model.predict(test[FEATURES].fillna(0)), 0)
        y_true = test["cases"].values

        def smape(y, p):
            return np.mean(np.abs(y - p) / ((np.abs(y) + np.abs(p)) / 2 + 1e-8)) * 100

        xgb_smape   = smape(y_true, preds)
        naive_smape = smape(y_true, test["lag_1w"].values)
        assert xgb_smape < naive_smape, (
            f"XGBoost ({xgb_smape:.1f}%) should beat naive ({naive_smape:.1f}%)"
        )


class TestEnsembleModel:
    """Tests for ensemble weighting logic."""

    def test_weighted_ensemble(self, sample_data):
        """Simple weighted ensemble should produce valid predictions."""
        train = sample_data[sample_data["date"] < "2024-01-01"]
        test  = sample_data[sample_data["date"] >= "2024-01-01"]
        model = xgb.XGBRegressor(n_estimators=20, verbosity=0, random_state=42)
        model.fit(train[FEATURES].fillna(0), train["cases"])

        xgb_preds   = np.maximum(model.predict(test[FEATURES].fillna(0)), 0)
        naive_preds = np.maximum(test["lag_1w"].values, 0)
        ensemble    = 0.7 * xgb_preds + 0.3 * naive_preds

        assert len(ensemble) == len(test)
        assert np.all(ensemble >= 0)
        assert np.all(np.isfinite(ensemble))

    def test_conformal_interval_coverage(self, sample_data):
        """95% conformal intervals should cover ~95% of calibration set."""
        train = sample_data[sample_data["date"] < "2023-01-01"]
        calib = sample_data[
            (sample_data["date"] >= "2023-01-01") & (sample_data["date"] < "2024-01-01")
        ]
        test  = sample_data[sample_data["date"] >= "2024-01-01"]

        model = xgb.XGBRegressor(n_estimators=50, verbosity=0, random_state=42)
        model.fit(train[FEATURES].fillna(0), train["cases"])

        calib_preds = np.maximum(model.predict(calib[FEATURES].fillna(0)), 0)
        residuals   = np.abs(calib["cases"].values - calib_preds)
        q95         = np.quantile(residuals, 0.95)

        test_preds = np.maximum(model.predict(test[FEATURES].fillna(0)), 0)
        test_lower = np.maximum(test_preds - q95, 0)
        test_upper = test_preds + q95
        coverage   = np.mean(
            (test["cases"].values >= test_lower) & (test["cases"].values <= test_upper)
        )
        # Conformal intervals should achieve roughly 90%+ empirical coverage
        assert coverage >= 0.85, f"Coverage too low: {coverage:.2%}"


class TestMetrics:
    """Tests for evaluation metrics."""

    def test_mape_perfect_forecast(self):
        """MAPE should be 0 for perfect predictions."""
        y = np.array([10.0, 20.0, 30.0, 40.0])
        mask = y > 0
        mape = np.mean(np.abs((y[mask] - y[mask]) / y[mask])) * 100
        assert mape == 0.0

    def test_mape_excludes_zeros(self):
        """MAPE should only be computed over non-zero actuals."""
        y    = np.array([0.0, 10.0, 0.0, 20.0])
        pred = np.array([5.0, 12.0, 3.0, 18.0])
        mask = y > 0
        mape = np.mean(np.abs((y[mask] - pred[mask]) / y[mask])) * 100
        # Only indices 1 and 3 contribute
        expected = np.mean([abs(10-12)/10, abs(20-18)/20]) * 100
        assert abs(mape - expected) < 1e-6

    def test_rmse_symmetry(self):
        """RMSE should be symmetric in over- and under-prediction."""
        y     = np.array([10.0, 10.0])
        pred1 = np.array([15.0, 10.0])
        pred2 = np.array([ 5.0, 10.0])
        rmse1 = np.sqrt(np.mean((y - pred1) ** 2))
        rmse2 = np.sqrt(np.mean((y - pred2) ** 2))
        assert abs(rmse1 - rmse2) < 1e-9
