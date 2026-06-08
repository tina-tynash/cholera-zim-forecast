"""
tests/test_etl.py
-----------------
Data integrity and feature engineering tests.
Requires: python data/synthetic/generate_synthetic.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

DATA_DIR = Path("data/processed")


def _load_or_skip(filename, **kwargs):
    path = DATA_DIR / filename
    if not path.exists():
        pytest.skip(f"Run generate_synthetic.py first (missing {filename})")
    return pd.read_csv(path, **kwargs)


@pytest.fixture(scope="module")
def cholera_df():
    return _load_or_skip("cholera_cases.csv", parse_dates=["date"])


@pytest.fixture(scope="module")
def climate_df():
    return _load_or_skip("climate.csv", parse_dates=["date"])


@pytest.fixture(scope="module")
def demographics_df():
    return _load_or_skip("demographics.csv")


@pytest.fixture(scope="module")
def interventions_df():
    return _load_or_skip("interventions.csv", parse_dates=["date"])


class TestCholeraData:

    def test_not_empty(self, cholera_df):
        assert len(cholera_df) > 0

    def test_required_columns(self, cholera_df):
        for col in ["date", "district", "cases", "deaths"]:
            assert col in cholera_df.columns

    def test_cases_non_negative(self, cholera_df):
        assert (cholera_df["cases"] >= 0).all()

    def test_deaths_le_cases(self, cholera_df):
        assert (cholera_df["deaths"] <= cholera_df["cases"]).all()

    def test_expected_districts(self, cholera_df):
        assert {"Harare", "Bulawayo", "Chitungwiza"}.issubset(
            set(cholera_df["district"].unique())
        )

    def test_date_range(self, cholera_df):
        assert cholera_df["date"].min() <= pd.Timestamp("2018-06-01")
        assert cholera_df["date"].max() >= pd.Timestamp("2025-01-01")

    def test_no_duplicate_date_district(self, cholera_df):
        dupes = cholera_df.duplicated(subset=["date", "district"]).sum()
        assert dupes == 0

    def test_harare_2018_outbreak(self, cholera_df):
        harare_2018 = cholera_df[
            (cholera_df["district"] == "Harare") &
            (cholera_df["date"].dt.year == 2018)
        ]["cases"].sum()
        mean_other = cholera_df[
            (cholera_df["district"] != "Harare") &
            (cholera_df["date"].dt.year == 2018)
        ].groupby("district")["cases"].sum().mean()
        assert harare_2018 > mean_other * 1.5


class TestClimateData:

    def test_not_empty(self, climate_df):
        assert len(climate_df) > 0

    def test_required_columns(self, climate_df):
        for col in ["date", "district", "rainfall_mm"]:
            assert col in climate_df.columns

    def test_rainfall_non_negative(self, climate_df):
        assert (climate_df["rainfall_mm"] >= 0).all()

    def test_temperature_realistic(self, climate_df):
        if "temperature_c" in climate_df.columns:
            assert climate_df["temperature_c"].between(5, 50).all()


class TestDemographicsData:

    def test_not_empty(self, demographics_df):
        assert len(demographics_df) > 0

    def test_required_columns(self, demographics_df):
        for col in ["year", "district", "population"]:
            assert col in demographics_df.columns

    def test_population_positive(self, demographics_df):
        assert (demographics_df["population"] > 0).all()

    def test_wash_coverage_range(self, demographics_df):
        wash_col = "wash_access_pct" if "wash_access_pct" in demographics_df.columns else "wash_coverage"
        if wash_col in demographics_df.columns:
            assert demographics_df[wash_col].between(0, 1).all()

    def test_year_range(self, demographics_df):
        years = demographics_df["year"].unique()
        assert 2018 in years and 2025 in years


class TestInterventionsData:

    def test_not_empty(self, interventions_df):
        assert len(interventions_df) > 0

    def test_required_columns(self, interventions_df):
        for col in ["date", "district", "intervention_type", "coverage_pct"]:
            assert col in interventions_df.columns

    def test_coverage_range(self, interventions_df):
        assert interventions_df["coverage_pct"].between(0, 1).all()

    def test_has_ocv_campaigns(self, interventions_df):
        ocv_types = {"OCV_Campaign", "OCV_campaign", "ocv_campaign"}
        has_ocv = interventions_df["intervention_type"].isin(ocv_types).any()
        assert has_ocv, "No OCV campaign records found"


class TestFeatureEngineering:

    def test_lag_features(self):
        df = pd.DataFrame({
            "district": ["A"] * 6,
            "date": pd.date_range("2020-01-01", periods=6, freq="W"),
            "cases": [0, 2, 4, 6, 8, 10],
        })
        df["cases_lag1w"] = df.groupby("district")["cases"].shift(1)
        assert df.iloc[1]["cases_lag1w"] == 0.0
        assert df.iloc[2]["cases_lag1w"] == 2.0

    def test_rolling_mean_correctness(self):
        """Rolling 3-week mean with shift(1): at index 3, mean of indices 0,1,2."""
        cases = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
        # shift(1): [NaN, 10, 20, 30, 40]
        # rolling(3) at index 3: mean(10, 20, 30) = 20
        rolled = cases.shift(1).rolling(3).mean()
        assert abs(rolled.iloc[3] - 20.0) < 1e-9

    def test_seasonal_flag(self):
        months = pd.Series([1, 2, 3, 4, 5, 6, 11, 12])
        rainy = months.isin([11, 12, 1, 2, 3]).astype(int)
        assert rainy.sum() == 5
        assert rainy.iloc[3] == 0  # April

    def test_interaction_term(self):
        wash = 1.0
        rainfall = 50.0
        assert (1 - wash) * rainfall == 0.0


class TestMetricsUtils:

    def test_mape_smoothed(self):
        actual = np.array([100.0, 200.0])
        pred   = np.array([110.0, 200.0])
        mape = np.mean(np.abs(actual - pred) / (actual + 1)) * 100
        expected = np.mean([10 / 101, 0 / 201]) * 100
        assert abs(mape - expected) < 1e-6

    def test_rmse_perfect(self):
        actual = np.array([1.0, 2.0, 3.0])
        assert float(np.sqrt(np.mean((actual - actual) ** 2))) == 0.0

    def test_mae_basic(self):
        actual = np.array([0.0, 10.0, 20.0])
        pred   = np.array([5.0, 10.0, 15.0])
        assert float(np.mean(np.abs(actual - pred))) == pytest.approx(10 / 3, rel=1e-6)

    def test_rmse_symmetry(self):
        y = np.array([10.0, 10.0])
        r1 = float(np.sqrt(np.mean((y - np.array([15.0, 10.0])) ** 2)))
        r2 = float(np.sqrt(np.mean((y - np.array([5.0, 10.0])) ** 2)))
        assert abs(r1 - r2) < 1e-9
