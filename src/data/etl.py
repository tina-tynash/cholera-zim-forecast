"""
etl.py
------
Full ETL pipeline: extract from sources, transform/clean,
engineer features, and load to SQLite (local-first).

Usage:
    python src/data/etl.py
"""

import os
import sqlite3
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_PROCESSED = Path("data/processed")
DB_PATH = "data/cholera_zim.db"


class CholeraETL:
    """ETL pipeline for Zimbabwe cholera forecasting data."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.engine = create_engine(f"sqlite:///{db_path}")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"ETL initialized | DB: {db_path}")

    # ------------------------------------------------------------------ #
    #  EXTRACT
    # ------------------------------------------------------------------ #

    def extract_cholera(self) -> pd.DataFrame:
        """Load cholera case data from processed CSV."""
        path = DATA_PROCESSED / "cholera_cases.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Cholera data not found at {path}. "
                "Run: python data/synthetic/generate_synthetic.py"
            )
        df = pd.read_csv(path, parse_dates=["date"])
        logger.info(f"Extracted cholera data: {len(df):,} rows")
        return df

    def extract_climate(self) -> pd.DataFrame:
        """Load climate data and aggregate to weekly."""
        path = DATA_PROCESSED / "climate.csv"
        if not path.exists():
            raise FileNotFoundError(f"Climate data not found at {path}.")
        df = pd.read_csv(path, parse_dates=["date"])

        # Aggregate daily → weekly (Monday-anchored)
        df["week"] = df["date"].dt.to_period("W-MON").dt.start_time
        weekly = (
            df.groupby(["week", "district"])
            .agg(
                rainfall_mm=("rainfall_mm", "sum"),
                temperature_c=("temperature_c", "mean"),
                humidity_pct=("humidity_pct", "mean"),
            )
            .reset_index()
            .rename(columns={"week": "date"})
        )
        logger.info(f"Extracted + aggregated climate data: {len(weekly):,} rows")
        return weekly

    def extract_demographics(self) -> pd.DataFrame:
        """Load demographic/socioeconomic data."""
        path = DATA_PROCESSED / "demographics.csv"
        if not path.exists():
            raise FileNotFoundError(f"Demographic data not found at {path}.")
        df = pd.read_csv(path)
        logger.info(f"Extracted demographic data: {len(df):,} rows")
        return df

    def extract_interventions(self) -> pd.DataFrame:
        """Load intervention event data."""
        path = DATA_PROCESSED / "interventions.csv"
        if not path.exists():
            raise FileNotFoundError(f"Interventions data not found at {path}.")
        df = pd.read_csv(path, parse_dates=["date"])
        logger.info(f"Extracted intervention data: {len(df):,} rows")
        return df

    # ------------------------------------------------------------------ #
    #  TRANSFORM
    # ------------------------------------------------------------------ #

    def transform_merge(
        self,
        cholera: pd.DataFrame,
        climate: pd.DataFrame,
        demographics: pd.DataFrame,
        interventions: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge all data sources into a single analysis-ready dataframe."""
        logger.info("Merging data sources...")

        # Ensure date types
        cholera["date"] = pd.to_datetime(cholera["date"])
        climate["date"] = pd.to_datetime(climate["date"])

        # Merge cholera + climate (weekly, by district)
        df = cholera.merge(climate, on=["date", "district"], how="left")

        # Add year column for demographic join
        df["year"] = df["date"].dt.year
        demographics["year"] = demographics["year"].astype(int)

        df = df.merge(
            demographics.drop(columns=["lat", "lon"], errors="ignore"),
            on=["district", "year"],
            how="left",
        )

        # Aggregate interventions to weekly counts per district
        interventions["week"] = interventions["date"].dt.to_period("W-MON").dt.start_time
        interv_weekly = (
            interventions.groupby(["week", "district"])
            .agg(
                intervention_count=("intervention_type", "count"),
                ocv_events=("intervention_type", lambda x: (x == "OCV_Campaign").sum()),
                wash_events=("intervention_type", lambda x: (x == "WASH_Infrastructure").sum()),
            )
            .reset_index()
            .rename(columns={"week": "date"})
        )
        interv_weekly["date"] = pd.to_datetime(interv_weekly["date"])
        df = df.merge(interv_weekly, on=["date", "district"], how="left")
        df[["intervention_count", "ocv_events", "wash_events"]] = (
            df[["intervention_count", "ocv_events", "wash_events"]].fillna(0)
        )

        logger.info(f"Merged dataset: {df.shape}")
        return df

    def transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer all forecasting features."""
        logger.info("Engineering features...")
        df = df.sort_values(["district", "date"]).copy()

        for district in df["district"].unique():
            mask = df["district"] == district
            sub = df.loc[mask].copy()

            # --- Lagged case features ---
            for lag in [1, 2, 4, 8, 12]:
                df.loc[mask, f"cases_lag{lag}w"] = sub["cases"].shift(lag)

            # --- Rolling statistics ---
            for window in [4, 8, 13]:
                df.loc[mask, f"cases_roll{window}w_mean"] = (
                    sub["cases"].shift(1).rolling(window).mean()
                )
                df.loc[mask, f"cases_roll{window}w_std"] = (
                    sub["cases"].shift(1).rolling(window).std()
                )

            # --- Climate lags ---
            for lag in [2, 4]:
                df.loc[mask, f"rainfall_lag{lag}w"] = sub["rainfall_mm"].shift(lag)

            # --- Rainfall anomaly (deviation from 52-week rolling mean) ---
            rain_mean_52 = sub["rainfall_mm"].shift(1).rolling(52, min_periods=4).mean()
            rain_std_52 = sub["rainfall_mm"].shift(1).rolling(52, min_periods=4).std()
            df.loc[mask, "rainfall_anomaly"] = (
                (sub["rainfall_mm"] - rain_mean_52) / (rain_std_52 + 1e-8)
            )

            # --- Cumulative rainfall (4-week) ---
            df.loc[mask, "rainfall_4w_cumulative"] = (
                sub["rainfall_mm"].shift(1).rolling(4).sum()
            )

        # --- WASH × Climate interaction ---
        if "wash_access_pct" in df.columns:
            df["wash_rain_interaction"] = df["wash_access_pct"] * df["rainfall_mm"]
            df["risk_score_raw"] = (
                (1 - df["wash_access_pct"]) * df["rainfall_4w_cumulative"].clip(0, 200) / 200
            )

        # --- Temporal features ---
        df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
        df["month"] = df["date"].dt.month
        df["year"] = df["date"].dt.year
        df["is_rainy_season"] = df["month"].isin([11, 12, 1, 2, 3, 4]).astype(int)
        df["year_idx"] = df["year"] - df["year"].min()   # trend index

        # --- Log-transform target ---
        df["cases_log"] = np.log1p(df["cases"])

        logger.info(f"Feature engineering complete. Columns: {df.shape[1]}")
        return df

    def transform_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final cleaning: handle missing, clip outliers."""
        # Forward-fill demographic columns (annual data → weekly)
        demo_cols = [
            "wash_access_pct", "poverty_rate", "literacy_rate",
            "health_access_pct", "pop_density_km2", "population",
        ]
        for col in demo_cols:
            if col in df.columns:
                df[col] = df.groupby("district")[col].transform(
                    lambda s: s.ffill().bfill()
                )

        # Clip extreme outliers (99.5th percentile)
        for col in ["cases", "rainfall_mm", "temperature_c"]:
            if col in df.columns:
                upper = df[col].quantile(0.995)
                df[col] = df[col].clip(upper=upper)

        # Drop rows where target is NA (first few weeks due to lags)
        df = df.dropna(subset=["cases"]).reset_index(drop=True)

        missing_pct = df.isnull().mean().mean()
        logger.info(f"Cleaning done | Overall missing: {missing_pct:.2%}")
        return df

    # ------------------------------------------------------------------ #
    #  LOAD
    # ------------------------------------------------------------------ #

    def load_to_db(self, df: pd.DataFrame, table_name: str = "cholera_features") -> None:
        """Load transformed dataframe into SQLite."""
        df.to_sql(table_name, self.engine, if_exists="replace", index=False)
        logger.info(f"Loaded {len(df):,} rows → DB table '{table_name}'")

    def load_to_csv(self, df: pd.DataFrame, filename: str = "features.csv") -> None:
        """Save features CSV for notebook use."""
        out = DATA_PROCESSED / filename
        df.to_csv(out, index=False)
        logger.info(f"Saved features → {out}")

    # ------------------------------------------------------------------ #
    #  RUN
    # ------------------------------------------------------------------ #

    def run(self) -> pd.DataFrame:
        """Execute the full ETL pipeline."""
        logger.info("=" * 55)
        logger.info("Starting Cholera Zimbabwe ETL Pipeline")
        logger.info("=" * 55)

        # Extract
        cholera = self.extract_cholera()
        climate = self.extract_climate()
        demographics = self.extract_demographics()
        interventions = self.extract_interventions()

        # Transform
        merged = self.transform_merge(cholera, climate, demographics, interventions)
        featured = self.transform_features(merged)
        clean = self.transform_clean(featured)

        # Load
        self.load_to_db(clean)
        self.load_to_csv(clean, "features.csv")

        logger.info("=" * 55)
        logger.info(f"ETL Complete | Final shape: {clean.shape}")
        logger.info("=" * 55)
        return clean


if __name__ == "__main__":
    etl = CholeraETL()
    df = etl.run()
    print(f"\nFinal dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Districts: {df['district'].nunique()}")
    print(f"Date range: {df['date'].min().date()} → {df['date'].max().date()}")
