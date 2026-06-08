"""
Data validation functions for all input datasets.

Uses assertion-based checks with descriptive error messages to catch
schema mismatches, out-of-range values, and null counts early in the pipeline.
"""

from __future__ import annotations

import pandas as pd
from src.utils.logger import logger


def _check_nulls(df: pd.DataFrame, name: str, max_null_pct: float = 0.05) -> None:
    """Raise if any column exceeds the null threshold."""
    null_pct = df.isnull().mean()
    violations = null_pct[null_pct > max_null_pct]
    if not violations.empty:
        raise ValueError(
            f"[{name}] High null rates: {violations.to_dict()}"
        )


def validate_cholera(df: pd.DataFrame) -> None:
    """
    Validate weekly cholera case data.

    Args:
        df: Raw cholera DataFrame.

    Raises:
        ValueError: On schema or range violations.
    """
    required = {"date", "district", "cases_weekly", "deaths_weekly"}
    missing = required - set(df.columns)
    assert not missing, f"Cholera data missing columns: {missing}"

    assert (df["cases_weekly"] >= 0).all(), "Negative case counts found."
    assert (df["deaths_weekly"] >= 0).all(), "Negative death counts found."
    assert (df["deaths_weekly"] <= df["cases_weekly"]).all(), "Deaths exceed cases."

    _check_nulls(df, "cholera")
    logger.info(
        f"[Validation] Cholera ✅ — {len(df):,} rows, "
        f"{df['district'].nunique()} districts, "
        f"dates {df['date'].min()} → {df['date'].max()}"
    )


def validate_climate(df: pd.DataFrame) -> None:
    """
    Validate weekly climate data.

    Args:
        df: Raw climate DataFrame.

    Raises:
        ValueError: On schema or range violations.
    """
    required = {"date", "district", "rainfall_mm", "temperature_c"}
    missing = required - set(df.columns)
    assert not missing, f"Climate data missing columns: {missing}"

    assert (df["rainfall_mm"] >= 0).all(), "Negative rainfall values found."
    assert df["temperature_c"].between(-5, 50).all(), "Temperature out of range [-5, 50]."

    _check_nulls(df, "climate")
    logger.info(
        f"[Validation] Climate ✅ — {len(df):,} rows, "
        f"rainfall max={df['rainfall_mm'].max():.1f}mm, "
        f"temp range={df['temperature_c'].min():.1f}–{df['temperature_c'].max():.1f}°C"
    )


def validate_demographics(df: pd.DataFrame) -> None:
    """
    Validate annual demographic / socioeconomic data.

    Args:
        df: Raw demographics DataFrame.

    Raises:
        ValueError: On schema or range violations.
    """
    required = {"year", "district", "population", "wash_coverage_pct"}
    missing = required - set(df.columns)
    assert not missing, f"Demographics missing columns: {missing}"

    assert (df["population"] > 0).all(), "Non-positive population found."
    assert df["wash_coverage_pct"].between(0, 100).all(), "WASH % out of [0, 100]."

    _check_nulls(df, "demographics")
    logger.info(
        f"[Validation] Demographics ✅ — {len(df):,} rows, "
        f"years {df['year'].min()}–{df['year'].max()}"
    )
