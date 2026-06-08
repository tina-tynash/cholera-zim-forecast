"""
Causal inference model using DoWhy for intervention impact estimation.

Builds a Directed Acyclic Graph (DAG) representing causal relationships:
    rainfall_mm  ──→  cases_weekly
    wash_coverage_pct ──→ cases_weekly
    ocv_campaign_active ──→ cases_weekly   (treatment)
    population_density_km2 ──→ cases_weekly
    poverty_index ──→ cases_weekly

Estimates the Average Treatment Effect (ATE) of:
- OCV vaccination campaigns on case reduction
- WASH coverage improvement on case reduction (counterfactual)

Usage:
    causal = CausalModel()
    causal.fit(df)
    ate = causal.estimate_ocv_ate()
    counterfactual = causal.what_if_wash(wash_increase_pct=20)
"""

from __future__ import annotations

import warnings
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

from src.utils.logger import logger

warnings.filterwarnings("ignore")


class CholleraCausalModel:
    """
    Causal inference wrapper for cholera intervention analysis.

    Uses DoWhy with a linear regression estimator (robust on small samples).
    Falls back to propensity-score matching if DoWhy is unavailable.
    """

    def __init__(self) -> None:
        self.df: Optional[pd.DataFrame] = None
        self.ate_ocv: Optional[float] = None
        self.ate_wash: Optional[float] = None
        self._dowhy_available = self._check_dowhy()

    def _check_dowhy(self) -> bool:
        """Check if DoWhy is importable."""
        try:
            import dowhy  # noqa: F401
            return True
        except ImportError:
            logger.warning("[Causal] DoWhy not installed. Using regression fallback.")
            return False

    def fit(self, df: pd.DataFrame) -> "CholleraCausalModel":
        """
        Store and prepare the dataset for causal estimation.

        Args:
            df: Master features DataFrame.

        Returns:
            Self.
        """
        required = [
            "cases_weekly", "ocv_campaign_active", "rainfall_mm",
            "wash_coverage_pct", "population_density_km2", "poverty_index"
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns for causal model: {missing}")

        self.df = df[required].dropna().copy()
        logger.info(f"[Causal] Prepared dataset: {len(self.df):,} rows.")
        return self

    def estimate_ocv_ate(self) -> float:
        """
        Estimate the Average Treatment Effect of OCV campaigns on weekly cases.

        Returns:
            ATE: expected reduction in weekly cases from OCV (negative = fewer cases).
        """
        if self.df is None:
            raise RuntimeError("Call .fit() before .estimate_ocv_ate().")

        if self._dowhy_available:
            return self._estimate_dowhy(
                treatment="ocv_campaign_active",
                outcome="cases_weekly",
                common_causes=["rainfall_mm", "wash_coverage_pct",
                               "population_density_km2", "poverty_index"],
            )
        else:
            return self._estimate_regression_ate("ocv_campaign_active")

    def _estimate_dowhy(
        self,
        treatment: str,
        outcome: str,
        common_causes: list[str],
    ) -> float:
        """Run DoWhy causal estimation."""
        import dowhy
        from dowhy import CausalModel

        graph_edges = " ".join([f"{c} -> {outcome};" for c in common_causes])
        graph_edges += f" {treatment} -> {outcome};"
        causal_graph = f"digraph {{ {graph_edges} }}"

        model = CausalModel(
            data=self.df,
            treatment=treatment,
            outcome=outcome,
            graph=causal_graph,
        )
        identified = model.identify_effect(proceed_when_unidentifiable=True)
        estimate = model.estimate_effect(
            identified,
            method_name="backdoor.linear_regression",
            control_value=0,
            treatment_value=1,
        )
        ate = float(estimate.value)
        self.ate_ocv = ate
        logger.info(f"[Causal/DoWhy] OCV ATE = {ate:.2f} cases/week")
        return ate

    def _estimate_regression_ate(self, treatment: str) -> float:
        """
        Regression-based ATE fallback (OLS with confounders as covariates).

        Args:
            treatment: Binary treatment column name.

        Returns:
            Estimated ATE.
        """
        from sklearn.linear_model import LinearRegression

        confounders = [
            c for c in ["rainfall_mm", "wash_coverage_pct",
                        "population_density_km2", "poverty_index"]
            if c in self.df.columns
        ]

        X = pd.get_dummies(self.df[[treatment] + confounders], drop_first=True)
        y = self.df["cases_weekly"].values

        model = LinearRegression().fit(X, y)
        # ATE = coefficient of the treatment indicator
        treatment_idx = list(X.columns).index(treatment)
        ate = float(model.coef_[treatment_idx])
        self.ate_ocv = ate
        logger.info(f"[Causal/Regression] OCV ATE = {ate:.2f} cases/week")
        return ate

    def what_if_wash(
        self,
        wash_increase_pct: float = 20.0,
        district: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Counterfactual analysis: what if WASH coverage increased by X%?

        Simulates the impact of WASH improvement on predicted case counts
        using a linear marginal effect estimated from the data.

        Args:
            wash_increase_pct: Percentage point increase in WASH coverage.
            district: Optionally restrict to a single district.

        Returns:
            Dictionary with baseline and counterfactual case estimates.
        """
        if self.df is None:
            raise RuntimeError("Call .fit() first.")

        data = self.df.copy()
        if district:
            data = data[data.get("district", pd.Series()) == district] if "district" in data else data

        # Marginal effect via OLS
        from sklearn.linear_model import LinearRegression
        features = ["wash_coverage_pct", "rainfall_mm",
                    "population_density_km2", "poverty_index"]
        available = [f for f in features if f in data.columns]

        X = data[available].fillna(data[available].median())
        y = data["cases_weekly"].values
        model = LinearRegression().fit(X, y)

        wash_idx = available.index("wash_coverage_pct")
        marginal_effect = model.coef_[wash_idx]

        baseline_cases = float(np.mean(y))
        reduction = marginal_effect * wash_increase_pct
        counterfactual_cases = max(0, baseline_cases + reduction)
        pct_change = (reduction / (baseline_cases + 1e-8)) * 100

        result = {
            "wash_increase_pct": wash_increase_pct,
            "baseline_weekly_cases": round(baseline_cases, 1),
            "counterfactual_weekly_cases": round(counterfactual_cases, 1),
            "estimated_reduction": round(-reduction, 1),
            "pct_change": round(pct_change, 2),
            "marginal_effect_per_pct": round(marginal_effect, 3),
        }

        logger.info(
            f"[Causal] WASH +{wash_increase_pct}% → "
            f"{result['estimated_reduction']:.1f} fewer cases/week "
            f"({result['pct_change']:.1f}%)"
        )
        return result

    def what_if_rainfall(
        self,
        rainfall_delta_mm: float = 20.0,
    ) -> Dict[str, Any]:
        """
        Counterfactual: impact of rainfall anomaly on expected case counts.

        Args:
            rainfall_delta_mm: Change in weekly rainfall (mm).

        Returns:
            Dictionary of baseline vs. counterfactual estimates.
        """
        if self.df is None:
            raise RuntimeError("Call .fit() first.")

        from sklearn.linear_model import LinearRegression
        features = ["rainfall_mm", "wash_coverage_pct",
                    "population_density_km2", "poverty_index"]
        available = [f for f in features if f in self.df.columns]

        X = self.df[available].fillna(self.df[available].median())
        y = self.df["cases_weekly"].values
        model = LinearRegression().fit(X, y)

        rain_idx = available.index("rainfall_mm")
        effect = model.coef_[rain_idx] * rainfall_delta_mm
        baseline = float(np.mean(y))

        return {
            "rainfall_delta_mm": rainfall_delta_mm,
            "baseline_weekly_cases": round(baseline, 1),
            "counterfactual_weekly_cases": round(max(0, baseline + effect), 1),
            "estimated_change": round(effect, 1),
            "pct_change": round((effect / (baseline + 1e-8)) * 100, 2),
        }

    def summary(self) -> Dict[str, Any]:
        """Return a summary of all estimated causal effects."""
        return {
            "ocv_ate_cases_per_week": self.ate_ocv,
            "wash_what_if_10pct": self.what_if_wash(10) if self.df is not None else None,
            "wash_what_if_20pct": self.what_if_wash(20) if self.df is not None else None,
        }
