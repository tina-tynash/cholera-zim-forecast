"""
Visualization utilities using Plotly.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np


DISTRICT_COORDS = {
    "Harare":      (-17.8252, 31.0335),
    "Bulawayo":    (-20.1325, 28.6262),
    "Chitungwiza": (-18.0127, 31.0753),
    "Mutare":      (-18.9707, 32.6709),
    "Gweru":       (-19.4500, 29.8167),
    "Kwekwe":      (-18.9281, 29.8149),
    "Masvingo":    (-20.0744, 30.8328),
    "Chinhoyi":    (-17.3667, 30.2000),
    "Marondera":   (-18.1833, 31.5500),
    "Norton":      (-17.8811, 30.7008),
    "Beitbridge":  (-22.2167, 30.0000),
    "Chipinge":    (-20.1922, 32.6240),
}

PALETTE = {
    "primary": "#C8102E",
    "secondary": "#FFD700",
    "neutral":  "#1E3A5F",
    "light":    "#F5F5F5",
    "accent":   "#2ECC71",
}


def plot_forecast(
    historical: pd.DataFrame,
    forecast: pd.DataFrame,
    district: str,
    title: str = None,
) -> go.Figure:
    """
    Plot historical cases with forecast ribbon.

    Parameters
    ----------
    historical : DataFrame with 'date' and 'cases' columns.
    forecast : DataFrame with 'date', 'yhat', 'yhat_lower', 'yhat_upper'.
    district : District name for labeling.
    """
    hist = historical[historical["district"] == district].sort_values("date")
    fc = forecast[forecast["district"] == district].sort_values("date")

    fig = go.Figure()

    # Historical
    fig.add_trace(go.Scatter(
        x=hist["date"], y=hist["cases"],
        name="Reported Cases", mode="lines+markers",
        line=dict(color=PALETTE["neutral"], width=2),
        marker=dict(size=4),
    ))

    # Confidence interval ribbon
    fig.add_trace(go.Scatter(
        x=pd.concat([fc["date"], fc["date"][::-1]]),
        y=pd.concat([fc["yhat_upper"], fc["yhat_lower"][::-1]]),
        fill="toself", fillcolor="rgba(200,16,46,0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name="95% CI",
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=fc["date"], y=fc["yhat"],
        name="Ensemble Forecast",
        line=dict(color=PALETTE["primary"], width=2.5, dash="dash"),
    ))

    fig.update_layout(
        title=title or f"Cholera Forecast — {district}",
        xaxis_title="Date",
        yaxis_title="Weekly Cases",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        font=dict(family="Georgia, serif"),
        height=420,
    )
    return fig


def plot_risk_map(risk_df: pd.DataFrame) -> go.Figure:
    """
    Bubble map of district risk scores.

    Parameters
    ----------
    risk_df : DataFrame with columns 'district', 'risk_score', 'cases'.
    """
    risk_df = risk_df.copy()
    risk_df["lat"] = risk_df["district"].map(lambda d: DISTRICT_COORDS.get(d, (-18, 30))[0])
    risk_df["lon"] = risk_df["district"].map(lambda d: DISTRICT_COORDS.get(d, (-18, 30))[1])

    fig = px.scatter_mapbox(
        risk_df, lat="lat", lon="lon",
        size="cases", color="risk_score",
        hover_name="district",
        hover_data={"cases": True, "risk_score": ":.2f", "lat": False, "lon": False},
        color_continuous_scale=["#2ECC71", "#FFD700", "#C8102E"],
        size_max=40, zoom=5.5,
        mapbox_style="carto-positron",
        title="Zimbabwe District Cholera Risk Map",
        center={"lat": -19.0, "lon": 30.0},
    )
    fig.update_layout(height=500, margin=dict(r=0, t=40, l=0, b=0))
    return fig


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Horizontal bar chart of XGBoost feature importance."""
    df = importance_df.head(top_n).sort_values("importance")
    fig = go.Figure(go.Bar(
        x=df["importance"], y=df["feature"],
        orientation="h",
        marker_color=PALETTE["primary"],
    ))
    fig.update_layout(
        title="Top Feature Importances (XGBoost)",
        xaxis_title="Importance Score",
        template="plotly_white",
        height=400,
        font=dict(family="Georgia, serif"),
    )
    return fig


def plot_district_comparison(panel: pd.DataFrame, metric: str = "cases") -> go.Figure:
    """Line chart comparing a metric across all districts."""
    fig = go.Figure()
    for district in panel["district"].unique():
        d = panel[panel["district"] == district].sort_values("date")
        fig.add_trace(go.Scatter(
            x=d["date"], y=d[metric],
            name=district, mode="lines", line=dict(width=1.5),
        ))
    fig.update_layout(
        title=f"{metric.replace('_', ' ').title()} by District",
        xaxis_title="Date", yaxis_title=metric,
        template="plotly_white", height=420,
        legend=dict(orientation="h"),
        font=dict(family="Georgia, serif"),
    )
    return fig
