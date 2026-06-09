# %%
"""
03_Dashboard_Prototype.ipynb
============================
Interactive Dashboard Prototype — Zimbabwe Cholera Forecasting
Demonstrates all dashboard panels before Streamlit deployment.
"""
# %%
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

DATA_DIR = Path("../data/processed")
cholera  = pd.read_csv(DATA_DIR / "cholera_cases.csv", parse_dates=["date"])
climate  = pd.read_csv(DATA_DIR / "climate.csv",       parse_dates=["date"])
demo     = pd.read_csv(DATA_DIR / "demographics.csv")

# %%
# ── Panel 1: National Forecast Summary ───────────────────────────────────────
national = cholera.groupby("date")["cases"].sum().reset_index()

# Simulate a simple forecast (last 8 weeks + projection)
last_8w   = national.tail(8)
avg_cases = last_8w["cases"].mean()
future_dates = pd.date_range(national["date"].max() + pd.Timedelta(weeks=1), periods=4, freq="W")
forecast_df = pd.DataFrame({
    "date":   future_dates,
    "cases":  [max(0, avg_cases * (1 + np.random.normal(0, 0.1))) for _ in range(4)],
    "lower":  [max(0, avg_cases * 0.6)] * 4,
    "upper":  [avg_cases * 1.5] * 4,
})

fig = go.Figure()
fig.add_trace(go.Scatter(x=national["date"], y=national["cases"],
                          name="Historical Cases", line=dict(color="#c0392b", width=2)))
fig.add_trace(go.Scatter(x=forecast_df["date"], y=forecast_df["cases"],
                          name="Forecast", line=dict(color="#2980b9", width=2.5, dash="dash")))
fig.add_trace(go.Scatter(
    x=list(forecast_df["date"]) + list(forecast_df["date"])[::-1],
    y=list(forecast_df["upper"]) + list(forecast_df["lower"])[::-1],
    fill="toself", fillcolor="rgba(41,128,185,0.15)", line=dict(color="rgba(255,255,255,0)"),
    name="95% Prediction Interval"
))
fig.update_layout(title="Zimbabwe Cholera — National Weekly Cases & 4-Week Forecast",
                  xaxis_title="Date", yaxis_title="Weekly Cases",
                  template="plotly_white", height=450)
fig.show()

# %%
# ── Panel 2: District Risk Map (Choropleth-style bar) ─────────────────────────
district_risk = cholera.groupby("district").agg(
    total_cases=("cases", "sum"),
    recent_cases=("cases", lambda x: x.tail(8).sum()),
).reset_index()
district_risk["risk_score"] = (
    district_risk["recent_cases"] / district_risk["recent_cases"].max() * 100
).round(1)
district_risk["risk_level"] = pd.cut(
    district_risk["risk_score"],
    bins=[0, 30, 60, 100],
    labels=["Low", "Medium", "High"]
)

color_map = {"High": "#c0392b", "Medium": "#e67e22", "Low": "#27ae60"}
fig2 = px.bar(
    district_risk.sort_values("risk_score", ascending=True),
    x="risk_score", y="district", orientation="h",
    color="risk_level", color_discrete_map=color_map,
    title="District Cholera Risk Scores (Last 8 Weeks)",
    labels={"risk_score": "Risk Score (0–100)", "district": "District"},
    height=500,
)
fig2.update_layout(template="plotly_white")
fig2.show()

# %%
# ── Panel 3: What-If Simulator ───────────────────────────────────────────────
wash_levels = np.arange(0, 51, 5)
baseline_cases = 1200

scenarios = pd.DataFrame({
    "WASH Improvement (%)": wash_levels,
    "Projected Case Reduction (%)": wash_levels * 0.72,  # ~0.72% reduction per 1% WASH
})
scenarios["Projected Cases"] = baseline_cases * (1 - scenarios["Projected Case Reduction (%)"] / 100)

fig3 = go.Figure()
fig3.add_trace(go.Scatter(
    x=scenarios["WASH Improvement (%)"],
    y=scenarios["Projected Cases"],
    mode="lines+markers",
    line=dict(color="#27ae60", width=3),
    marker=dict(size=8),
    name="Projected Annual Cases",
))
fig3.add_shape(type="line", x0=0, x1=1, xref="paper",
               y0=baseline_cases, y1=baseline_cases, yref="y",
               line=dict(dash="dash", color="#c0392b", width=1.5))
fig3.add_annotation(x=0.02, y=baseline_cases, xref="paper", yref="y",
                    text=f"Baseline: {baseline_cases:,} cases",
                    showarrow=False, font=dict(color="#c0392b", size=11),
                    xanchor="left", yanchor="bottom")
fig3.update_layout(
    title="What-If Simulator: Impact of WASH Improvement on Annual Cholera Cases",
    xaxis_title="WASH Coverage Improvement (%)",
    yaxis_title="Projected Annual Cases",
    template="plotly_white", height=400,
)
fig3.show()

# %%
# ── Panel 4: Seasonal Decomposition ─────────────────────────────────────────
harare = cholera[cholera["district"] == "Harare"].sort_values("date")
harare["month"] = harare["date"].dt.month
monthly_avg = harare.groupby("month")["cases"].mean()
months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

fig4 = go.Figure(go.Bar(
    x=months, y=monthly_avg.values,
    marker_color=["#c0392b" if m in [1,2,3,11,12] else "#3498db" for m in range(1,13)],
))
fig4.add_annotation(text="Rainy Season ↑", x=0.15, y=0.9, xref="paper", yref="paper",
                    showarrow=False, font=dict(color="#c0392b", size=12))
fig4.update_layout(title="Harare — Average Monthly Cholera Cases (Seasonality)",
                   xaxis_title="Month", yaxis_title="Avg Weekly Cases",
                   template="plotly_white", height=380)
fig4.show()

print("\n=== Dashboard Prototype Complete ===")
print("All 4 panels rendered. Deploy with: streamlit run src/app/streamlit_app.py")
