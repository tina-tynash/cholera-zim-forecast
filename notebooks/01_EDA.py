# %%
"""
01_EDA.ipynb  (run as .py with Jupyter/VS Code)
================================================
Exploratory Data Analysis — Zimbabwe Cholera Forecasting Project
================================================================
Sections:
  1. Data Loading & Quality Check
  2. Temporal Trends (national + district)
  3. Spatial Patterns (district heatmaps)
  4. Outbreak Seasonality
  5. Climate–Cholera Correlation
  6. WASH & Socioeconomic Risk Factors
"""

# %% [markdown]
# # Exploratory Data Analysis
# **Project**: Enhancing Cholera Forecasting in Zimbabwe
# **Data Range**: 2018–2025 | 15 Districts

# %%
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path

DATA_DIR = Path("../data/processed")
FIGURES_DIR = Path("../paper/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# %%
# ── 1. Load Data ──────────────────────────────────────────────────────────────
cholera = pd.read_csv(DATA_DIR / "cholera_cases.csv", parse_dates=["date"])
climate = pd.read_csv(DATA_DIR / "climate.csv",       parse_dates=["date"])
demo    = pd.read_csv(DATA_DIR / "demographics.csv")
interv  = pd.read_csv(DATA_DIR / "interventions.csv", parse_dates=["date"])

print("=== Dataset Shapes ===")
print(f"Cholera:        {cholera.shape}")
print(f"Climate:        {climate.shape}")
print(f"Demographics:   {demo.shape}")
print(f"Interventions:  {interv.shape}")

# %%
# Quick quality check
print("\n=== Missing Values ===")
for name, df in [("Cholera", cholera), ("Climate", climate)]:
    pct = df.isnull().mean() * 100
    flagged = pct[pct > 0]
    print(f"{name}: {len(flagged)} columns with missing values")
    if len(flagged): print(flagged.to_string())

# %%
# ── 2. National Temporal Trends ───────────────────────────────────────────────
national = cholera.groupby("date")[["cases", "deaths"]].sum().reset_index()

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
fig.suptitle("Zimbabwe Cholera: National Weekly Trends (2018–2025)", fontsize=14, fontweight="bold")

axes[0].fill_between(national["date"], national["cases"], alpha=0.4, color="#c0392b")
axes[0].plot(national["date"], national["cases"], color="#c0392b", linewidth=1.2)
axes[0].set_ylabel("Weekly Cases")
axes[0].set_title("Reported Cholera Cases")

axes[1].fill_between(national["date"], national["deaths"], alpha=0.4, color="#2c3e50")
axes[1].plot(national["date"], national["deaths"], color="#2c3e50", linewidth=1.2)
axes[1].set_ylabel("Weekly Deaths")
axes[1].set_title("Reported Deaths")
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig1_national_trends.png", bbox_inches="tight")
plt.show()
print("Saved: fig1_national_trends.png")

# %%
# ── 3. District-Level Breakdown ───────────────────────────────────────────────
top_districts = (
    cholera.groupby("district")["cases"].sum()
           .sort_values(ascending=False)
           .head(8).index.tolist()
)

district_weekly = cholera[cholera["district"].isin(top_districts)]\
    .groupby(["date", "district"])["cases"].sum().reset_index()

fig, ax = plt.subplots(figsize=(14, 6))
for dist in top_districts:
    d = district_weekly[district_weekly["district"] == dist]
    ax.plot(d["date"], d["cases"], label=dist, linewidth=1.2, alpha=0.85)

ax.set_title("Weekly Cholera Cases — Top 8 Districts", fontsize=13, fontweight="bold")
ax.set_ylabel("Weekly Cases")
ax.legend(ncol=4, fontsize=8, loc="upper right")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig2_district_trends.png", bbox_inches="tight")
plt.show()

# %%
# ── 4. Seasonality Heatmap ────────────────────────────────────────────────────
cholera["year"]  = cholera["date"].dt.year
cholera["month"] = cholera["date"].dt.month
monthly = cholera.groupby(["year", "month"])["cases"].sum().unstack(level=0)

month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]

fig, ax = plt.subplots(figsize=(12, 5))
sns.heatmap(
    monthly,
    cmap="YlOrRd",
    annot=True,
    fmt=".0f",
    linewidths=0.4,
    ax=ax,
    yticklabels=month_labels,
    cbar_kws={"label": "Total Cases"},
)
ax.set_title("Seasonal Cholera Heatmap — Zimbabwe (2018–2025)", fontsize=13, fontweight="bold")
ax.set_xlabel("Year")
ax.set_ylabel("Month")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig3_seasonality_heatmap.png", bbox_inches="tight")
plt.show()

# %%
# ── 5. Climate–Cholera Correlation ────────────────────────────────────────────
weekly_climate = (
    climate.assign(
        week=lambda d: d["date"] - pd.to_timedelta(d["date"].dt.dayofweek, unit="d")
    )
    .groupby(["week", "district"])
    .agg(rain_sum=("rainfall_mm", "sum"), temp_mean=("temp_max_c", "mean"))
    .reset_index()
    .rename(columns={"week": "date"})
)

merged = pd.merge(
    cholera.groupby(["date", "district"])["cases"].sum().reset_index(),
    weekly_climate,
    on=["date", "district"],
    how="inner",
)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(merged["rain_sum"], merged["cases"], alpha=0.15, s=12, color="#2980b9")
axes[0].set_xlabel("Weekly Rainfall (mm)")
axes[0].set_ylabel("Weekly Cases")
axes[0].set_title("Rainfall vs. Cholera Cases")
r_rain = merged[["rain_sum","cases"]].corr().iloc[0,1]
axes[0].annotate(f"r = {r_rain:.3f}", xy=(0.7, 0.9), xycoords="axes fraction", fontsize=11)

axes[1].scatter(merged["temp_mean"], merged["cases"], alpha=0.15, s=12, color="#e74c3c")
axes[1].set_xlabel("Mean Temperature (°C)")
axes[1].set_ylabel("Weekly Cases")
axes[1].set_title("Temperature vs. Cholera Cases")
r_temp = merged[["temp_mean","cases"]].corr().iloc[0,1]
axes[1].annotate(f"r = {r_temp:.3f}", xy=(0.7, 0.9), xycoords="axes fraction", fontsize=11)

plt.suptitle("Climate–Cholera Correlations (All Districts, 2018–2025)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig4_climate_correlation.png", bbox_inches="tight")
plt.show()

# %%
# ── 6. WASH & Socioeconomic Risk ──────────────────────────────────────────────
latest_demo = demo[demo["year"] == 2023].copy()
total_cases = cholera.groupby("district")["cases"].sum().reset_index(name="total_cases")
risk_df = pd.merge(total_cases, latest_demo, on="district")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(risk_df["wash_coverage_pct"], risk_df["total_cases"],
                s=risk_df["population"] / 30000, alpha=0.7, color="#27ae60")
for _, row in risk_df.iterrows():
    axes[0].annotate(row["district"], (row["wash_coverage_pct"], row["total_cases"]),
                     fontsize=7, alpha=0.8)
axes[0].set_xlabel("WASH Coverage (%)")
axes[0].set_ylabel("Total Cases (2018–2025)")
axes[0].set_title("WASH Coverage vs. Cumulative Cholera Cases")

axes[1].scatter(risk_df["poverty_index"], risk_df["total_cases"],
                s=risk_df["population"] / 30000, alpha=0.7, color="#8e44ad")
for _, row in risk_df.iterrows():
    axes[1].annotate(row["district"], (row["poverty_index"], row["total_cases"]),
                     fontsize=7, alpha=0.8)
axes[1].set_xlabel("Poverty Index")
axes[1].set_ylabel("Total Cases (2018–2025)")
axes[1].set_title("Poverty vs. Cumulative Cholera Cases")

plt.suptitle("Socioeconomic Risk Factors — Zimbabwe Districts", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig5_wash_poverty.png", bbox_inches="tight")
plt.show()

print("\n=== EDA Complete ===")
print(f"5 figures saved to {FIGURES_DIR}")
print(f"Total cases 2018-2025: {national['cases'].sum():,}")
print(f"Peak week:   {national.loc[national['cases'].idxmax(), 'date'].date()} "
      f"({national['cases'].max():,} cases)")
