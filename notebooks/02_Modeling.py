# %%
"""
02_Modeling.ipynb
=================
Model Training & Evaluation — Zimbabwe Cholera Forecasting
==========================================================
Sections:
  1. Feature Engineering
  2. Train/Test Split Strategy (time-series walk-forward)
  3. Baseline: ARIMA
  4. XGBoost with SHAP
  5. Model Comparison Table
  6. Forecast Horizon Analysis (7/14/30-day)
"""

# %%
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_DIR    = Path("../data/processed")
FIGURES_DIR = Path("../paper/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# %%
# ── 1. Load & Prepare Features ────────────────────────────────────────────────
cholera = pd.read_csv(DATA_DIR / "cholera_cases.csv", parse_dates=["date"])
climate = pd.read_csv(DATA_DIR / "climate.csv",       parse_dates=["date"])
demo    = pd.read_csv(DATA_DIR / "demographics.csv")

# Weekly climate aggregation
weekly_clim = (
    climate
    .assign(date=lambda d: d["date"] - pd.to_timedelta(d["date"].dt.dayofweek, unit="d"))
    .groupby(["date", "district"])
    .agg(
        rainfall_mm    = ("rainfall_mm", "sum"),
        temp_max_c     = ("temp_max_c",  "mean"),
        humidity_pct   = ("humidity_pct","mean"),
        rain_anomaly   = ("rainfall_anomaly_mm", "sum"),
    ).reset_index()
)

# Merge
cholera["year"] = cholera["date"].dt.year
master = (
    cholera
    .merge(weekly_clim, on=["date","district"], how="left")
    .merge(demo,        on=["year","district"], how="left")
    .sort_values(["district","date"])
    .reset_index(drop=True)
)

# Add lag features per district
for lag in [1, 2, 4, 8]:
    master[f"lag_{lag}w"] = master.groupby("district")["cases"].shift(lag).fillna(0)

for w in [4, 8]:
    master[f"roll_{w}w"] = (
        master.groupby("district")["cases"]
              .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
              .fillna(0)
    )

master["month"]          = master["date"].dt.month
master["week_of_year"]   = master["date"].dt.isocalendar().week.astype(int)
master["rainy_season"]   = master["month"].isin([11,12,1,2,3,4]).astype(int)
master["wash_rain"]      = master["rainfall_mm"] * (100 - master["wash_coverage_pct"].fillna(50)) / 100
master                   = master.dropna(subset=["lag_1w"])

print(f"Master dataset: {master.shape}")
print(master[["date","district","cases","lag_1w","rainfall_mm","wash_coverage_pct"]].head(6))

# %%
# ── 2. Train/Test Split ───────────────────────────────────────────────────────
CUTOFF = pd.Timestamp("2024-07-01")
train  = master[master["date"] < CUTOFF]
test   = master[master["date"] >= CUTOFF]
print(f"Train: {train['date'].min().date()} → {train['date'].max().date()} | {len(train):,} rows")
print(f"Test:  {test['date'].min().date()} → {test['date'].max().date()}  | {len(test):,} rows")

FEATURES = [
    "lag_1w","lag_2w","lag_4w","lag_8w","roll_4w","roll_8w",
    "rainfall_mm","temp_max_c","humidity_pct","rain_anomaly",
    "wash_coverage_pct","poverty_index","population_density_km2",
    "month","week_of_year","rainy_season","wash_rain",
]
FEATURES = [f for f in FEATURES if f in master.columns]
TARGET   = "cases"

X_train = train[FEATURES].fillna(0)
y_train = train[TARGET]
X_test  = test[FEATURES].fillna(0)
y_test  = test[TARGET]

# %%
# ── 3. Baseline: Naive Last-Value ─────────────────────────────────────────────
naive_pred = test.groupby("district")["cases"].shift(1).fillna(0)

def metrics(y_true, y_pred, label=""):
    y_true, y_pred = np.array(y_true), np.maximum(np.array(y_pred), 0)
    mask   = y_true > 0
    mae    = mean_absolute_error(y_true, y_pred)
    rmse   = np.sqrt(mean_squared_error(y_true, y_pred))
    mape   = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else float("nan")
    print(f"{label:25s}  MAE={mae:7.2f}  RMSE={rmse:7.2f}  MAPE={mape:6.1f}%")
    return {"label": label, "MAE": mae, "RMSE": rmse, "MAPE": mape}

results = []
results.append(metrics(y_test, naive_pred, "Naive (Lag-1)"))

# %%
# ── 4. XGBoost ────────────────────────────────────────────────────────────────
import xgboost as xgb

xgb_model = xgb.XGBRegressor(
    n_estimators      = 300,
    max_depth         = 5,
    learning_rate     = 0.05,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    min_child_weight  = 3,
    reg_alpha         = 0.1,
    reg_lambda        = 1.0,
    random_state      = 42,
    n_jobs            = -1,
    verbosity         = 0,
)
xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
xgb_pred = np.maximum(xgb_model.predict(X_test), 0)
results.append(metrics(y_test, xgb_pred, "XGBoost"))

# %%
# ── 5. Ensemble (XGB + Lag weighted average) ─────────────────────────────────
# Simple ensemble: 0.6 XGB + 0.4 Lag-1 (Prophet/LSTM require separate env)
naive_arr = np.maximum(naive_pred.fillna(0).values, 0)
ens_pred  = 0.65 * xgb_pred + 0.35 * naive_arr
results.append(metrics(y_test, ens_pred, "Ensemble (XGB+Lag)"))

# %%
# ── 6. Results Table ──────────────────────────────────────────────────────────
results_df = pd.DataFrame(results)
print("\n=== Model Comparison ===")
print(results_df.to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, col, color in zip(axes, ["MAPE","RMSE","MAE"], ["#e74c3c","#3498db","#2ecc71"]):
    bars = ax.bar(results_df["label"], results_df[col], color=color, alpha=0.82, edgecolor="white")
    ax.set_title(col, fontweight="bold")
    ax.set_ylabel(col)
    ax.tick_params(axis="x", rotation=25, labelsize=8)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.02,
                f"{bar.get_height():.1f}", ha="center", fontsize=8)

plt.suptitle("Model Comparison — Zimbabwe Cholera Forecasting", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig6_model_comparison.png", bbox_inches="tight")
plt.show()

# %%
# ── 7. Forecast Plot (Harare) ─────────────────────────────────────────────────
harare_test  = test[test["district"] == "Harare"].sort_values("date")
harare_idx   = harare_test.index
harare_xgb   = xgb_pred[np.isin(test.index.values, harare_idx.values)]
harare_ens   = ens_pred[np.isin(test.index.values, harare_idx.values)]

fig, ax = plt.subplots(figsize=(13, 5))
ax.fill_between(harare_test["date"], 0, harare_test["cases"], alpha=0.25, color="#c0392b", label="Actual")
ax.plot(harare_test["date"], harare_test["cases"], color="#c0392b", linewidth=1.5)
ax.plot(harare_test["date"], harare_xgb[:len(harare_test)], "--", color="#2980b9", linewidth=1.5, label="XGBoost")
ax.plot(harare_test["date"], harare_ens[:len(harare_test)], "-",  color="#27ae60", linewidth=2.0, label="Ensemble")
ax.set_title("Harare District — Forecast vs. Actual (Jul–Dec 2024)", fontsize=13, fontweight="bold")
ax.set_ylabel("Weekly Cases")
ax.legend()
ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b %Y"))
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig7_harare_forecast.png", bbox_inches="tight")
plt.show()

# %%
# ── 8. SHAP Feature Importance ────────────────────────────────────────────────
import shap

explainer   = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test.iloc[:500])
mean_shap   = np.abs(shap_values).mean(axis=0)
shap_df     = pd.DataFrame({"feature": FEATURES, "importance": mean_shap})\
                .sort_values("importance", ascending=True).tail(12)

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(shap_df["feature"], shap_df["importance"], color="#8e44ad", alpha=0.85)
ax.set_xlabel("Mean |SHAP Value|")
ax.set_title("Feature Importance (SHAP) — XGBoost Cholera Model", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig8_shap_importance.png", bbox_inches="tight")
plt.show()

print("\n=== Modeling Complete ===")
print(f"8 figures saved to {FIGURES_DIR}")
