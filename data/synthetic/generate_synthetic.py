"""
generate_synthetic.py
---------------------
Generates reproducible synthetic cholera, climate, and demographic datasets
for Zimbabwe districts (2018–2025), matching known epidemiological patterns.

Usage:
    python data/synthetic/generate_synthetic.py
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Reproducible seed
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Zimbabwe districts with coordinates and population
DISTRICTS = [
    {"name": "Harare",      "lat": -17.8292, "lon": 31.0522, "pop": 2123132, "wash_base": 0.72, "density": 2450},
    {"name": "Bulawayo",    "lat": -20.1325, "lon": 28.6264, "pop": 653337,  "wash_base": 0.68, "density": 1200},
    {"name": "Chitungwiza", "lat": -18.0127, "lon": 31.0758, "pop": 354472,  "wash_base": 0.58, "density": 3100},
    {"name": "Mutare",      "lat": -18.9707, "lon": 32.6709, "pop": 188243,  "wash_base": 0.61, "density": 420},
    {"name": "Gweru",       "lat": -19.4500, "lon": 29.8167, "pop": 157400,  "wash_base": 0.65, "density": 380},
    {"name": "Kwekwe",      "lat": -18.9281, "lon": 29.8147, "pop": 99146,   "wash_base": 0.60, "density": 290},
    {"name": "Kadoma",      "lat": -18.3333, "lon": 29.9167, "pop": 91688,   "wash_base": 0.62, "density": 260},
    {"name": "Masvingo",    "lat": -20.0667, "lon": 30.8333, "pop": 90286,   "wash_base": 0.55, "density": 210},
    {"name": "Chinhoyi",    "lat": -17.3667, "lon": 30.2000, "pop": 74309,   "wash_base": 0.63, "density": 180},
    {"name": "Norton",      "lat": -17.8833, "lon": 30.7000, "pop": 67527,   "wash_base": 0.59, "density": 320},
]

# Outbreak periods (historically grounded)
OUTBREAK_PERIODS = [
    ("2018-09-01", "2019-06-30", 3.5),   # Major 2018-19 outbreak
    ("2020-01-01", "2020-04-30", 1.8),   # 2020 minor surge
    ("2021-11-01", "2022-03-31", 2.1),   # 2021-22 surge
    ("2023-02-01", "2023-08-31", 2.8),   # 2023 outbreak
    ("2024-11-01", "2025-03-31", 1.5),   # 2024-25 surge
]


def outbreak_multiplier(date: pd.Timestamp) -> float:
    """Return outbreak multiplier for a given date."""
    for start, end, mult in OUTBREAK_PERIODS:
        if pd.Timestamp(start) <= date <= pd.Timestamp(end):
            return mult
    return 1.0


def generate_cholera_cases() -> pd.DataFrame:
    """
    Generate weekly district-level cholera case counts.
    Models: baseline seasonality + rainfall response + outbreak periods + noise.
    """
    print("Generating cholera case data...")
    start = datetime(2018, 1, 1)
    end = datetime(2025, 12, 31)
    dates = pd.date_range(start, end, freq="W-MON")

    records = []
    for district in DISTRICTS:
        base_rate = district["pop"] / 100000 * 0.8   # ~0.8 cases per 100k baseline
        wash_factor = 1 - (district["wash_base"] - 0.5) * 1.2  # lower WASH → higher risk

        for date in dates:
            # Seasonal component: rainy season Nov-Apr peaks
            week_of_year = date.dayofyear / 365
            seasonal = 1 + 1.8 * np.sin(2 * np.pi * (week_of_year - 0.1)) ** 2
            if 11 <= date.month or date.month <= 4:
                seasonal *= 1.4

            # Outbreak multiplier
            outbreak_mult = outbreak_multiplier(date)

            # Expected cases
            expected = base_rate * seasonal * wash_factor * outbreak_mult
            expected = max(0.1, expected)

            # Negative binomial noise (overdispersed counts)
            n_param = 5.0
            p_param = n_param / (n_param + expected)
            cases = int(rng.negative_binomial(n_param, p_param))

            # Deaths (~2% CFR during outbreaks, 0.5% otherwise)
            cfr = 0.02 if outbreak_mult > 2 else 0.005
            deaths = int(rng.binomial(max(0, cases), cfr))

            records.append({
                "date": date,
                "district": district["name"],
                "lat": district["lat"],
                "lon": district["lon"],
                "cases": cases,
                "deaths": deaths,
                "population": district["pop"],
                "incidence_rate": round(cases / district["pop"] * 100000, 4),
                "cfr": round(deaths / max(1, cases), 4),
            })

    df = pd.DataFrame(records)
    df = df.sort_values(["district", "date"]).reset_index(drop=True)
    out_path = OUTPUT_DIR / "cholera_cases.csv"
    df.to_csv(out_path, index=False)
    print(f"  ✓ Saved {len(df):,} records → {out_path}")
    return df


def generate_climate_data() -> pd.DataFrame:
    """
    Generate daily climate data for each district.
    Models: Zimbabwe's bi-modal rainfall + temperature seasonality.
    """
    print("Generating climate data...")
    start = datetime(2018, 1, 1)
    end = datetime(2025, 12, 31)
    dates = pd.date_range(start, end, freq="D")

    records = []
    for district in DISTRICTS:
        # Temperature baseline varies by elevation proxy (lat)
        temp_base = 22 + (district["lat"] + 20) * 0.3

        for date in dates:
            doy = date.dayofyear
            t = doy / 365.25

            # Temperature: ~25°C peak in Oct, ~15°C in July
            temp = temp_base + 5 * np.sin(2 * np.pi * t - 0.8) + rng.normal(0, 1.5)

            # Rainfall: Nov–April rainy season
            rain_seasonal = max(0, 3.5 * np.sin(2 * np.pi * (t - 0.1)) ** 3)
            if 11 <= date.month or date.month <= 4:
                rain_mean = rain_seasonal * (15 + 8 * rng.random())
            else:
                rain_mean = rain_seasonal * 0.3

            # Stochastic rainfall (gamma distribution)
            if rain_mean > 0.5:
                rainfall = float(rng.gamma(shape=1.5, scale=rain_mean / 1.5))
            else:
                rainfall = float(rng.exponential(0.3))

            # Humidity
            humidity = 40 + 40 * (rainfall / (rainfall + 5)) + rng.normal(0, 5)
            humidity = float(np.clip(humidity, 10, 98))

            records.append({
                "date": date,
                "district": district["name"],
                "temperature_c": round(float(temp), 2),
                "rainfall_mm": round(float(rainfall), 3),
                "humidity_pct": round(float(humidity), 2),
            })

    df = pd.DataFrame(records)
    df = df.sort_values(["district", "date"]).reset_index(drop=True)
    out_path = OUTPUT_DIR / "climate.csv"
    df.to_csv(out_path, index=False)
    print(f"  ✓ Saved {len(df):,} records → {out_path}")
    return df


def generate_demographics() -> pd.DataFrame:
    """
    Generate annual district-level socioeconomic indicators.
    Sources: ZimStat, World Bank structure (synthetic values).
    """
    print("Generating demographic/socioeconomic data...")
    years = list(range(2018, 2026))
    records = []

    for district in DISTRICTS:
        wash = district["wash_base"]
        for year in years:
            # Slight annual improvement in WASH, noisy
            wash_current = min(0.95, wash + (year - 2018) * 0.008 + rng.normal(0, 0.005))
            poverty = max(0.05, 0.45 - (year - 2018) * 0.01 + rng.normal(0, 0.02))
            literacy = min(0.98, 0.88 + (year - 2018) * 0.003 + rng.normal(0, 0.005))
            health_access = min(0.95, 0.65 + (year - 2018) * 0.01 + rng.normal(0, 0.01))

            records.append({
                "year": year,
                "district": district["name"],
                "lat": district["lat"],
                "lon": district["lon"],
                "population": int(district["pop"] * (1.025 ** (year - 2018))),  # 2.5% growth
                "pop_density_km2": district["density"],
                "wash_access_pct": round(float(wash_current), 4),
                "poverty_rate": round(float(poverty), 4),
                "literacy_rate": round(float(literacy), 4),
                "health_access_pct": round(float(health_access), 4),
                "open_defecation_pct": round(max(0, 0.15 - wash_current * 0.1), 4),
                "water_trucking_events": int(rng.poisson(2 + (1 - wash_current) * 8)),
                "ocv_coverage_pct": round(float(np.clip(rng.normal(0.55, 0.15), 0, 0.95)), 4),
            })

    df = pd.DataFrame(records)
    out_path = OUTPUT_DIR / "demographics.csv"
    df.to_csv(out_path, index=False)
    print(f"  ✓ Saved {len(df):,} records → {out_path}")
    return df


def generate_interventions() -> pd.DataFrame:
    """Generate event-level intervention records (OCV campaigns, WASH projects)."""
    print("Generating intervention data...")
    records = []
    intervention_types = ["OCV_Campaign", "Water_Trucking", "WASH_Infrastructure",
                          "Hygiene_Promotion", "Case_Isolation", "Water_Treatment"]

    for district in DISTRICTS:
        n_events = rng.integers(15, 40)
        for _ in range(int(n_events)):
            year = rng.integers(2018, 2026)
            month = rng.integers(1, 13)
            day = rng.integers(1, 28)
            try:
                event_date = datetime(int(year), int(month), int(day))
            except ValueError:
                event_date = datetime(int(year), int(month), 1)

            itype = rng.choice(intervention_types)
            records.append({
                "date": event_date,
                "district": district["name"],
                "intervention_type": itype,
                "coverage_pct": round(float(rng.uniform(0.2, 0.9)), 3),
                "duration_days": int(rng.integers(3, 90)),
                "beneficiaries": int(rng.integers(500, 50000)),
                "funding_source": rng.choice(["WHO", "UNICEF", "MoHCC", "MSF", "USAID", "World Bank"]),
            })

    df = pd.DataFrame(records).sort_values(["district", "date"]).reset_index(drop=True)
    out_path = OUTPUT_DIR / "interventions.csv"
    df.to_csv(out_path, index=False)
    print(f"  ✓ Saved {len(df):,} records → {out_path}")
    return df


def main() -> None:
    """Generate all synthetic datasets."""
    print("\n" + "="*60)
    print("  Synthetic Data Generator — Zimbabwe Cholera Forecasting")
    print(f"  Random seed: {RANDOM_SEED} | Period: 2018–2025")
    print("="*60 + "\n")

    cholera_df = generate_cholera_cases()
    climate_df = generate_climate_data()
    demo_df = generate_demographics()
    interv_df = generate_interventions()

    # Summary stats
    print("\n📊 Summary:")
    print(f"  Cholera records:      {len(cholera_df):>8,}")
    print(f"  Climate records:      {len(climate_df):>8,}")
    print(f"  Demographic records:  {len(demo_df):>8,}")
    print(f"  Intervention records: {len(interv_df):>8,}")
    print(f"\n  Total cases generated: {cholera_df['cases'].sum():,}")
    print(f"  Total deaths generated: {cholera_df['deaths'].sum():,}")
    print(f"  Date range: {cholera_df['date'].min().date()} → {cholera_df['date'].max().date()}")
    print("\n✅ All synthetic datasets saved to data/processed/\n")


if __name__ == "__main__":
    main()
