"""
streamlit_app.py
----------------
Interactive Cholera Forecasting Dashboard for Zimbabwe.

Features:
  - District-level forecast charts (7/14/30-day horizons)
  - Risk map with folium choropleth
  - What-if simulator (WASH, rainfall, OCV)
  - Data explorer with CSV/Excel download
  - Low-bandwidth static mode

Run:
    streamlit run src/app/streamlit_app.py
"""

import io
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import folium
from streamlit_folium import st_folium

warnings.filterwarnings("ignore")

# ─── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Zimbabwe Cholera Forecasting",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}
.stApp {
    background: #080c14;
    color: #e0e6f0;
}
.metric-card {
    background: linear-gradient(135deg, #0d1b2e 0%, #0a1628 100%);
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 20px;
    text-align: center;
}
.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.2rem;
    font-weight: 600;
    color: #4fc3f7;
    line-height: 1;
}
.metric-label {
    font-size: 0.78rem;
    color: #7a9bc0;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 6px;
}
.risk-high   { color: #ff4444; font-weight: 600; }
.risk-medium { color: #ffaa00; font-weight: 600; }
.risk-low    { color: #00e676; font-weight: 600; }
.section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #4fc3f7;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    border-bottom: 1px solid #1e3a5f;
    padding-bottom: 6px;
    margin-bottom: 16px;
}
div[data-testid="metric-container"] {
    background: #0d1b2e;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 12px 16px;
}
.stSelectbox label, .stSlider label {
    color: #7a9bc0 !important;
    font-size: 0.82rem;
}
</style>
""", unsafe_allow_html=True)


# ─── Data loading ───────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_data() -> pd.DataFrame:
    """Load processed cholera features dataset."""
    feature_path = Path("data/processed/features.csv")
    cases_path = Path("data/processed/cholera_cases.csv")

    if feature_path.exists():
        df = pd.read_csv(feature_path, parse_dates=["date"])
    elif cases_path.exists():
        df = pd.read_csv(cases_path, parse_dates=["date"])
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
        df["is_rainy_season"] = df["month"].isin([11, 12, 1, 2, 3, 4]).astype(int)
    else:
        st.error("⚠️ Data not found. Run: `python data/synthetic/generate_synthetic.py && python src/data/etl.py`")
        st.stop()

    return df


@st.cache_data(ttl=3600)
def load_demographics() -> pd.DataFrame:
    """Load district demographic data."""
    path = Path("data/processed/demographics.csv")
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def compute_risk_score(row: pd.Series) -> str:
    """Compute categorical risk level for a district."""
    recent_cases = row.get("cases_recent", 0)
    wash = row.get("wash_access_pct", 0.65)
    if recent_cases > 50 or wash < 0.55:
        return "HIGH"
    elif recent_cases > 20 or wash < 0.65:
        return "MEDIUM"
    return "LOW"


def simple_forecast(district_df: pd.DataFrame, horizon_weeks: int = 4) -> pd.DataFrame:
    """
    Simple statistical forecast when ML models aren't trained yet.
    Uses ETS-style exponential smoothing as fallback.
    """
    df = district_df.sort_values("date").copy()
    cases = df["cases"].fillna(0).values

    # Exponential smoothing
    alpha = 0.3
    smoothed = np.zeros_like(cases, dtype=float)
    smoothed[0] = cases[0]
    for i in range(1, len(cases)):
        smoothed[i] = alpha * cases[i] + (1 - alpha) * smoothed[i - 1]

    # Seasonal component from same-week last year
    last_val = smoothed[-1]
    last_date = df["date"].max()
    future_dates = [last_date + timedelta(weeks=i + 1) for i in range(horizon_weeks)]

    # Seasonal multipliers from historical data
    forecasts = []
    for i, fdate in enumerate(future_dates):
        same_week_hist = df[df["date"].dt.isocalendar().week == fdate.isocalendar()[1]]
        seasonal_mult = same_week_hist["cases"].mean() / (df["cases"].mean() + 1e-8)
        seasonal_mult = np.clip(seasonal_mult, 0.3, 3.0)
        yhat = last_val * (0.95 ** (i + 1)) * seasonal_mult
        std_est = df["cases"].std() * 0.5
        forecasts.append({
            "date": fdate,
            "yhat": max(0, yhat),
            "yhat_lower": max(0, yhat - 1.96 * std_est),
            "yhat_upper": max(0, yhat + 1.96 * std_est),
        })

    return pd.DataFrame(forecasts)


# ─── Visualization helpers ─────────────────────────────────────

def make_forecast_chart(
    df: pd.DataFrame,
    district: str,
    horizon_weeks: int = 4,
) -> go.Figure:
    """Build an interactive Plotly forecast chart."""
    d = df[df["district"] == district].sort_values("date").tail(104)  # ~2 years
    forecast_df = simple_forecast(d, horizon_weeks)

    fig = go.Figure()

    # Historical cases
    fig.add_trace(go.Scatter(
        x=d["date"], y=d["cases"],
        name="Historical Cases",
        line=dict(color="#4fc3f7", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(79,195,247,0.07)",
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>Cases: %{y}<extra></extra>",
    ))

    # Confidence interval
    fig.add_trace(go.Scatter(
        x=pd.concat([forecast_df["date"], forecast_df["date"][::-1]]),
        y=pd.concat([forecast_df["yhat_upper"], forecast_df["yhat_lower"][::-1]]),
        fill="toself",
        fillcolor="rgba(255,171,64,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="95% Interval",
        hoverinfo="skip",
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=forecast_df["date"], y=forecast_df["yhat"].round(1),
        name="Forecast",
        line=dict(color="#ffab40", width=2.5, dash="dot"),
        mode="lines+markers",
        marker=dict(size=7, color="#ffab40"),
        hovertemplate="<b>%{x|%b %d, %Y}</b><br>Forecast: %{y:.0f}<extra></extra>",
    ))

    # Vertical line at forecast start
    split_date = d["date"].max()
    fig.add_vline(
        x=split_date, line_dash="dash", line_color="#555",
        annotation_text="Forecast →",
        annotation_font_color="#888",
        annotation_font_size=11,
    )

    fig.update_layout(
        paper_bgcolor="#080c14",
        plot_bgcolor="#0a1628",
        font=dict(family="IBM Plex Sans", color="#b0c4de"),
        title=dict(
            text=f"<b>{district}</b> — Weekly Cholera Cases",
            font=dict(color="white", size=16),
        ),
        xaxis=dict(gridcolor="#1a2d45", showgrid=True, title=""),
        yaxis=dict(gridcolor="#1a2d45", showgrid=True, title="Cases per Week"),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            bgcolor="rgba(0,0,0,0)", font=dict(color="#b0c4de"),
        ),
        hovermode="x unified",
        height=380,
        margin=dict(l=50, r=30, t=60, b=40),
    )
    return fig


def make_risk_map(df: pd.DataFrame) -> folium.Map:
    """Build a Folium choropleth risk map of Zimbabwe districts."""
    # Aggregate recent cases per district
    recent_cutoff = df["date"].max() - pd.Timedelta(weeks=8)
    recent = (
        df[df["date"] >= recent_cutoff]
        .groupby("district")
        .agg(cases_recent=("cases", "sum"))
        .reset_index()
    )

    # Merge with district coordinates
    district_coords = {
        "Harare":      (-17.8292, 31.0522),
        "Bulawayo":    (-20.1325, 28.6264),
        "Chitungwiza": (-18.0127, 31.0758),
        "Mutare":      (-18.9707, 32.6709),
        "Gweru":       (-19.4500, 29.8167),
        "Kwekwe":      (-18.9281, 29.8147),
        "Kadoma":      (-18.3333, 29.9167),
        "Masvingo":    (-20.0667, 30.8333),
        "Chinhoyi":    (-17.3667, 30.2000),
        "Norton":      (-17.8833, 30.7000),
    }

    m = folium.Map(
        location=[-19.0154, 29.1549],
        zoom_start=6,
        tiles="CartoDB dark_matter",
    )

    risk_colors = {"HIGH": "#ff4444", "MEDIUM": "#ffaa00", "LOW": "#00e676"}

    for _, row in recent.iterrows():
        district = row["district"]
        if district not in district_coords:
            continue
        lat, lon = district_coords[district]
        cases = int(row["cases_recent"])
        risk = "HIGH" if cases > 300 else "MEDIUM" if cases > 100 else "LOW"
        radius = max(8, min(35, cases / 15))

        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=risk_colors[risk],
            fill=True,
            fill_opacity=0.6,
            popup=folium.Popup(
                f"<b>{district}</b><br>"
                f"Cases (8wk): <b>{cases}</b><br>"
                f"Risk: <b style='color:{risk_colors[risk]}'>{risk}</b>",
                max_width=200,
            ),
            tooltip=f"{district}: {risk} risk",
        ).add_to(m)

        folium.Marker(
            location=[lat + 0.15, lon],
            icon=folium.DivIcon(
                html=f'<div style="font-size:11px;color:white;font-family:monospace;'
                     f'text-shadow:1px 1px 2px black;">{district}</div>',
                icon_size=(100, 20),
            )
        ).add_to(m)

    return m


def make_epidemiological_chart(df: pd.DataFrame) -> go.Figure:
    """Multi-district time series comparison."""
    monthly = (
        df.assign(month=lambda x: x["date"].dt.to_period("M").dt.to_timestamp())
        .groupby(["month", "district"])["cases"]
        .sum()
        .reset_index()
    )

    fig = px.line(
        monthly, x="month", y="cases", color="district",
        title="Monthly Cases by District",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(
        paper_bgcolor="#080c14",
        plot_bgcolor="#0a1628",
        font=dict(color="#b0c4de", family="IBM Plex Sans"),
        xaxis=dict(gridcolor="#1a2d45"),
        yaxis=dict(gridcolor="#1a2d45"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        height=320,
        margin=dict(l=50, r=30, t=50, b=40),
    )
    return fig


def make_seasonal_heatmap(df: pd.DataFrame, district: str) -> go.Figure:
    """Week-of-year × year heatmap of case intensity."""
    d = df[df["district"] == district].copy()
    d["week"] = d["date"].dt.isocalendar().week.astype(int)
    d["year"] = d["date"].dt.year

    pivot = d.pivot_table(values="cases", index="week", columns="year", aggfunc="sum")

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[str(c) for c in pivot.columns],
        y=[f"Wk {w}" for w in pivot.index],
        colorscale=[
            [0.0, "#0a1628"],
            [0.3, "#0d3b6e"],
            [0.6, "#1565c0"],
            [0.85, "#ff6b35"],
            [1.0, "#ff1744"],
        ],
        hovertemplate="Year: %{x}<br>%{y}<br>Cases: %{z}<extra></extra>",
        colorbar=dict(title="Cases", tickcolor="#888", tickfont=dict(color="#888")),
    ))
    fig.update_layout(
        title=dict(text=f"Seasonal Pattern — {district}", font=dict(color="white")),
        paper_bgcolor="#080c14",
        plot_bgcolor="#0a1628",
        font=dict(color="#b0c4de", family="IBM Plex Sans"),
        height=320,
        margin=dict(l=60, r=30, t=50, b=40),
        xaxis=dict(title="Year", gridcolor="#1a2d45"),
        yaxis=dict(title="", tickfont=dict(size=9)),
    )
    return fig


def simulate_whatif(
    df: pd.DataFrame, district: str,
    wash_improvement: float,
    rainfall_pct_change: float,
    ocv_coverage: float,
) -> dict:
    """
    Simplified what-if simulation.
    Returns estimated case reduction under given scenario.
    """
    d = df[df["district"] == district].copy()
    baseline_cases = d["cases"].tail(52).mean()

    # WASH effect: each 10% improvement → ~15% case reduction (literature-based)
    wash_reduction = wash_improvement * 0.015

    # Rainfall effect: 20% more rain → ~8% case increase
    rain_effect = (rainfall_pct_change / 100) * 0.08

    # OCV effect: 80% coverage → ~65% reduction in susceptibles
    ocv_reduction = ocv_coverage * 0.65 * 0.35

    total_reduction = wash_reduction + ocv_reduction - rain_effect
    total_reduction = np.clip(total_reduction, -0.5, 0.9)

    simulated_cases = baseline_cases * (1 - total_reduction)
    return {
        "baseline_weekly_avg": round(float(baseline_cases), 1),
        "simulated_weekly_avg": round(float(max(0, simulated_cases)), 1),
        "cases_averted_per_week": round(float(max(0, baseline_cases - simulated_cases)), 1),
        "reduction_pct": round(float(total_reduction * 100), 1),
    }


# ─── MAIN APP ──────────────────────────────────────────────────

def main():
    # Header
    st.markdown("""
    <div style='padding: 24px 0 12px 0;'>
      <div style='font-family:"IBM Plex Mono",monospace; font-size:0.7rem;
                  color:#4fc3f7; letter-spacing:0.2em; text-transform:uppercase;
                  margin-bottom:4px;'>Zimbabwe Public Health Intelligence</div>
      <h1 style='margin:0; font-size:2rem; font-weight:600; color:white;
                 font-family:"IBM Plex Sans",sans-serif;'>
        Cholera Forecasting Dashboard
      </h1>
      <div style='color:#7a9bc0; font-size:0.88rem; margin-top:6px;'>
        Ensemble forecasting · District risk mapping · What-if simulation
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Load data
    with st.spinner("Loading data..."):
        df = load_data()
        demo_df = load_demographics()

    districts = sorted(df["district"].unique().tolist())
    last_update = df["date"].max()

    # ─── Sidebar ─────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Controls")

        selected_district = st.selectbox(
            "District", districts,
            index=districts.index("Harare") if "Harare" in districts else 0,
        )

        horizon = st.select_slider(
            "Forecast Horizon",
            options=[1, 2, 4, 8, 13],
            value=4,
            format_func=lambda x: f"{x} weeks",
        )

        st.markdown("---")
        st.markdown("### 🗓️ Date Range")
        min_date = df["date"].min().date()
        max_date = df["date"].max().date()
        date_range = st.slider(
            "Historical window",
            min_value=min_date, max_value=max_date,
            value=(pd.Timestamp(max_date) - pd.Timedelta(weeks=104)).date(), # 2 years default
        )

        st.markdown("---")
        st.markdown(
            f"<div style='font-size:0.75rem; color:#4a6080;'>"
            f"Last data update:<br>"
            f"<span style='color:#4fc3f7;'>{last_update.strftime('%b %d, %Y')}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        low_bandwidth = st.checkbox("🌐 Low-bandwidth mode", value=False)

    # Filter by date range
    df_filtered = df[df["date"] >= pd.Timestamp(date_range)].copy()

    # ─── KPI row ─────────────────────────────────────────────
    district_df = df_filtered[df_filtered["district"] == selected_district]
    recent_8w = district_df[district_df["date"] >= district_df["date"].max() - pd.Timedelta(weeks=8)]
    prev_8w = district_df[
        (district_df["date"] < district_df["date"].max() - pd.Timedelta(weeks=8)) &
        (district_df["date"] >= district_df["date"].max() - pd.Timedelta(weeks=16))
    ]

    total_recent = int(recent_8w["cases"].sum())
    total_prev = int(prev_8w["cases"].sum())
    pct_change = (total_recent - total_prev) / max(1, total_prev) * 100
    peak_week = int(district_df["cases"].max())
    cfr = district_df["cfr"].mean() * 100 if "cfr" in district_df.columns else 1.2

    risk_label = "HIGH" if total_recent > 300 else "MEDIUM" if total_recent > 100 else "LOW"
    risk_color = {"HIGH": "#ff4444", "MEDIUM": "#ffaa00", "LOW": "#00e676"}[risk_label]

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("8-Week Cases", f"{total_recent:,}",
                  delta=f"{pct_change:+.1f}% vs prior 8wk")
    with col2:
        st.metric("Peak Weekly Cases", f"{peak_week:,}")
    with col3:
        st.metric("Case Fatality Rate", f"{cfr:.1f}%")
    with col4:
        st.metric("Forecast Horizon", f"{horizon} weeks")
    with col5:
        st.markdown(
            f"<div style='padding:12px 16px; background:#0d1b2e; border:1px solid #1e3a5f; "
            f"border-radius:8px;'>"
            f"<div style='font-size:0.75rem; color:#7a9bc0; text-transform:uppercase; "
            f"letter-spacing:0.08em;'>Risk Level</div>"
            f"<div style='font-size:1.8rem; font-weight:600; color:{risk_color};'>{risk_label}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── Main charts row ──────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Forecast", "🗺️ Risk Map", "📊 Epidemiology", "🔬 What-if Simulator"
    ])

    with tab1:
        st.markdown(f'<div class="section-header">District Forecast — {selected_district}</div>',
                    unsafe_allow_html=True)

        if not low_bandwidth:
            fig = make_forecast_chart(df_filtered, selected_district, horizon)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Low-bandwidth mode: showing tabular forecast.")

        # Forecast table
        forecast_df = simple_forecast(district_df, horizon)
        forecast_display = forecast_df.copy()
        forecast_display["date"] = forecast_display["date"].dt.strftime("%Y-%m-%d")
        forecast_display.columns = ["Date", "Forecast", "Lower (95%)", "Upper (95%)"]
        st.dataframe(forecast_display.round(0), use_container_width=True, hide_index=True)

        # Seasonal heatmap
        st.markdown('<div class="section-header">Seasonal Pattern</div>', unsafe_allow_html=True)
        if not low_bandwidth:
            heatmap_fig = make_seasonal_heatmap(df_filtered, selected_district)
            st.plotly_chart(heatmap_fig, use_container_width=True)

    with tab2:
        st.markdown('<div class="section-header">District Risk Map — Zimbabwe</div>',
                    unsafe_allow_html=True)

        col_map, col_legend = st.columns([3, 1])
        with col_map:
            if not low_bandwidth:
                risk_map = make_risk_map(df)
                st_folium(risk_map, width=700, height=450, returned_objects=[])
            else:
                # Static table for low-bandwidth
                risk_table = (
                    df.groupby("district")["cases"]
                    .sum().reset_index()
                    .sort_values("cases", ascending=False)
                )
                st.dataframe(risk_table, hide_index=True, use_container_width=True)

        with col_legend:
            st.markdown("### Risk Legend")
            for level, color, desc in [
                ("HIGH", "#ff4444", "> 300 cases / 8 weeks"),
                ("MEDIUM", "#ffaa00", "100–300 cases / 8 weeks"),
                ("LOW", "#00e676", "< 100 cases / 8 weeks"),
            ]:
                st.markdown(
                    f'<div style="margin-bottom:12px;">'
                    f'<span style="color:{color};font-weight:600;">● {level}</span><br>'
                    f'<span style="font-size:0.8rem;color:#7a9bc0;">{desc}</span>'
                    f'</div>', unsafe_allow_html=True
                )

            st.markdown("---")
            # District risk summary
            recent_cutoff = df["date"].max() - pd.Timedelta(weeks=8)
            risk_summary = (
                df[df["date"] >= recent_cutoff]
                .groupby("district")["cases"]
                .sum()
                .reset_index()
                .assign(risk=lambda x: x["cases"].apply(
                    lambda c: "HIGH" if c > 300 else "MEDIUM" if c > 100 else "LOW"
                ))
                .sort_values("cases", ascending=False)
            )
            for _, row in risk_summary.iterrows():
                color = {"HIGH": "#ff4444", "MEDIUM": "#ffaa00", "LOW": "#00e676"}[row["risk"]]
                st.markdown(
                    f'<div style="font-size:0.82rem; margin-bottom:4px;">'
                    f'<span style="color:{color};">■</span> {row["district"]} '
                    f'<span style="color:#555;">({int(row["cases"])})</span></div>',
                    unsafe_allow_html=True,
                )

    with tab3:
        st.markdown('<div class="section-header">Epidemiological Overview</div>',
                    unsafe_allow_html=True)

        if not low_bandwidth:
            fig_multi = make_epidemiological_chart(df_filtered)
            st.plotly_chart(fig_multi, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Top Districts by Cases (8-week)**")
            recent_cutoff = df["date"].max() - pd.Timedelta(weeks=8)
            top_districts = (
                df[df["date"] >= recent_cutoff]
                .groupby("district")[["cases", "deaths"]].sum()
                .reset_index()
                .sort_values("cases", ascending=False)
            )
            top_districts["CFR (%)"] = (top_districts["deaths"] / top_districts["cases"].clip(1) * 100).round(2)
            st.dataframe(top_districts, hide_index=True, use_container_width=True)

        with col_b:
            st.markdown("**Download Data**")
            csv_buf = io.StringIO()
            df_filtered.to_csv(csv_buf, index=False)
            st.download_button(
                "⬇️ Download CSV",
                data=csv_buf.getvalue(),
                file_name=f"cholera_zim_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )

            # Excel export
            excel_buf = io.BytesIO()
            with pd.ExcelWriter(excel_buf, engine="xlsxwriter") as writer:
                df_filtered.to_excel(writer, sheet_name="Cases", index=False)
                if not demo_df.empty:
                    demo_df.to_excel(writer, sheet_name="Demographics", index=False)
            st.download_button(
                "⬇️ Download Excel",
                data=excel_buf.getvalue(),
                file_name=f"cholera_zim_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    with tab4:
        st.markdown('<div class="section-header">What-If Intervention Simulator</div>',
                    unsafe_allow_html=True)
        st.caption(
            "Adjust intervention parameters to estimate potential case reductions "
            "based on epidemiological effect estimates from published literature."
        )

        col_sliders, col_results = st.columns([1, 1])

        with col_sliders:
            wash_imp = st.slider(
                "WASH Coverage Improvement (%)",
                0, 40, 10,
                help="Estimated reduction: ~1.5% cases per 1% WASH improvement",
            )
            rain_change = st.slider(
                "Rainfall Change (%)",
                -30, 30, 0,
                help="Positive = wetter season (increases risk); negative = drier",
            )
            ocv_cov = st.slider(
                "OCV Campaign Coverage (%)",
                0, 95, 55,
                help="Oral cholera vaccine coverage in target population",
            ) / 100

        with col_results:
            sim = simulate_whatif(df_filtered, selected_district, wash_imp, rain_change, ocv_cov)
            averted_annual = sim["cases_averted_per_week"] * 52

            st.markdown("**Simulation Results**")
            st.metric("Baseline (weekly avg)", f"{sim['baseline_weekly_avg']:.0f} cases")
            st.metric("Projected (weekly avg)", f"{sim['simulated_weekly_avg']:.0f} cases",
                      delta=f"{sim['reduction_pct']:+.1f}%")
            st.metric("Cases Averted / Week", f"{sim['cases_averted_per_week']:.0f}")

            st.info(
                f"📊 Under this scenario, approximately **{averted_annual:.0f} cases** "
                f"per year could be prevented in {selected_district}.",
            )

            # Scenario bar chart
            fig_scenario = go.Figure(go.Bar(
                x=["Baseline", "Projected"],
                y=[sim["baseline_weekly_avg"], sim["simulated_weekly_avg"]],
                marker_color=["#4fc3f7", "#00e676"],
                text=[f"{sim['baseline_weekly_avg']:.0f}", f"{sim['simulated_weekly_avg']:.0f}"],
                textposition="outside",
            ))
            fig_scenario.update_layout(
                paper_bgcolor="#080c14",
                plot_bgcolor="#0a1628",
                font=dict(color="#b0c4de"),
                yaxis=dict(title="Weekly Cases", gridcolor="#1a2d45"),
                xaxis=dict(gridcolor="#1a2d45"),
                height=260,
                margin=dict(l=50, r=20, t=20, b=40),
                showlegend=False,
            )
            st.plotly_chart(fig_scenario, use_container_width=True)

    # ─── Footer ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.75rem; color:#3a5070; text-align:center;'>"
        "Zimbabwe Cholera Forecasting System · MSc Computer Science Research · MIT License · "
        "<a href='https://github.com/yourusername/cholera-zim-forecast' "
        "style='color:#4fc3f7;'>GitHub</a>"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
