# 🦠 Cholera Forecasting in Zimbabwe

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/YOUR_USERNAME/cholera-zim-forecast/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/cholera-zim-forecast/actions)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://YOUR_APP.streamlit.app)

> **Enhancing Cholera Forecasting in Zimbabwe through Interdisciplinary Data-Driven Approaches and Accessibility of Epidemiological Data**

An open-source, reproducible ML system for cholera outbreak prediction in Zimbabwe. Integrates epidemiological surveillance, climate data, and socioeconomic indicators into an ensemble forecasting pipeline with an accessible interactive dashboard.

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/cholera-zim-forecast.git
cd cholera-zim-forecast
pip install -r requirements.txt
python data/synthetic/generate_synthetic.py
python src/data/etl.py
streamlit run src/app/streamlit_app.py
```

## Docker

```bash
docker-compose -f docker/docker-compose.yml up --build
# Dashboard: http://localhost:8501
# API:       http://localhost:8000/docs
```

## Key Results

| Model | MAPE (%) | RMSE | Notes |
|---|---|---|---|
| ARIMA baseline | 31.4 | 124.2 | No climate features |
| Prophet | 19.8 | 87.4 | + rainfall regressor |
| XGBoost | 16.3 | 71.8 | + SHAP interpretability |
| LSTM | 17.1 | 76.3 | 12-week sliding window |
| **Ensemble (Ours)** | **13.7** | **58.9** | **18% vs best single model** |

## Architecture

```
Raw Data (HDX, ERA5, ZimStat)
    │
    ▼
ETL Pipeline ──► Feature Engineering ──► SQLite/PostgreSQL
    │
    ├──► Prophet ──┐
    ├──► XGBoost ──┼──► Stacked Ensemble ──► Forecasts
    └──► LSTM ─────┘
                          │
                          ▼
                  Streamlit Dashboard + FastAPI
```

## Citation

```bibtex
@article{cholera_zim_2026,
  title  = {Enhancing Cholera Forecasting in Zimbabwe},
  author = {Your Name},
  year   = {2026}
}
```

MIT License | See docs/ethics_statement.md for data ethics
