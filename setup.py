"""
setup.py — Package installation for cholera-zim-forecast.

This allows the project to be installed as a package:
    pip install -e .

Which enables `from src.models.ensemble import EnsembleForecaster` from any directory.
"""
from setuptools import setup, find_packages

setup(
    name="cholera-zim-forecast",
    version="1.0.0",
    description="Cholera forecasting system for Zimbabwe — ML + epidemiology",
    author="Your Name",
    license="MIT",
    packages=find_packages(exclude=["tests*", "notebooks*"]),
    python_requires=">=3.10",
    install_requires=[
        "pandas>=2.1",
        "numpy>=1.26",
        "scikit-learn>=1.4",
        "xgboost>=2.0",
        "shap>=0.44",
        "sqlalchemy>=2.0",
        "fastapi>=0.109",
        "uvicorn>=0.27",
        "streamlit>=1.31",
        "plotly>=5.18",
        "folium>=0.15",
        "pydantic>=2.5",
        "requests>=2.31",
        "pyyaml>=6.0",
        "loguru>=0.7",
    ],
    extras_require={
        "dev": ["pytest", "pytest-cov", "flake8", "black", "httpx"],
        "deep": ["torch>=2.1"],
        "causal": ["dowhy>=0.11"],
    },
    entry_points={
        "console_scripts": [
            "cholera-dashboard=src.app.streamlit_app:main",
            "cholera-api=src.app.api:app",
            "cholera-train=src.models.train_ensemble:main",
        ]
    },
)
