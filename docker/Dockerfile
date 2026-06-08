FROM python:3.10-slim

LABEL maintainer="MSc CS Research"
LABEL description="Zimbabwe Cholera Forecasting System"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential libpq-dev curl git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (CPU-only torch)
COPY requirements.txt .
RUN pip install --no-cache-dir numpy pandas scipy scikit-learn \
    xgboost prophet shap statsmodels \
    streamlit fastapi uvicorn httpx sqlalchemy \
    plotly folium pydantic requests tqdm pyyaml \
    pytest pytest-cov

# Copy project
COPY . .

# Generate synthetic data
RUN python data/synthetic/generate_synthetic.py --output-dir data/processed

EXPOSE 8501 8000

CMD ["streamlit", "run", "src/app/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true"]
