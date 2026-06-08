#!/bin/bash
set -e

echo "=== Zimbabwe Cholera Forecasting Setup ==="

if ! command -v python3 &> /dev/null; then
    echo "Python 3 not found. Please install Python 3.10+"
    exit 1
fi

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Generating synthetic datasets..."
python data/synthetic/generate_synthetic.py

echo "Running tests..."
pytest tests/ -q --tb=short

echo ""
echo "Setup complete!"
echo ""
echo "Start the dashboard:  streamlit run src/app/streamlit_app.py"
echo "Start the API:        uvicorn src.app.api:app --reload --port 8000"
echo "Train models:         python -m src.models.train_ensemble"
