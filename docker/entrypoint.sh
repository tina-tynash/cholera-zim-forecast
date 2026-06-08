#!/bin/bash
set -e
uvicorn src.app.api:app --host 0.0.0.0 --port 8000 &
streamlit run src/app/streamlit_app.py --server.address 0.0.0.0 --server.port 8501
