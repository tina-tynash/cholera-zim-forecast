"""metrics.py — Evaluation metrics and statistical utilities."""

import numpy as np
import pandas as pd
from typing import Dict

def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    a, p = np.asarray(actual, float), np.asarray(predicted, float)
    return float(np.mean(np.abs(a - p) / (a + 1)) * 100)

def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(predicted)) ** 2)))

def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(predicted))))

def all_metrics(actual: np.ndarray, predicted: np.ndarray, model: str = "") -> Dict:
    mask = ~np.isnan(predicted)
    a, p = np.asarray(actual)[mask], np.asarray(predicted)[mask]
    return {"model": model, "mape": round(mape(a, p), 3),
            "rmse": round(rmse(a, p), 3), "mae": round(mae(a, p), 3)}
