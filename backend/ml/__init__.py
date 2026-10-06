"""ML engine (Person C).

- anomaly detection: scikit-learn Isolation Forest, trained in-memory
  per session (no joblib, no model file written to disk)
- trend prediction: linear regression estimating time until 90% utilisation
"""

from .anomaly import MIN_POINTS_TO_TRAIN, detect_anomalies, summarize
from .predictor import format_minutes, predict_utilization_trend

__all__ = [
    "MIN_POINTS_TO_TRAIN",
    "detect_anomalies",
    "summarize",
    "predict_utilization_trend",
    "format_minutes",
]
