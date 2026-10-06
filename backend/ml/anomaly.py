"""Anomaly detection using scikit-learn Isolation Forest (Person C).

The model is trained in-memory on every call and thrown away afterwards.
No joblib, no model file on disk - keeps the mini project simple.
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

MIN_POINTS_TO_TRAIN = 20


def detect_anomalies(points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flag unusual values in a metric series. Returns copies of the input dicts."""
    if len(points) < MIN_POINTS_TO_TRAIN:
        return [dict(p, is_anomaly=False, score=0.0) for p in points]

    df = pd.DataFrame(points)
    values = df["value"].to_numpy()

    # a perfectly flat series (e.g. idle disk) has nothing to flag
    if float(np.ptp(values)) == 0.0:
        return [dict(p, is_anomaly=False, score=0.0) for p in points]

    model = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=42,
    )
    labels = model.fit_predict(values.reshape(-1, 1))
    scores = model.decision_function(values.reshape(-1, 1))

    return [
        dict(point, is_anomaly=bool(label == -1), score=round(float(score), 4))
        for point, label, score in zip(points, labels, scores)
    ]


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    values = np.array([r["value"] for r in results], dtype=float)
    anomalies = [r for r in results if r["is_anomaly"]]
    return {
        "points": len(results),
        "anomalies": len(anomalies),
        "min": round(float(values.min()), 2) if len(values) else 0.0,
        "max": round(float(values.max()), 2) if len(values) else 0.0,
        "avg": round(float(values.mean()), 2) if len(values) else 0.0,
    }
