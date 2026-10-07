"""ML engine .

Two in-memory scikit-learn models, nothing persisted (no joblib, no model
file on disk):

- anomaly detection: Isolation Forest flagging unusual values in any series
- trend prediction: linear regression estimating when a *percentage*
  utilisation metric will cross 90%

The percentage-metric registry lives in the ingestion layer (single source
of truth for metric names/units) and is what lets the predictor safely
bypass absolute metrics such as Watts, ops or ms.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression

from .ingestion import PCT_METRICS

MIN_POINTS_TO_TRAIN = 20
MIN_POINTS_TO_PREDICT = 10


# ======================================================================
# anomaly detection - Isolation Forest
# ======================================================================
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
    """Min/avg/max stats for any metric. Values are kept at full precision;
    the UI/CLI decide how to round for display."""
    values = np.array([r["value"] for r in results], dtype=float)
    anomalies = [r for r in results if r["is_anomaly"]]
    return {
        "points": len(results),
        "anomalies": len(anomalies),
        "min": float(values.min()) if len(values) else 0.0,
        "max": float(values.max()) if len(values) else 0.0,
        "avg": float(values.mean()) if len(values) else 0.0,
    }


# ======================================================================
# trend prediction - linear regression
# ======================================================================
def predict_utilization_trend(
    points: List[Dict[str, Any]],
    threshold: float = 90.0,
    metric: str = "cpu",
    min_points: int = MIN_POINTS_TO_PREDICT,
) -> Dict[str, Any]:
    """Linear-regression trend for a 0-100 utilisation metric.

    Returns a dict with slope/eta fields; `supported` is False when the metric
    is not a percentage (e.g. power in W) or there is not enough data.
    """
    result: Dict[str, Any] = {
        "supported": False,
        "reason": None,
        "metric": metric,
        "threshold": threshold,
        "current": None,
        "direction": None,
        "slope_per_min": None,
        "intercept": None,
        "r2": None,
        "minutes_to_threshold": None,
        "eta": None,
    }

    if metric not in PCT_METRICS:
        result["reason"] = (
            f"'{metric}' is not a percentage metric - trend prediction "
            "only applies to 0-100% utilisation (CPU/RAM/Disk/GPU)"
        )
        return result

    if len(points) < min_points:
        result["reason"] = f"need at least {min_points} points, got {len(points)}"
        return result

    timestamps = [p["timestamp"] for p in points]
    t0 = timestamps[0]
    x = np.array([(ts - t0).total_seconds() / 60.0 for ts in timestamps])
    y = np.array([float(p["value"]) for p in points], dtype=float)

    if x[-1] <= 0:
        result["reason"] = "timestamps are not increasing"
        return result

    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)

    slope = float(model.coef_[0])
    intercept = float(model.intercept_)
    current = float(y[-1])

    if slope > 1e-9:
        direction = "rising"
    elif slope < -1e-9:
        direction = "falling"
    else:
        direction = "flat"

    minutes: Optional[float] = None
    eta: Optional[datetime] = None
    reason = None

    if current >= threshold:
        minutes = 0.0
        eta = timestamps[-1]
        reason = "current value is already at or above the threshold"
    elif slope > 0:
        minutes = (threshold - current) / slope
        eta = timestamps[-1] + timedelta(minutes=minutes)

    result.update(
        {
            "supported": True,
            "reason": reason,
            "current": round(current, 2),
            "direction": direction,
            "slope_per_min": round(slope, 4),
            "intercept": intercept,
            "r2": round(float(model.score(x.reshape(-1, 1), y)), 3),
            "minutes_to_threshold": round(minutes, 1) if minutes is not None else None,
            "eta": eta,
        }
    )
    return result


def format_minutes(minutes: Optional[float]) -> str:
    """125.5 -> '2h 5m'"""
    if minutes is None:
        return "n/a"
    minutes = float(minutes)
    if minutes < 1:
        return "under a minute"
    if minutes < 60:
        return f"{int(minutes)}m"
    total = int(round(minutes))
    days, rem = divmod(total, 60 * 24)
    hours, mins = divmod(rem, 60)
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {mins}m"
    return f"{mins}m"
