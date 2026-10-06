"""Simple trend prediction (Person C).

Fits an ordinary least squares line (scikit-learn LinearRegression) over the
recent history of a utilisation metric and estimates when it will cross a
threshold (default 90%). Also trained in-memory, nothing is persisted.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.linear_model import LinearRegression

MIN_POINTS_TO_PREDICT = 10


def predict_utilization_trend(
    points: List[Dict[str, Any]],
    threshold: float = 90.0,
    metric: str = "cpu",
    min_points: int = MIN_POINTS_TO_PREDICT,
) -> Dict[str, Any]:
    """Linear-regression trend for a 0-100 utilisation metric.

    Returns a dict with slope/eta fields; `supported` is False when there is
    not enough data to fit a line.
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
