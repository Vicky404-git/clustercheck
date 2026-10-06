"""Kaggle vmCloud dataset ingestion (Person B).

Reads ``vmCloud_data.csv`` (Kaggle VM Cloud dataset) and exposes three of its
columns through the standard pipeline shape:

    [{"timestamp": datetime, "value": float}, ...]

The dataset is large (~2M rows), so only the two needed columns are parsed,
rows with missing/unparseable values are dropped, and the result is sorted
chronologically before applying ``limit`` (most recent rows kept).
"""

import os
from typing import Dict, List

import pandas as pd

DEFAULT_CSV_PATH = "vmCloud_data.csv"
TIMESTAMP_COLUMN = "timestamp"

# pipeline metric name -> CSV column name
KAGGLE_METRICS: Dict[str, str] = {
    "cpu": "cpu_usage",
    "ram": "memory_usage",
    "network": "network_traffic",
}


def fetch_kaggle_series(
    csv_path: str = DEFAULT_CSV_PATH,
    metric: str = "cpu",
    limit: int = 100,
) -> List[dict]:
    """Read one metric from the Kaggle CSV, cleaned and sorted by time."""
    if metric not in KAGGLE_METRICS:
        raise RuntimeError(
            f"Unsupported Kaggle metric '{metric}'. "
            f"Choose from: {', '.join(KAGGLE_METRICS)}"
        )
    column = KAGGLE_METRICS[metric]

    if not os.path.isfile(csv_path):
        raise RuntimeError(
            f"CSV file not found: {csv_path} "
            "(place vmCloud_data.csv in the project root or pass --csv)"
        )

    header = _read_header(csv_path)
    missing = [name for name in (TIMESTAMP_COLUMN, column) if name not in header]
    if missing:
        raise RuntimeError(
            f"{csv_path} is missing column(s): {', '.join(missing)}. "
            f"Found: {', '.join(header)}"
        )

    try:
        frame = pd.read_csv(csv_path, usecols=[TIMESTAMP_COLUMN, column])
    except Exception as exc:  # noqa: BLE001 - surface any parser error cleanly
        raise RuntimeError(f"Could not read {csv_path}: {exc}") from exc

    frame[TIMESTAMP_COLUMN] = pd.to_datetime(frame[TIMESTAMP_COLUMN], errors="coerce")
    frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=[TIMESTAMP_COLUMN, column])

    if frame.empty:
        raise RuntimeError(
            f"No usable rows for metric '{metric}' in {csv_path} "
            "(values missing or timestamps unparseable)"
        )

    frame = frame.sort_values(TIMESTAMP_COLUMN, kind="stable")
    if limit and int(limit) > 0:
        frame = frame.tail(int(limit))

    timestamps = frame[TIMESTAMP_COLUMN].tolist()
    values = frame[column].tolist()
    return [
        {
            "timestamp": ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
            "value": float(value),
        }
        for ts, value in zip(timestamps, values)
    ]


def _read_header(csv_path: str) -> List[str]:
    try:
        return list(pd.read_csv(csv_path, nrows=0).columns)
    except Exception as exc:  # noqa: BLE001 - empty/corrupt file etc.
        raise RuntimeError(f"Could not read {csv_path}: {exc}") from exc
