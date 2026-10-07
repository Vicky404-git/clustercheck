"""Offline dataset sources (Person B): generated demo series + Kaggle CSV.

Both feed the pipeline without network access or hardware access, so they
live together:

- demo   : deterministic fake data with a distinct behaviour per metric
- kaggle : vmCloud_data.csv from the Kaggle VM Cloud dataset

Every series uses the standard pipeline shape:

    [{"timestamp": datetime, "value": float}, ...]
"""

import math
import os
import random
from datetime import datetime, timedelta, timezone
from typing import Dict, List

import pandas as pd

# ======================================================================
# demo - deterministic fake metrics
# ======================================================================
# - CPU    : mostly idle, random periodic spikes with decay
# - RAM    : slow upward leak across the window
# - Disk   : nearly flat with a slow creep
# - Network: bursty and unpredictable (different profiles for in/out)

DEMO_METRICS = ["cpu", "ram", "disk", "network_in", "network_out"]


def get_demo_metrics(
    hours: int = 6,
    period: int = 300,
    seed: int = 42,
) -> Dict[str, List[dict]]:
    """Return a full set of demo series sharing the same timestamps."""
    period = max(60, int(period))
    total = max(6, int((hours * 3600) / period))
    rng = random.Random(seed)

    start = datetime.now(timezone.utc) - timedelta(seconds=total * period)
    timestamps = [start + timedelta(seconds=i * period) for i in range(total)]

    raw = {
        "cpu": _cpu(rng, total),
        "ram": _ram(rng, total),
        "disk": _disk(rng, total),
        "network_in": _network(rng, total, base=45_000, burst_p=0.12, burst=(4.0, 14.0)),
        "network_out": _network(rng, total, base=28_000, burst_p=0.08, burst=(3.0, 10.0)),
    }

    return {
        name: [{"timestamp": ts, "value": value} for ts, value in zip(timestamps, values)]
        for name, values in raw.items()
    }


def _cpu(rng: random.Random, total: int) -> List[float]:
    spike_count = max(1, total // 8)
    spikes = set(rng.sample(range(2, total), spike_count))
    values = []
    decay = 0.0
    for i in range(total):
        base = 11 + 5 * math.sin(2 * math.pi * 2 * i / total) + rng.gauss(0, 2)
        if i in spikes:
            decay = rng.uniform(38, 75)
        elif decay > 0.5:
            decay *= 0.45
        else:
            decay = 0.0
        values.append(round(min(99.0, max(0.5, base + decay)), 2))
    return values


def _ram(rng: random.Random, total: int) -> List[float]:
    start = rng.uniform(30, 40)
    leak = rng.uniform(22, 35)
    values = []
    for i in range(total):
        frac = i / (total - 1) if total > 1 else 1.0
        value = start + leak * frac + rng.gauss(0, 1.0)
        values.append(round(min(96.0, max(1.0, value)), 2))
    return values


def _disk(rng: random.Random, total: int) -> List[float]:
    values = []
    for i in range(total):
        frac = i / (total - 1) if total > 1 else 1.0
        value = 38 + 4 * frac + rng.gauss(0, 0.4)
        values.append(round(min(97.0, max(1.0, value)), 2))
    return values


def _network(
    rng: random.Random,
    total: int,
    base: float,
    burst_p: float,
    burst: tuple,
) -> List[int]:
    values = []
    remaining = 0
    multiplier = 1.0
    for _ in range(total):
        value = base * (1 + rng.gauss(0, 0.2))
        if remaining == 0 and rng.random() < burst_p:
            remaining = rng.randint(2, 5)
            multiplier = rng.uniform(*burst)
        if remaining > 0:
            value += base * multiplier * rng.uniform(0.6, 1.3)
            remaining -= 1
        values.append(int(max(0, value)))
    return values


# ======================================================================
# kaggle - vmCloud_data.csv (Kaggle VM Cloud dataset)
# ======================================================================
# The dataset is large (~2M rows), so only the two needed columns are parsed,
# rows with missing/unparseable values are dropped, and the result is sorted
# chronologically before applying `limit` (most recent rows kept).

DEFAULT_CSV_PATH = "vmCloud_data.csv"
TIMESTAMP_COLUMN = "timestamp"

# pipeline metric name -> CSV column name (units live in METRIC_SPECS,
# backend/ingestion/__init__.py)
KAGGLE_METRICS: Dict[str, str] = {
    "cpu": "cpu_usage",
    "ram": "memory_usage",
    "network": "network_traffic",
    "power": "power_consumption",
    "instructions": "num_executed_instructions",
    "execution_time": "execution_time",
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

    header = _read_csv_header(csv_path)
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


def _read_csv_header(csv_path: str) -> List[str]:
    try:
        return list(pd.read_csv(csv_path, nrows=0).columns)
    except Exception as exc:  # noqa: BLE001 - empty/corrupt file etc.
        raise RuntimeError(f"Could not read {csv_path}: {exc}") from exc
