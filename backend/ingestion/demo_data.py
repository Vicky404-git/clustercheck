"""Deterministic fake metric data with a distinct behaviour per metric (Person B).

- CPU    : mostly idle, random periodic spikes with decay
- RAM    : slow upward leak across the window
- Disk   : nearly flat with a slow creep
- Network: bursty and unpredictable (different profiles for in/out)
"""

import math
import random
from datetime import datetime, timedelta, timezone
from typing import Dict, List

Point = Dict[str, object]

DEMO_METRICS = ["cpu", "ram", "disk", "network_in", "network_out"]


def get_demo_metrics(
    hours: int = 6,
    period: int = 300,
    seed: int = 42,
) -> Dict[str, List[Point]]:
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
