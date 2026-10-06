"""Data ingestion layer (Person B).

Four interchangeable sources, all returning the same series shape:

    {"timestamp": datetime, "value": float}

- local  : psutil readings from this machine
- remote : HTTP polling of another machine running backend.ingestion.agent
- aws    : CloudWatch metrics via boto3
- demo   : deterministic fake data (see demo_data.py)
"""

from .aws_ingestion import AWS_METRIC_NAMES, fetch_metric, list_instances
from .demo_data import DEMO_METRICS, get_demo_metrics
from .local_pc import (
    BASE_METRICS,
    available_local_metrics,
    collect_local_metrics,
    fetch_remote_metrics,
    list_remote_metrics,
    snapshot_local_metrics,
)

SOURCES = ["local", "aws", "demo", "remote"]

METRIC_LABELS = {
    "cpu": "CPU Utilization (%)",
    "ram": "Memory Usage (%)",
    "disk": "Disk Usage (%)",
    "gpu": "GPU Utilization (%)",
    "network_in": "Network In (bytes/sec)",
    "network_out": "Network Out (bytes/sec)",
    "CPUUtilization": "CPUUtilization (%)",
    "NetworkIn": "NetworkIn (bytes)",
    "NetworkOut": "NetworkOut (bytes)",
}

# metrics measured in percent, used for the 90% utilisation prediction
PCT_METRICS = {"cpu", "ram", "disk", "gpu", "CPUUtilization"}


def metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric)


def list_metrics(source: str) -> list:
    if source == "aws":
        return list(AWS_METRIC_NAMES)
    if source == "demo":
        return list(DEMO_METRICS)
    if source == "local":
        return available_local_metrics()
    if source == "remote":
        return list(BASE_METRICS)
    raise ValueError(f"Unknown source: {source}")


def fetch_series(source: str, metric: str, **opts) -> list:
    """Fetch one metric series from any source. Raises RuntimeError on failure."""
    if source == "aws":
        instance_id = opts.get("instance_id") or opts.get("instance") or ""
        if not instance_id:
            raise RuntimeError("instance_id is required for the AWS source")
        return fetch_metric(
            instance_id,
            metric,
            opts.get("region", "ap-south-1"),
            hours=int(opts.get("hours", 6)),
            period=int(opts.get("period", 300)),
        )

    if source == "demo":
        data = get_demo_metrics(
            hours=int(opts.get("hours", 6)),
            period=int(opts.get("period", 300)),
            seed=int(opts.get("seed", 42)),
        )
        return _pick(data, source, metric)

    if source == "local":
        data = collect_local_metrics(
            duration=float(opts.get("duration", 30)),
            interval=float(opts.get("interval", 0.5)),
        )
        return _pick(data, source, metric)

    if source == "remote":
        data = fetch_remote_metrics(
            opts.get("host", "127.0.0.1"),
            int(opts.get("port", 8777)),
            duration=float(opts.get("duration", 30)),
            interval=float(opts.get("interval", 0.5)),
        )
        return _pick(data, source, metric)

    raise ValueError(f"Unknown source: {source}")


def _pick(data: dict, source: str, metric: str) -> list:
    if metric not in data:
        available = sorted(k for k, v in data.items() if v)
        raise RuntimeError(
            f"Metric '{metric}' not available for source '{source}'. "
            f"Available: {', '.join(available) or 'none'}"
        )
    return data[metric]
