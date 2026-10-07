"""Data ingestion layer  + the shared metric registry.

Five interchangeable sources, all returning the same series shape:

    {"timestamp": datetime, "value": float}

- local  : psutil readings from this machine (local_pc.py, which also hosts
           the HTTP agent used by the remote source)
- remote : HTTP polling of another machine running that agent
- aws    : CloudWatch metrics via boto3 (aws_ingestion.py)
- demo   : deterministic fake data (datasets.py)
- kaggle : vmCloud_data.csv from the Kaggle dataset (datasets.py)

METRIC_SPECS below is the single source of truth for every metric's human
name and unit. app.py, debug.py and the ML predictor all read it from here,
so units are never duplicated or forgotten.
"""

from .aws_ingestion import AWS_METRIC_NAMES, fetch_metric, list_instances
from .datasets import (
    DEFAULT_CSV_PATH,
    DEMO_METRICS,
    KAGGLE_METRICS,
    fetch_kaggle_series,
    get_demo_metrics,
)
from .local_pc import (
    BASE_METRICS,
    available_local_metrics,
    collect_local_metrics,
    fetch_remote_metrics,
    list_remote_metrics,
    snapshot_local_metrics,
)

SOURCES = ["local", "aws", "demo", "remote", "kaggle"]

# metric key -> (human name, unit)
METRIC_SPECS = {
    # percentage utilisation - the unit "%" is also what marks a metric as
    # eligible for the 90% trend prediction (see PCT_METRICS)
    "cpu": ("CPU Utilization", "%"),
    "ram": ("Memory Usage", "%"),
    "disk": ("Disk Usage", "%"),
    "gpu": ("GPU Utilization", "%"),
    "CPUUtilization": ("CPUUtilization", "%"),
    # rates and counters
    "network": ("Network Traffic", "bytes/sec"),
    "network_in": ("Network In", "bytes/sec"),
    "network_out": ("Network Out", "bytes/sec"),
    "NetworkIn": ("NetworkIn", "bytes"),
    "NetworkOut": ("NetworkOut", "bytes"),
    # Kaggle VM Cloud raw metrics (absolute, non-percentage)
    "power": ("Power Consumption", "W"),
    "instructions": ("Executed Instructions", "ops"),
    "execution_time": ("Execution Time", "ms"),
}

# percentage metrics drive the 90% utilisation prediction; everything else
# (W, ops, ms, bytes/sec) is bypassed there but still gets anomaly detection
PCT_METRICS = {key for key, (_name, unit) in METRIC_SPECS.items() if unit == "%"}


def metric_unit(metric: str) -> str:
    return METRIC_SPECS.get(metric, (metric, ""))[1]


def metric_label(metric: str) -> str:
    """'power' -> 'Power Consumption (W)'; unknown keys pass through."""
    name, unit = METRIC_SPECS.get(metric, (metric, ""))
    return f"{name} ({unit})" if unit else name


def format_value(value, metric: str = None) -> str:
    """Display a value at full useful precision with its unit.

    Integers keep thousands separators and no decimals ('7,527 ops'),
    fractional values keep up to 6 decimals and never go scientific
    ('403.044812 W', '0.064105 ms') - so absolute metrics are never
    rounded away to look like percentages.
    """
    if value is None:
        return "n/a"
    number = float(value)
    if number == int(number) and abs(number) < 1e15:
        text = f"{int(number):,}"
    else:
        text = f"{number:,.6f}".rstrip("0").rstrip(".")
        if text in {"", ".", "-", "-."}:
            text = repr(number)
    if metric is None:
        return text
    unit = metric_unit(metric)
    return f"{text} {unit}".rstrip()


def list_metrics(source: str) -> list:
    if source == "aws":
        return list(AWS_METRIC_NAMES)
    if source == "demo":
        return list(DEMO_METRICS)
    if source == "local":
        return available_local_metrics()
    if source == "remote":
        return list(BASE_METRICS)
    if source == "kaggle":
        return list(KAGGLE_METRICS)
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

    if source == "kaggle":
        return fetch_kaggle_series(
            csv_path=opts.get("csv_path", DEFAULT_CSV_PATH),
            metric=metric,
            limit=int(opts.get("limit", 100)),
        )

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
