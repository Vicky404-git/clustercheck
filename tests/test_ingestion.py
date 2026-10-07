import threading

import pytest

from backend.ingestion import (
    PCT_METRICS,
    fetch_series,
    format_value,
    list_metrics,
    metric_label,
    metric_unit,
)
from backend.ingestion.aws_ingestion import fetch_metric
from backend.ingestion.local_pc import (
    collect_local_metrics,
    fetch_remote_metrics,
    list_remote_metrics,
    make_server,
    snapshot_local_metrics,
)


def test_local_snapshot_has_core_metrics():
    snap = snapshot_local_metrics()
    for key in ("cpu", "ram", "disk"):
        assert isinstance(snap[key], float)
        assert 0 <= snap[key] <= 100
    assert "hostname" in snap
    assert snap["gpu"] is None or 0 <= snap["gpu"] <= 100


def test_local_collection_builds_series():
    series = collect_local_metrics(duration=2, interval=0.5)
    assert len(series["cpu"]) >= 3
    assert "ram" in series and "disk" in series
    assert "network_in" in series
    assert series["cpu"][0]["value"] >= 0
    if "gpu" in series:
        assert 0 <= series["gpu"][0]["value"] <= 100


def test_fetch_series_local_via_dispatcher():
    points = fetch_series("local", "ram", duration=2, interval=0.5)
    assert len(points) >= 3
    assert {"timestamp", "value"} <= set(points[0].keys())


def test_remote_agent_roundtrip():
    server = make_server(host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        remote_metrics = list_remote_metrics("127.0.0.1", port)
        assert "cpu" in remote_metrics and "ram" in remote_metrics
        series = fetch_remote_metrics("127.0.0.1", port, duration=2, interval=0.5)
        assert len(series["cpu"]) >= 3
        assert "ram" in series
    finally:
        server.shutdown()
        server.server_close()


def test_remote_unreachable_gives_friendly_error():
    with pytest.raises(RuntimeError, match="Cannot reach agent"):
        fetch_remote_metrics("127.0.0.1", 1, duration=1, interval=0.5)


def test_aws_rejects_unknown_metric():
    with pytest.raises(RuntimeError, match="Unsupported metric"):
        fetch_metric("i-123", "BogusMetric", "ap-south-1")


def test_aws_requires_instance_id():
    with pytest.raises(RuntimeError, match="instance_id is required"):
        fetch_series("aws", "CPUUtilization")


def test_list_metrics_per_source():
    assert list_metrics("aws") == ["CPUUtilization", "NetworkIn", "NetworkOut"]
    assert list_metrics("demo") == ["cpu", "ram", "disk", "network_in", "network_out"]
    local = list_metrics("local")
    assert "cpu" in local and "ram" in local and "disk" in local


# ------------------------------------------------------- metric unit registry
def test_metric_labels_carry_their_units():
    assert metric_label("cpu") == "CPU Utilization (%)"
    assert metric_label("power") == "Power Consumption (W)"
    assert metric_label("instructions") == "Executed Instructions (ops)"
    assert metric_label("execution_time") == "Execution Time (ms)"
    assert metric_label("network") == "Network Traffic (bytes/sec)"
    assert metric_label("something_unknown") == "something_unknown"


def test_metric_units():
    assert metric_unit("cpu") == "%"
    assert metric_unit("power") == "W"
    assert metric_unit("instructions") == "ops"
    assert metric_unit("execution_time") == "ms"
    assert metric_unit("network_in") == "bytes/sec"


def test_percentage_metrics_are_exactly_the_percent_ones():
    assert PCT_METRICS == {"cpu", "ram", "disk", "gpu", "CPUUtilization"}


def test_format_value_keeps_absolute_values_visible():
    assert format_value(7527.0, "instructions") == "7,527 ops"
    assert format_value(403.0448124670642, "power") == "403.044812 W"
    assert format_value(0.06410499147967352, "execution_time") == "0.064105 ms"
    assert format_value(90.86058839973406, "cpu") == "90.860588 %"
    assert format_value(1234.5) == "1,234.5"  # no unit requested
    assert format_value(None) == "n/a"
