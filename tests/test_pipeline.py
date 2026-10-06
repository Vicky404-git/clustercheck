import numpy as np

from backend.ingestion.demo_data import get_demo_metrics
from backend.ml.anomaly import detect_anomalies, summarize
from backend.ml.predictor import format_minutes, predict_utilization_trend
from backend.storage import clear_alerts, count_alerts, get_recent_alerts, init_db, insert_alerts


# ------------------------------------------------------------- demo shapes
def test_demo_returns_all_metrics_with_shared_timestamps():
    data = get_demo_metrics(hours=6, period=300)
    assert set(data) == {"cpu", "ram", "disk", "network_in", "network_out"}
    for points in data.values():
        assert len(points) == 72
        assert {"timestamp", "value"} <= set(points[0].keys())
    stamps = [p["timestamp"] for p in data["cpu"]]
    for points in data.values():
        assert [p["timestamp"] for p in points] == stamps


def test_demo_cpu_has_random_spikes():
    values = [p["value"] for p in get_demo_metrics()["cpu"]]
    median = float(np.median(values))
    assert max(values) > 3 * median  # spikes stand out from the idle baseline


def test_demo_ram_leaks_upward():
    values = [p["value"] for p in get_demo_metrics()["ram"]]
    quarter = len(values) // 4
    assert np.mean(values[-quarter:]) > np.mean(values[:quarter]) + 10
    slope = np.polyfit(range(len(values)), values, 1)[0]
    assert slope > 0


def test_demo_network_is_bursty():
    values = [p["value"] for p in get_demo_metrics()["network_in"]]
    median = float(np.median(values))
    assert max(values) > 4 * median  # bursts far above the baseline
    quiet = sum(1 for v in values if v < 1.5 * median) / len(values)
    assert quiet > 0.5  # mostly quiet with occasional spikes


def test_demo_metrics_are_visually_distinct():
    data = get_demo_metrics()
    cpu = np.array([p["value"] for p in data["cpu"]])
    ram = np.array([p["value"] for p in data["ram"]])
    net = np.array([p["value"] for p in data["network_in"]])
    assert not np.allclose(np.sort(cpu)[: len(ram)], np.sort(ram)[: len(cpu)])
    assert not np.allclose(cpu, net)
    # different coefficient of variation profiles
    assert np.std(cpu) > np.std(ram)
    assert np.std(net) / np.mean(net) > np.std(ram) / np.mean(ram)


# ---------------------------------------------------------------- anomalies
def test_detector_finds_anomalies_on_demo_cpu():
    points = get_demo_metrics()["cpu"]
    results = detect_anomalies(points)
    assert len(results) == len(points)
    assert any(r["is_anomaly"] for r in results)
    assert all("score" in r for r in results)


def test_detector_handles_small_series():
    points = get_demo_metrics(hours=1, period=300)["cpu"][:10]
    results = detect_anomalies(points)
    assert len(results) == 10
    assert not any(r["is_anomaly"] for r in results)


def test_detector_ignores_flat_series():
    points = [{"timestamp": p["timestamp"], "value": 42.0} for p in get_demo_metrics()["cpu"]]
    results = detect_anomalies(points)
    assert not any(r["is_anomaly"] for r in results)


def test_summarize_counts():
    stats = summarize(detect_anomalies(get_demo_metrics()["cpu"]))
    assert stats["points"] == 72
    assert stats["anomalies"] >= 1
    assert stats["max"] >= stats["min"]


# --------------------------------------------------------------- predictor
def _series(formula, count=60, step_s=60):
    from datetime import datetime, timedelta, timezone

    start = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    return [
        {
            "timestamp": start + timedelta(seconds=i * step_s),
            "value": float(formula(i)),
        }
        for i in range(count)
    ]


def test_predictor_eta_on_linear_rise():
    pred = predict_utilization_trend(_series(lambda t: 20 + 0.5 * t), threshold=90.0)
    assert pred["supported"]
    assert pred["direction"] == "rising"
    assert abs(pred["slope_per_min"] - 0.5) < 0.01
    assert pred["r2"] > 0.99
    current = pred["current"]
    expected = (90.0 - current) / pred["slope_per_min"]
    assert abs(pred["minutes_to_threshold"] - expected) < 1.0
    assert pred["eta"] is not None


def test_predictor_no_eta_when_falling():
    pred = predict_utilization_trend(_series(lambda t: 80 - 0.2 * t), threshold=90.0)
    assert pred["supported"]
    assert pred["direction"] == "falling"
    assert pred["minutes_to_threshold"] is None
    assert pred["eta"] is None


def test_predictor_already_above_threshold():
    pred = predict_utilization_trend(_series(lambda t: 94 + 0.01 * t), threshold=90.0)
    assert pred["supported"]
    assert pred["minutes_to_threshold"] == 0.0
    assert pred["eta"] == _series(lambda t: 94 + 0.01 * t)[-1]["timestamp"]


def test_predictor_needs_enough_points():
    pred = predict_utilization_trend(_series(lambda t: 20 + t, count=5))
    assert not pred["supported"]
    assert "at least" in pred["reason"]


def test_format_minutes():
    assert format_minutes(45) == "45m"
    assert format_minutes(125) == "2h 5m"
    assert format_minutes(60 * 27) == "1d 3h"
    assert format_minutes(None) == "n/a"


# --------------------------------------------------------------------- db
def test_database_roundtrip(tmp_path):
    db = str(tmp_path / "test_alerts.db")
    init_db(db)
    assert count_alerts(db) == 0

    results = detect_anomalies(get_demo_metrics()["cpu"])
    saved = insert_alerts(
        results, source="demo", target="unit-test", metric="cpu", db_path=db
    )
    assert saved == sum(1 for r in results if r["is_anomaly"])
    assert count_alerts(db) == saved

    rows = get_recent_alerts(limit=10, db_path=db)
    assert len(rows) == min(10, saved)
    assert rows[0]["source"] == "demo"
    assert rows[0]["target"] == "unit-test"
    assert rows[0]["metric"] == "cpu"

    clear_alerts(db)
    assert count_alerts(db) == 0
