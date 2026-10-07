import pytest

from backend.ingestion import KAGGLE_METRICS, SOURCES, fetch_series, list_metrics
from backend.ingestion.datasets import fetch_kaggle_series

CSV_CONTENT = """vm_id,timestamp,cpu_usage,memory_usage,network_traffic,power_consumption,num_executed_instructions,execution_time
v2,2023-01-01 10:00:00,10.0,40.0,90.0,190.0,7527.0,69.34557502116084
v1,2023-01-02 10:00:00,20.0,50.0,100.0,200.0,5348.0,41.39603965431024
v3,2023-01-03 10:00:00,,60.0,110.0,210.0,5483.0,24.602548884707854
v4,not-a-timestamp,30.0,70.0,120.0,220.0,5876.0,16.456669858380668
v5,2023-01-04 10:00:00,40.0,80.0,,230.0,,55.30799161503772
"""


@pytest.fixture()
def csv_file(tmp_path):
    path = tmp_path / "vmCloud_data.csv"
    path.write_text(CSV_CONTENT)
    return str(path)


def test_cpu_mapping_drops_nulls_and_sorts(csv_file):
    points = fetch_kaggle_series(csv_file, metric="cpu", limit=100)
    assert [p["value"] for p in points] == [10.0, 20.0, 40.0]
    assert points == sorted(points, key=lambda p: p["timestamp"])
    assert all(isinstance(p["timestamp"], object) for p in points)
    assert all(isinstance(p["value"], float) for p in points)


def test_ram_mapping(csv_file):
    points = fetch_kaggle_series(csv_file, metric="ram", limit=100)
    # v4 has a valid memory_usage but an unparseable timestamp -> dropped
    assert [p["value"] for p in points] == [40.0, 50.0, 60.0, 80.0]


def test_network_mapping(csv_file):
    points = fetch_kaggle_series(csv_file, metric="network", limit=100)
    # v5 network_traffic is null -> dropped
    assert [p["value"] for p in points] == [90.0, 100.0, 110.0]


def test_power_mapping(csv_file):
    points = fetch_kaggle_series(csv_file, metric="power", limit=100)
    assert [p["value"] for p in points] == [190.0, 200.0, 210.0, 230.0]


def test_instructions_mapping(csv_file):
    points = fetch_kaggle_series(csv_file, metric="instructions", limit=100)
    # v5 has no instruction count -> dropped, chronological order kept
    assert [p["value"] for p in points] == [7527.0, 5348.0, 5483.0]


def test_execution_time_mapping_keeps_precision(csv_file):
    points = fetch_kaggle_series(csv_file, metric="execution_time", limit=100)
    assert [p["value"] for p in points] == pytest.approx(
        [
            69.34557502116084,
            41.39603965431024,
            24.602548884707854,
            55.30799161503772,
        ],
        rel=1e-12,  # only CSV float-parsing precision, no rounding on our side
    )


def test_limit_keeps_most_recent_rows(csv_file):
    points = fetch_kaggle_series(csv_file, metric="cpu", limit=2)
    assert [p["value"] for p in points] == [20.0, 40.0]


def test_missing_file_raises(tmp_path):
    with pytest.raises(RuntimeError, match="CSV file not found"):
        fetch_kaggle_series(str(tmp_path / "nope.csv"), metric="cpu")


def test_missing_column_raises(tmp_path):
    path = tmp_path / "partial.csv"
    path.write_text("vm_id,timestamp,cpu_usage,memory_usage\nv1,2023-01-01 00:00:00,1.0,2.0\n")
    with pytest.raises(RuntimeError, match="missing column"):
        fetch_kaggle_series(str(path), metric="network")
    with pytest.raises(RuntimeError, match="missing column"):
        fetch_kaggle_series(str(path), metric="power")


def test_unsupported_metric_raises(csv_file):
    with pytest.raises(RuntimeError, match="Unsupported Kaggle metric"):
        fetch_kaggle_series(csv_file, metric="disk")


def test_all_values_missing_raises(tmp_path):
    path = tmp_path / "empty_metric.csv"
    path.write_text(
        "timestamp,cpu_usage,memory_usage,network_traffic\n"
        "2023-01-01 00:00:00,,1.0,2.0\n"
        "2023-01-02 00:00:00,,3.0,4.0\n"
    )
    with pytest.raises(RuntimeError, match="No usable rows"):
        fetch_kaggle_series(str(path), metric="cpu")


def test_unreadable_file_raises(tmp_path):
    path = tmp_path / "garbage.csv"
    path.write_text("")
    with pytest.raises(RuntimeError, match="Could not read"):
        fetch_kaggle_series(str(path), metric="cpu")


def test_kaggle_registered_in_dispatcher(csv_file):
    assert "kaggle" in SOURCES
    assert list_metrics("kaggle") == [
        "cpu",
        "ram",
        "network",
        "power",
        "instructions",
        "execution_time",
    ]
    assert KAGGLE_METRICS == {
        "cpu": "cpu_usage",
        "ram": "memory_usage",
        "network": "network_traffic",
        "power": "power_consumption",
        "instructions": "num_executed_instructions",
        "execution_time": "execution_time",
    }
    points = fetch_series("kaggle", "cpu", csv_path=csv_file, limit=1)
    assert len(points) == 1
    assert points[0]["value"] == 40.0  # most recent row


def test_kaggle_bad_file_via_dispatcher(tmp_path):
    with pytest.raises(RuntimeError, match="CSV file not found"):
        fetch_series("kaggle", "cpu", csv_path=str(tmp_path / "gone.csv"))
