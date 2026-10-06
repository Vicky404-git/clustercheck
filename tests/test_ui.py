import os

from streamlit.testing.v1 import AppTest

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


def _by_label(widgets, label):
    return [w for w in widgets if w.label == label]


def test_app_demo_flow():
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception
    assert at.title[0].value == "Cloud Performance Monitor - Mini Project"

    source = _by_label(at.sidebar.selectbox, "Data Source")[0]
    source.set_value("Demo")
    at.run()
    assert not at.exception

    metric = _by_label(at.sidebar.selectbox, "Metric")[0]
    metric.set_value("ram")
    at.run()

    _by_label(at.sidebar.button, "Fetch & Analyze")[0].click().run()
    assert not at.exception

    labels = [m.label for m in at.metric]
    assert "Data points" in labels
    assert "Anomalies flagged" in labels
    assert any(m.label == "Data points" and m.value == "72" for m in at.metric)
    assert len(at.dataframe) >= 1


def test_app_bad_source_shows_error_not_crash():
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    source = _by_label(at.sidebar.selectbox, "Data Source")[0]
    source.set_value("Remote PC (HTTP)")
    at.run()
    _by_label(at.sidebar.button, "Fetch & Analyze")[0].click().run()
    assert not at.exception
    assert len(at.error) == 1
    assert "Cannot reach agent" in at.error[0].value


def test_app_kaggle_source(tmp_path):
    csv = tmp_path / "vmCloud_data.csv"
    csv.write_text(
        "vm_id,timestamp,cpu_usage,memory_usage,network_traffic\n"
        "v1,2023-01-01 10:00:00,10.0,40.0,90.0\n"
        "v2,2023-01-02 10:00:00,20.0,50.0,100.0\n"
        "v3,2023-01-03 10:00:00,,60.0,110.0\n"
    )
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    _by_label(at.sidebar.selectbox, "Data Source")[0].set_value("Kaggle Dataset (CSV)")
    at.run()
    assert not at.exception

    _by_label(at.sidebar.text_input, "CSV path")[0].set_value(str(csv))
    at.run()
    _by_label(at.sidebar.button, "Fetch & Analyze")[0].click().run()
    assert not at.exception
    assert not at.error
    points = [m for m in at.metric if m.label == "Data points"]
    assert points and points[0].value == "2"  # null cpu row skipped


def test_app_kaggle_missing_file_shows_error(tmp_path):
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    _by_label(at.sidebar.selectbox, "Data Source")[0].set_value("Kaggle Dataset (CSV)")
    at.run()
    _by_label(at.sidebar.text_input, "CSV path")[0].set_value(
        str(tmp_path / "does_not_exist.csv")
    )
    at.run()
    _by_label(at.sidebar.button, "Fetch & Analyze")[0].click().run()
    assert not at.exception
    assert len(at.error) == 1
    assert "CSV file not found" in at.error[0].value
