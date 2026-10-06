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
