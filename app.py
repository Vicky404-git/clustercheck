import platform
from datetime import datetime, timedelta

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from backend.ingestion import (
    PCT_METRICS,
    fetch_series,
    list_instances,
    list_metrics,
    list_remote_metrics,
    metric_label,
)
from backend.ml import detect_anomalies, format_minutes, predict_utilization_trend, summarize
from backend.storage import count_alerts, get_recent_alerts, init_db, insert_alerts

st.set_page_config(
    page_title="Cloud Performance Monitor - Mini Project",
    layout="wide",
)

init_db()

THRESHOLD = 90.0
SOURCE_LABELS = {
    "Local PC": "local",
    "AWS (Live)": "aws",
    "Demo": "demo",
    "Remote PC (HTTP)": "remote",
}


@st.cache_data(ttl=300, show_spinner=False)
def cached_instances(region: str):
    return list_instances(region)


@st.cache_data(ttl=60, show_spinner=False)
def cached_remote_metrics(host: str, port: int):
    return list_remote_metrics(host, port)


def target_label(source: str, opts: dict) -> str:
    if source == "local":
        return platform.node() or "localhost"
    if source == "remote":
        return f"{opts.get('host')}:{opts.get('port')}"
    if source == "aws":
        return opts.get("instance_id", "")
    return "demo-sim"


# ---------------------------------------------------------------- sidebar
st.sidebar.header("Inputs")
source_label = st.sidebar.selectbox("Data Source", list(SOURCE_LABELS))
source = SOURCE_LABELS[source_label]

opts = {}
if source == "local":
    opts["duration"] = st.sidebar.slider("Sampling duration (s)", 10, 60, 30, step=5)
    opts["interval"] = 0.5
    st.sidebar.caption("psutil reads CPU, RAM, Disk, Network and GPU from this machine.")
elif source == "remote":
    opts["host"] = st.sidebar.text_input("Agent host / IP", "127.0.0.1")
    opts["port"] = st.sidebar.number_input("Agent port", 1, 65535, 8777)
    opts["duration"] = st.sidebar.slider("Sampling duration (s)", 10, 60, 30, step=5)
    opts["interval"] = 0.5
    st.sidebar.caption(
        "Run `python -m backend.ingestion.agent` on the other PC first."
    )
elif source == "aws":
    opts["region"] = st.sidebar.text_input("AWS Region", "ap-south-1")
    instances = cached_instances(opts["region"])
    if instances:
        opts["instance_id"] = st.sidebar.selectbox("Instance ID", instances)
    else:
        opts["instance_id"] = st.sidebar.text_input("Instance ID", "i-0123456789abcdef0")
    opts["hours"] = st.sidebar.number_input("Lookback (hours)", 1, 48, 6)
    opts["period"] = st.sidebar.selectbox("Period (s)", [60, 300, 900], index=1)
else:
    opts["hours"] = st.sidebar.number_input("Lookback (hours)", 1, 48, 6)
    opts["period"] = st.sidebar.selectbox("Period (s)", [60, 300, 900], index=1)
    st.sidebar.caption("Generated locally: spiky CPU, leaking RAM, bursty network.")

if source == "remote":
    metric_options = cached_remote_metrics(opts["host"], int(opts["port"]))
else:
    metric_options = list_metrics(source)
metric = st.sidebar.selectbox("Metric", metric_options, key=f"metric_{source}")

run = st.sidebar.button("Fetch & Analyze", width="stretch")

st.sidebar.divider()
st.sidebar.caption(f"Total alerts in cloud_monitor.db: **{count_alerts()}**")


# ----------------------------------------------------------------- charts
def plot_metric(results, label, pred, threshold, target):
    times = [r["timestamp"] for r in results]
    values = [r["value"] for r in results]
    bad = [r for r in results if r["is_anomaly"]]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, values, color="tab:blue", linewidth=1.2, label=label)
    if bad:
        ax.scatter(
            [r["timestamp"] for r in bad],
            [r["value"] for r in bad],
            color="red",
            s=45,
            zorder=3,
            label="anomaly",
        )

    if threshold is not None:
        ax.axhline(threshold, color="gray", linestyle=":", linewidth=1.2,
                   label=f"threshold {threshold:.0f}%")

    if pred and pred.get("supported"):
        t0 = times[0]
        span_min = (times[-1] - t0).total_seconds() / 60.0
        x_end = span_min * 1.4
        minutes_to = pred.get("minutes_to_threshold")
        if minutes_to and 0 < minutes_to <= span_min * 3:
            x_end = minutes_to * 1.06
        xs = np.linspace(0, x_end, 60)
        ys = pred["intercept"] + pred["slope_per_min"] * xs
        xdt = [t0 + timedelta(minutes=float(v)) for v in xs]
        ax.plot(xdt, ys, linestyle="--", color="tab:orange", linewidth=1.4,
                label="linear trend")
        if minutes_to and minutes_to <= x_end and pred.get("eta") is not None:
            ax.axvline(pred["eta"], color="tab:green", linestyle=":", linewidth=1.2,
                       label="projected 90%")

    span = (times[-1] - times[0]).total_seconds()
    fmt = "%H:%M:%S" if span < 900 else "%H:%M"
    ax.set_title(f"{label} - {target}")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel(label)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left", fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    return fig


def show_prediction(pred):
    if pred is None:
        return
    if not pred["supported"]:
        st.caption(f"Trend prediction unavailable: {pred['reason']}")
        return
    st.write(
        f"**Trend:** {pred['direction']} "
        f"({pred['slope_per_min']:+.3f} %/min, R2 = {pred['r2']})"
    )
    if pred["minutes_to_threshold"] is None:
        st.write("Time until 90%: not projected (trend is not rising).")
    elif pred["minutes_to_threshold"] == 0:
        st.write("Time until 90%: value is already at or above 90%.")
    else:
        eta = pred["eta"].strftime("%Y-%m-%d %H:%M UTC")
        st.write(
            f"Time until 90%: about **{format_minutes(pred['minutes_to_threshold'])}** "
            f"(around {eta})"
        )


# ------------------------------------------------------------------ header
st.title("Cloud Performance Monitor - Mini Project")
st.caption(
    "Metrics from psutil / AWS CloudWatch / demo data, anomaly detection with a "
    "scikit-learn Isolation Forest, linear-regression trend prediction, alerts in SQLite."
)


# ------------------------------------------------------------------ fetch
if run:
    if source == "aws" and not str(opts.get("instance_id", "")).strip():
        st.error("Enter an Instance ID first.")
    else:
        progress = st.spinner(f"Fetching metrics from {source_label}")
        with progress:
            try:
                points = fetch_series(source, metric, **opts)
            except (RuntimeError, ValueError) as exc:
                st.error(str(exc))
                points = []

        if points:
            results = detect_anomalies(points)
            stats = summarize(results)
            pred = (
                predict_utilization_trend(results, threshold=THRESHOLD, metric=metric)
                if metric in PCT_METRICS
                else None
            )
            target = target_label(source, opts)
            saved = insert_alerts(results, source=source, target=target, metric=metric)
            st.session_state.update(
                results=results,
                stats=stats,
                pred=pred,
                saved=saved,
                metric=metric,
                source=source,
                target=target,
                run_label=(
                    f"{source_label} | {target} | {metric_label(metric)} | "
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                ),
            )

# ---------------------------------------------------------------- results
if "results" in st.session_state:
    stats = st.session_state["stats"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Data points", stats["points"])
    c2.metric("Anomalies flagged", stats["anomalies"])
    c3.metric("Max value", stats["max"])
    c4.metric("Alerts written to DB", st.session_state["saved"])

    st.caption("Last run: " + st.session_state["run_label"])
    st.pyplot(
        plot_metric(
            st.session_state["results"],
            metric_label(st.session_state["metric"]),
            st.session_state.get("pred"),
            THRESHOLD if st.session_state["metric"] in PCT_METRICS else None,
            st.session_state["target"],
        )
    )

    if st.session_state.get("pred"):
        st.markdown("**ML trend prediction (linear regression)**")
        show_prediction(st.session_state["pred"])

    if stats["anomalies"] == 0:
        st.info("No anomalies found in this range.")
else:
    st.info("Pick your inputs on the left and click **Fetch & Analyze**.")

# ----------------------------------------------------------------- alerts
st.subheader("Recent Alerts (from cloud_monitor.db)")
rows = get_recent_alerts(limit=50)
if rows:
    st.dataframe(pd.DataFrame([dict(r) for r in rows]), width="stretch", hide_index=True)
    st.caption(
        "Query: SELECT id, source, target, metric, timestamp, value, score, "
        "created_at FROM alerts ORDER BY id DESC LIMIT 50"
    )
else:
    st.write("No alerts recorded yet.")
