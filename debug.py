#!/usr/bin/env python3
"""CLI debug tool for the Cloud Performance Monitor mini project (Person A).

Lets you exercise every layer from the terminal without opening Streamlit:

    python debug.py metrics
    python debug.py local --duration 10
    python debug.py remote --host 192.168.1.50 --port 8777
    python debug.py aws --instance i-0123456789abcdef0 --metric CPUUtilization
    python debug.py demo --metric ram --hours 6
    python debug.py anomaly --source demo --metric cpu
    python debug.py predict --source demo --metric ram
    python debug.py db-test
    python debug.py db-show
    python debug.py db-clear
"""

import argparse
import sys

from backend.ingestion import fetch_series, list_metrics, metric_label, snapshot_local_metrics
from backend.ingestion.local_pc import collect_local_metrics, fetch_remote_metrics
from backend.ml import detect_anomalies, format_minutes, predict_utilization_trend, summarize
from backend.storage import clear_alerts, count_alerts, get_recent_alerts, init_db, insert_alerts


# ----------------------------------------------------------------- helpers
def add_source_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", choices=["local", "aws", "demo", "remote"], default="demo")
    parser.add_argument("--metric", default=None, help="metric name (see: debug.py metrics)")
    parser.add_argument("--region", default="ap-south-1")
    parser.add_argument("--instance", default="", help="EC2 instance id (aws source)")
    parser.add_argument("--host", default="127.0.0.1", help="agent host (remote source)")
    parser.add_argument("--port", type=int, default=8777, help="agent port (remote source)")
    parser.add_argument("--hours", type=int, default=6, help="lookback (aws/demo)")
    parser.add_argument("--period", type=int, default=300, help="seconds per datapoint (aws/demo)")
    parser.add_argument("--duration", type=float, default=15, help="sampling seconds (local/remote)")
    parser.add_argument("--interval", type=float, default=0.5, help="sampling gap seconds (local/remote)")


def source_opts(args: argparse.Namespace) -> dict:
    return {
        "region": args.region,
        "instance_id": args.instance,
        "host": args.host,
        "port": args.port,
        "hours": args.hours,
        "period": args.period,
        "duration": args.duration,
        "interval": args.interval,
    }


def resolve_metric(args: argparse.Namespace) -> str:
    if args.metric:
        return args.metric
    return list_metrics(args.source)[0]


def fetch_or_exit(source: str, metric: str, opts: dict) -> list:
    try:
        return fetch_series(source, metric, **opts)
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


def print_points(points: list, limit: int = 15) -> None:
    print(f"{'#':>4}  {'timestamp':<32} {'value':>14}")
    print("-" * 52)
    for i, p in enumerate(points[:limit], 1):
        print(f"{i:>4}  {str(p['timestamp']):<32} {float(p['value']):>14.2f}")
    if len(points) > limit:
        print(f"... {len(points) - limit} more rows (total {len(points)})")


def print_series_summary(series: dict) -> None:
    print(f"{'metric':<16} {'points':>7} {'min':>12} {'max':>12} {'avg':>12}")
    print("-" * 62)
    for name, points in series.items():
        if not points:
            continue
        values = [float(p["value"]) for p in points]
        print(
            f"{name:<16} {len(points):>7} {min(values):>12.2f} "
            f"{max(values):>12.2f} {sum(values) / len(values):>12.2f}"
        )


# ---------------------------------------------------------------- commands
def cmd_metrics(_args) -> None:
    for source in ["local", "demo", "aws"]:
        print(f"{source:<8} {', '.join(list_metrics(source))}")
    print(f"{'remote':<8} {', '.join(list_metrics('remote'))} (defaults, needs a live agent)")


def cmd_local(args) -> None:
    print("== snapshot (single reading) ==")
    snap = snapshot_local_metrics()
    print(f"hostname : {snap['hostname']} ({snap['platform']})")
    print(f"time     : {snap['timestamp']}")
    for key in ("cpu", "ram", "disk", "gpu"):
        value = snap.get(key)
        print(f"{key:<8} : {value if value is not None else 'no GPU detected'}")

    if args.duration > 0:
        print(f"\n== collected over {args.duration}s (interval {args.interval}s) ==")
        series = collect_local_metrics(duration=args.duration, interval=args.interval)
        print_series_summary(series)
        if args.metric and args.metric in series:
            print(f"\n== {metric_label(args.metric)} ==")
            print_points(series[args.metric])


def cmd_remote(args) -> None:
    try:
        series = fetch_remote_metrics(
            args.host, args.port, duration=args.duration, interval=args.interval
        )
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"== {args.host}:{args.port} ==")
    print_series_summary(series)
    if args.metric and args.metric in series:
        print(f"\n== {metric_label(args.metric)} ==")
        print_points(series[args.metric])


def cmd_aws(args) -> None:
    metric = args.metric or list_metrics("aws")[0]
    opts = {
        "instance_id": args.instance,
        "region": args.region,
        "hours": args.hours,
        "period": args.period,
    }
    points = fetch_or_exit("aws", metric, opts)
    print(f"== {metric_label(metric)} | {args.instance} | {args.region} ==")
    print_points(points, limit=20)


def cmd_demo(args) -> None:
    metric = args.metric or "cpu"
    points = fetch_or_exit("demo", metric, {"hours": args.hours, "period": args.period})
    print(f"== {metric_label(metric)} ==")
    print_points(points)


def cmd_anomaly(args) -> None:
    metric = resolve_metric(args)
    points = fetch_or_exit(args.source, metric, source_opts(args))
    results = detect_anomalies(points)
    stats = summarize(results)
    print(f"== anomaly detection ({args.source} / {metric_label(metric)}) ==")
    print(f"points   : {stats['points']}")
    print(f"anomalies: {stats['anomalies']}")
    print(f"min/avg/max: {stats['min']} / {stats['avg']} / {stats['max']}")

    flagged = [r for r in results if r["is_anomaly"]]
    if flagged:
        print(f"\n{'timestamp':<32} {'value':>12} {'score':>10}")
        print("-" * 56)
        for row in flagged:
            print(f"{str(row['timestamp']):<32} {float(row['value']):>12.2f} {row['score']:>10}")

    if args.save:
        init_db()
        saved = insert_alerts(
            results, source=args.source, target=args.target, metric=metric
        )
        print(f"\nsaved {saved} alert(s) to cloud_monitor.db "
              f"(total now: {count_alerts()})")


def cmd_predict(args) -> None:
    metric = resolve_metric(args)
    points = fetch_or_exit(args.source, metric, source_opts(args))
    pred = predict_utilization_trend(points, threshold=90.0, metric=metric)
    print(f"== trend prediction ({args.source} / {metric_label(metric)}) ==")
    if not pred["supported"]:
        print(f"not supported: {pred['reason']}")
        return
    print(f"current  : {pred['current']} %")
    print(f"trend    : {pred['direction']} ({pred['slope_per_min']} %/min, R2={pred['r2']})")
    if pred["minutes_to_threshold"] is None:
        print("time to 90%: not projected (trend is not rising)")
    elif pred["minutes_to_threshold"] == 0:
        print("time to 90%: already at or above 90%")
    else:
        print(
            f"time to 90%: {format_minutes(pred['minutes_to_threshold'])} "
            f"(around {pred['eta']})"
        )


def cmd_db_test(_args) -> None:
    init_db()
    points = fetch_or_exit("demo", "cpu", {"hours": 6, "period": 300})
    results = detect_anomalies(points)
    saved = insert_alerts(results, source="demo", target="debug-db-test", metric="cpu")
    print(f"inserted {saved} alert(s), table now has {count_alerts()} row(s)\n")
    cmd_db_show(argparse.Namespace(limit=10))


def cmd_db_show(args) -> None:
    init_db()
    rows = get_recent_alerts(limit=args.limit)
    if not rows:
        print("alerts table is empty")
        return
    print(f"{'id':>4} {'source':<7} {'target':<20} {'metric':<8} {'value':>10} {'timestamp':<32}")
    print("-" * 88)
    for row in rows:
        print(
            f"{row['id']:>4} {row['source']:<7} {row['target']:<20} {row['metric']:<8} "
            f"{row['value']:>10.2f} {row['timestamp']:<32}"
        )


def cmd_db_clear(_args) -> None:
    removed = clear_alerts()
    print(f"cleared {removed} row(s), {count_alerts()} remaining")


# ------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Debug every layer of the Cloud Performance Monitor from the CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("metrics", help="list metrics available per source").set_defaults(
        func=cmd_metrics
    )

    p = sub.add_parser("local", help="read this machine via psutil")
    p.add_argument("--duration", type=float, default=10)
    p.add_argument("--interval", type=float, default=0.5)
    p.add_argument("--metric", default=None)
    p.set_defaults(func=cmd_local)

    p = sub.add_parser("remote", help="poll another PC running the agent")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8777)
    p.add_argument("--duration", type=float, default=10)
    p.add_argument("--interval", type=float, default=0.5)
    p.add_argument("--metric", default=None)
    p.set_defaults(func=cmd_remote)

    p = sub.add_parser("aws", help="fetch CloudWatch metrics")
    p.add_argument("--instance", required=True)
    p.add_argument("--region", default="ap-south-1")
    p.add_argument("--metric", default=None)
    p.add_argument("--hours", type=int, default=6)
    p.add_argument("--period", type=int, default=300)
    p.set_defaults(func=cmd_aws)

    p = sub.add_parser("demo", help="fetch generated demo metrics")
    p.add_argument("--metric", default="cpu")
    p.add_argument("--hours", type=int, default=6)
    p.add_argument("--period", type=int, default=300)
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("anomaly", help="run the Isolation Forest detector")
    add_source_options(p)
    p.add_argument("--save", action="store_true", help="write alerts to cloud_monitor.db")
    p.add_argument("--target", default="debug-cli", help="target name used when saving")
    p.set_defaults(func=cmd_anomaly)

    p = sub.add_parser("predict", help="run the linear-regression trend predictor")
    add_source_options(p)
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("db-test", help="insert demo anomalies and read them back")
    p.set_defaults(func=cmd_db_test)

    p = sub.add_parser("db-show", help="print recent rows from cloud_monitor.db")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_db_show)

    sub.add_parser("db-clear", help="delete every row from the alerts table").set_defaults(
        func=cmd_db_clear
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
