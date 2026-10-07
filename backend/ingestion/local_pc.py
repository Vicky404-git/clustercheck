"""Live metrics from this machine, plus the tiny HTTP agent that exposes
them to other machines on the LAN (Person B).

Every source ends up producing the same shape of data, so the ML layer and
the UI never care where the numbers came from:

    {"cpu": [{"timestamp": dt, "value": float}, ...], "ram": [...], ...}

Run on any PC you want to monitor:

    python -m backend.ingestion.local_pc --port 8777

then poll it from the UI/CLI as the "Remote PC (HTTP)" source.
"""

import argparse
import json
import os
import platform
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional

import psutil

try:
    import GPUtil
except ImportError:  # GPUtil is optional
    GPUtil = None

DEFAULT_DURATION = 30
DEFAULT_INTERVAL = 0.5
REQUEST_TIMEOUT = 3
DEFAULT_AGENT_PORT = 8777

BASE_METRICS = ["cpu", "ram", "disk", "network_in", "network_out"]


# ======================================================================
# reading this machine (psutil)
# ======================================================================
def get_gpu_percent() -> Optional[float]:
    """NVIDIA GPU utilisation via GPUtil, or None when no GPU/driver exists."""
    if GPUtil is None:
        return None
    try:
        gpus = GPUtil.getGPUs()
    except Exception:
        return None
    if not gpus:
        return None
    return round(float(gpus[0].load) * 100, 2)


def _disk_percent() -> float:
    if platform.system() == "Windows":
        path = os.environ.get("SystemDrive", "C:") + "\\"
    else:
        path = "/"
    return float(psutil.disk_usage(path).percent)


def snapshot_local_metrics(cpu_interval: float = 0.1) -> dict:
    """One reading of this machine. Returned as a plain JSON-safe dict."""
    net = psutil.net_io_counters()
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        "hostname": platform.node(),
        "platform": platform.system(),
        "cpu": psutil.cpu_percent(interval=cpu_interval),
        "ram": float(psutil.virtual_memory().percent),
        "disk": _disk_percent(),
        "gpu": get_gpu_percent(),
        "net_bytes_recv": int(net.bytes_recv),
        "net_bytes_sent": int(net.bytes_sent),
    }


def available_local_metrics() -> List[str]:
    metrics = list(BASE_METRICS)
    if get_gpu_percent() is not None:
        metrics.append("gpu")
    return metrics


def collect_local_metrics(
    duration: float = DEFAULT_DURATION,
    interval: float = DEFAULT_INTERVAL,
) -> Dict[str, List[dict]]:
    """Poll this machine for `duration` seconds and build metric series."""
    interval = max(0.1, float(interval))
    steps = max(3, int(float(duration) / interval))

    psutil.cpu_percent(interval=0.1)  # warm-up
    snapshots = []
    for _ in range(steps):
        time.sleep(interval)
        snapshots.append(snapshot_local_metrics(cpu_interval=0.0))
    return _build_series(snapshots)


# ======================================================================
# polling a remote machine running the agent below
# ======================================================================
def fetch_remote_metrics(
    host: str,
    port: int,
    duration: float = DEFAULT_DURATION,
    interval: float = DEFAULT_INTERVAL,
) -> Dict[str, List[dict]]:
    """Poll a remote agent and build metric series."""
    interval = max(0.1, float(interval))
    steps = max(3, int(float(duration) / interval))
    url = f"http://{host}:{port}/metrics"

    snapshots = []
    for i in range(steps):
        if i:
            time.sleep(interval)
        snapshots.append(_get_snapshot(url, host, port))
    return _build_series(snapshots)


def list_remote_metrics(host: str, port: int, timeout: int = 2) -> List[str]:
    """Ask a remote agent which metrics it exposes (best effort)."""
    try:
        snapshot = _get_snapshot(f"http://{host}:{port}/metrics", host, port, timeout)
    except RuntimeError:
        return list(BASE_METRICS)
    metrics = [m for m in BASE_METRICS]
    if snapshot.get("gpu") is not None:
        metrics.append("gpu")
    return metrics


def _get_snapshot(url: str, host: str, port: int, timeout: int = REQUEST_TIMEOUT) -> dict:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Agent at {host}:{port} returned HTTP {exc.code}") from exc
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise RuntimeError(
            f"Cannot reach agent at {host}:{port} - is it running? "
            f"(python -m backend.ingestion.local_pc). Details: {exc}"
        ) from exc


def _parse_ts(value) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def _build_series(snapshots: List[dict]) -> Dict[str, List[dict]]:
    series: Dict[str, List[dict]] = {
        "cpu": [],
        "ram": [],
        "disk": [],
        "gpu": [],
        "network_in": [],
        "network_out": [],
    }
    previous = None
    previous_ts = None
    for snapshot in snapshots:
        ts = _parse_ts(snapshot["timestamp"])
        for key in ("cpu", "ram", "disk", "gpu"):
            value = snapshot.get(key)
            if value is not None:
                series[key].append({"timestamp": ts, "value": float(value)})
        if previous is not None:
            elapsed = (ts - previous_ts).total_seconds()
            if elapsed > 0:
                recv = snapshot.get("net_bytes_recv", 0) - previous.get("net_bytes_recv", 0)
                sent = snapshot.get("net_bytes_sent", 0) - previous.get("net_bytes_sent", 0)
                series["network_in"].append({"timestamp": ts, "value": round(recv / elapsed, 1)})
                series["network_out"].append({"timestamp": ts, "value": round(sent / elapsed, 1)})
        previous = snapshot
        previous_ts = ts
    return {name: points for name, points in series.items() if points}


# ======================================================================
# serving this machine to others (HTTP agent)
# ======================================================================
class MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 - required by BaseHTTPRequestHandler
        if self.path.rstrip("/").endswith("/metrics"):
            payload = json.dumps(snapshot_local_metrics()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        else:
            body = b'{"error": "use /metrics"}'
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def log_message(self, fmt, *args):
        print(f"[agent] {self.client_address[0]} - {fmt % args}")


def make_server(host: str = "0.0.0.0", port: int = DEFAULT_AGENT_PORT) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), MetricsHandler)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Expose this machine's metrics as JSON over HTTP (agent)"
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_AGENT_PORT)
    args = parser.parse_args()

    server = make_server(args.host, args.port)
    print(f"Metrics agent listening on http://{args.host}:{args.port}/metrics")
    print("Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping agent")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
