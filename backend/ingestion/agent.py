"""Tiny HTTP agent that exposes this machine's metrics as JSON (Person B).

Run it on any PC you want to monitor:

    python -m backend.ingestion.agent --port 8777

Then from the UI/CLI (same LAN):

    http://<friend-ip>:8777/metrics
"""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .local_pc import snapshot_local_metrics

DEFAULT_PORT = 8777


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


def make_server(host: str = "0.0.0.0", port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), MetricsHandler)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve local machine metrics as JSON")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
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
