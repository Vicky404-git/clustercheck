"""SQLite storage for detected anomalies (alerts) - Person C.

`source` records where the data came from (local / remote / aws / demo) and
`target` is whatever machine it describes (hostname, EC2 instance id, ...).
"""

import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Sequence

# override with CLOUD_MONITOR_DB=/path/to.db when testing
DB_PATH = os.environ.get("CLOUD_MONITOR_DB", "cloud_monitor.db")

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    target TEXT NOT NULL,
    metric TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    value REAL NOT NULL,
    score REAL NOT NULL,
    created_at TEXT NOT NULL
)
"""


def _connect(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DB_PATH) -> None:
    with _connect(db_path) as conn:
        table = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='alerts'"
        ).fetchone()
        if table:
            columns = {row[1] for row in conn.execute("PRAGMA table_info(alerts)")}
            if "source" not in columns or "target" not in columns:
                conn.execute("DROP TABLE alerts")  # old schema, rebuild
        conn.execute(CREATE_TABLE_SQL)


def insert_alerts(
    anomalies: Sequence[Dict[str, Any]],
    source: str,
    target: str,
    metric: str,
    db_path: str = DB_PATH,
) -> int:
    rows = [
        (
            source,
            target,
            metric,
            _ts(a["timestamp"]),
            float(a["value"]),
            float(a.get("score", 0.0)),
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        for a in anomalies
        if a.get("is_anomaly")
    ]
    if not rows:
        return 0
    with _connect(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO alerts
                (source, target, metric, timestamp, value, score, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
    return len(rows)


def get_recent_alerts(limit: int = 50, db_path: str = DB_PATH) -> List[sqlite3.Row]:
    init_db(db_path)
    with _connect(db_path) as conn:
        cursor = conn.execute(
            "SELECT id, source, target, metric, timestamp, value, score, "
            "created_at FROM alerts ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        return cursor.fetchall()


def count_alerts(db_path: str = DB_PATH) -> int:
    init_db(db_path)
    with _connect(db_path) as conn:
        return conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]


def clear_alerts(db_path: str = DB_PATH) -> int:
    init_db(db_path)
    with _connect(db_path) as conn:
        removed = conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
        conn.execute("DELETE FROM alerts")
    return removed


def _ts(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat(timespec="seconds")
    return str(value)
