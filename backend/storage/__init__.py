"""SQLite storage layer (Person C)."""

from .database import (
    DB_PATH,
    clear_alerts,
    count_alerts,
    get_recent_alerts,
    init_db,
    insert_alerts,
)

__all__ = [
    "DB_PATH",
    "init_db",
    "insert_alerts",
    "get_recent_alerts",
    "count_alerts",
    "clear_alerts",
]
