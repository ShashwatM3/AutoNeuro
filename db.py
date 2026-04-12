"""SQLite schema for AutoNeuro dashboard (runs + flags)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DB_PATH = REPO_ROOT / "database.db"

_RUNS_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    iteration INTEGER,
    commit_hash TEXT,
    status TEXT,
    metric_value REAL,
    diff TEXT,
    error_log TEXT,
    timestamp TEXT
);
"""

_FLAGS_SQL = """
CREATE TABLE IF NOT EXISTS flags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    iteration INTEGER,
    trigger_reason TEXT,
    context_summary TEXT,
    human_instruction TEXT,
    resolved INTEGER DEFAULT 0,
    timestamp TEXT
);
"""


def init_db() -> None:
    """Create database tables if they do not exist."""
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.executescript(_RUNS_SQL + _FLAGS_SQL)
        conn.commit()
    finally:
        conn.close()


def clear_dashboard_db() -> None:
    """Delete all runs/flags rows and reset row IDs."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM runs")
        cursor.execute("DELETE FROM flags")
        cursor.execute("DELETE FROM sqlite_sequence WHERE name IN ('runs', 'flags')")
        conn.commit()
    finally:
        conn.close()
