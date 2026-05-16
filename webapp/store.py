"""SQLite store for prediction history."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict


FeatureValue = float | int | None
FeaturesByDriver = dict[str, dict[str, FeatureValue]]


class PredictionRow(TypedDict):
    id: int
    created_at: str
    year: int
    event_id: int
    event_name: str
    session: str
    drivers: list[tuple[str, float]]
    features: FeaturesByDriver


class RaceSummary(TypedDict):
    year: int
    event_id: int
    event_name: str
    session: str
    last_at: str


_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    year INTEGER NOT NULL,
    event_id INTEGER NOT NULL,
    event_name TEXT NOT NULL,
    session TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_predictions_race
    ON predictions(year, event_id, session, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_created
    ON predictions(created_at DESC);
"""


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with _connect(db_path) as conn:
        conn.executescript(_SCHEMA)
        # Migration: add features_json to pre-existing rows. SQLite raises
        # OperationalError if the column already exists.
        try:
            conn.execute(
                "ALTER TABLE predictions "
                "ADD COLUMN features_json TEXT NOT NULL DEFAULT '{}'"
            )
        except sqlite3.OperationalError:
            pass


def insert_prediction(
    db_path: Path,
    year: int,
    event_id: int,
    event_name: str,
    session: str,
    drivers: list[tuple[str, float]],
    features: FeaturesByDriver,
) -> int:
    payload = json.dumps([{"driver": d, "pred": p} for d, p in drivers])
    features_payload = json.dumps(features)
    now = datetime.now(timezone.utc).isoformat()
    with _connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO predictions "
            "(created_at, year, event_id, event_name, session, payload_json, features_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (now, year, event_id, event_name, session, payload, features_payload),
        )
        return cur.lastrowid or 0


def _row_to_prediction(row: sqlite3.Row) -> PredictionRow:
    drivers = [
        (d["driver"], float(d["pred"])) for d in json.loads(row["payload_json"])
    ]
    features_raw = row["features_json"] if "features_json" in row.keys() else "{}"
    return PredictionRow(
        id=row["id"],
        created_at=row["created_at"],
        year=row["year"],
        event_id=row["event_id"],
        event_name=row["event_name"],
        session=row["session"],
        drivers=drivers,
        features=json.loads(features_raw or "{}"),
    )


def latest(db_path: Path) -> PredictionRow | None:
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM predictions ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
    return _row_to_prediction(row) if row else None


def latest_for_race(
    db_path: Path, year: int, event_id: int, session: str
) -> PredictionRow | None:
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM predictions "
            "WHERE year = ? AND event_id = ? AND session = ? "
            "ORDER BY created_at DESC LIMIT 1",
            (year, event_id, session),
        ).fetchone()
    return _row_to_prediction(row) if row else None


def distinct_races(db_path: Path) -> list[RaceSummary]:
    """One row per (year, event_id, session), keyed by most recent prediction."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT year, event_id, event_name, session, "
            "MAX(created_at) AS last_at "
            "FROM predictions "
            "GROUP BY year, event_id, session "
            "ORDER BY year DESC, event_id DESC, session"
        ).fetchall()
    return [
        RaceSummary(
            year=r["year"],
            event_id=r["event_id"],
            event_name=r["event_name"],
            session=r["session"],
            last_at=r["last_at"],
        )
        for r in rows
    ]
