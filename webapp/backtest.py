"""Per-event backtest predictions for the current season.

Enumerates events on disk for a given year, runs ``predict_session`` for each
target session that has real (non-placeholder) results, stores the result via
``webapp.store``, and exposes helpers for resolving actual top-N finishers.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TypedDict

import polars as pl

from scripts.inference import predict_session
from webapp import store

log = logging.getLogger(__name__)

TARGET_SESSIONS: tuple[str, ...] = ("Sprint", "Q", "R")


class EventInfo(TypedDict):
    year: int
    event_id: int
    event_name: str
    event_path: Path
    target_sessions: list[str]


def _event_name(event_path: Path) -> str:
    raw = event_path.name.split("_", 1)[1] if "_" in event_path.name else event_path.name
    return " ".join(w.capitalize() for w in raw.split("_"))


def _has_real_results(results_path: Path) -> bool:
    if not results_path.exists():
        return False
    df = pl.read_parquet(results_path)
    if "Position" not in df.columns:
        return False
    return df["Position"].drop_nulls().len() > 0


def enumerate_events(year: int, data_dir: Path) -> list[EventInfo]:
    """Events under ``data_dir/<year>`` that have at least one target session
    with real (non-placeholder) results on disk."""
    year_dir = data_dir / str(year)
    if not year_dir.is_dir():
        return []
    out: list[EventInfo] = []
    for event_path in sorted(year_dir.iterdir()):
        if not event_path.is_dir():
            continue
        try:
            event_id = int(event_path.name.split("_", 1)[0])
        except (ValueError, IndexError):
            continue
        sessions = [
            sess for sess in TARGET_SESSIONS
            if _has_real_results(event_path / sess / "results.parquet")
        ]
        if not sessions:
            continue
        out.append(EventInfo(
            year=year,
            event_id=event_id,
            event_name=_event_name(event_path),
            event_path=event_path,
            target_sessions=sessions,
        ))
    return out


def actual_top(
    event_path: Path, session: str, n: int
) -> list[str]:
    """Top-N drivers (Abbreviation) by Position for the given session, or []."""
    results_path = event_path / session / "results.parquet"
    if not _has_real_results(results_path):
        return []
    df = (
        pl.read_parquet(results_path)
        .filter(pl.col("Position").is_not_null())
        .sort("Position")
    )
    if "Abbreviation" not in df.columns:
        return []
    return df["Abbreviation"].head(n).to_list()


def run_backtest(
    model_dir: Path,
    data_dir: Path,
    store_db: Path,
    year: int,
    *,
    force: bool = False,
) -> None:
    """For every (event, target session) in ``year`` with real results, run a
    prediction (skipping ones already in the DB unless ``force=True``) and
    insert into the store. Failures are logged and skipped."""
    events = enumerate_events(year, data_dir)
    log.info("backtest scan: %d events with target results in %d", len(events), year)
    for ev in events:
        for sess in ev["target_sessions"]:
            existing = store.latest_for_race(
                store_db, ev["year"], ev["event_id"], sess
            )
            if existing is not None and not force:
                continue
            try:
                ordered, features = predict_session(
                    model_dir,
                    ev["year"],
                    ev["event_id"],
                    sess,
                    data_dir,
                    auto_download=False,
                )
            except (ValueError, RuntimeError, FileNotFoundError) as e:
                log.warning(
                    "backtest %s/%s/%s failed: %s",
                    ev["year"], ev["event_id"], sess, e,
                )
                continue
            store.insert_prediction(
                store_db,
                ev["year"],
                ev["event_id"],
                ev["event_name"],
                sess,
                ordered,
                features,
            )
            log.info(
                "backtest stored %s/%s/%s top: %s",
                ev["year"], ev["event_id"], sess,
                [d for d, _ in ordered[:3]],
            )
