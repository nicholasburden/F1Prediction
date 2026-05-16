"""Inference job invoked by the scheduler and the manual-refresh endpoint."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TypedDict

from scripts.inference import _next_upcoming_session, predict_session
from webapp import store

log = logging.getLogger(__name__)


class TargetSession(TypedDict):
    year: int
    event_id: int
    event_name: str
    session: str


def event_name(year: int, round_num: int) -> str:
    import fastf1

    schedule = fastf1.get_event_schedule(year, include_testing=False)
    rows = schedule[schedule["RoundNumber"] == round_num]
    if rows.empty:
        return f"Round {round_num}"
    return str(rows.iloc[0]["EventName"])


def next_target(model_dir: Path) -> TargetSession | None:
    """Return the next upcoming session in the model's target list, or None
    if the schedule lookup or the model config can't be read."""
    try:
        cfg = json.loads((model_dir / "config.json").read_text())
        year, round_num, session = _next_upcoming_session(
            cfg["training"]["target_sessions"]
        )
    except Exception as e:
        log.warning("could not determine next target: %s", e)
        return None
    return TargetSession(
        year=year,
        event_id=round_num,
        event_name=event_name(year, round_num),
        session=session,
    )


def run_prediction(
    model_dir: Path, data_dir: Path, store_db: Path
) -> dict[str, object] | None:
    """Predict the next upcoming session and persist the ordering. Returns a
    summary dict, or ``None`` if no prediction was producible (e.g. no
    upcoming session in the model's target list, or the next session has no
    preceding data on disk yet)."""
    cfg = json.loads((model_dir / "config.json").read_text())
    target_sessions = cfg["training"]["target_sessions"]
    try:
        year, round_num, session = _next_upcoming_session(target_sessions)
    except Exception as e:
        log.warning("no upcoming session found: %s", e)
        return None
    log.info("predicting %s round %s %s", year, round_num, session)
    try:
        ordered, features = predict_session(
            model_dir, year, round_num, session, data_dir, auto_download=True
        )
    except (ValueError, RuntimeError, FileNotFoundError) as e:
        log.info(
            "cannot predict %s round %s %s yet: %s",
            year, round_num, session, e,
        )
        return None
    name = event_name(year, round_num)
    store.insert_prediction(
        store_db, year, round_num, name, session, ordered, features
    )
    return {
        "year": year,
        "round": round_num,
        "session": session,
        "event_name": name,
        "n_drivers": len(ordered),
    }
