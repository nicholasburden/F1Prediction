"""Inference + data-sync + retrain jobs invoked by the scheduler and the
status endpoints. Shared retrain state lives here so both the FastAPI app
(reading it from the event loop) and the APScheduler worker thread (writing
it during a daily-sync-triggered retrain) can access it under a single lock."""
from __future__ import annotations

import json
import logging
import os
import shutil
import threading
from datetime import datetime, timezone
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


def scheduled_sessions(year: int, round_num: int) -> set[str]:
    """Set of session abbreviations ('FP1', 'Q', 'Sprint', 'R', …) that the
    fastf1 schedule lists for this event. Used to filter prediction targets so
    we don't conjure a Sprint prediction for a non-Sprint weekend, etc.
    Returns the empty set if the event isn't on the schedule."""
    import fastf1

    from f1prediction.data.download import SESSION_ABBREVS

    schedule = fastf1.get_event_schedule(year, include_testing=False)
    rows = schedule[schedule["RoundNumber"] == round_num]
    if rows.empty:
        return set()
    row = rows.iloc[0]
    out: set[str] = set()
    for i in range(1, 6):
        name = row.get(f"Session{i}")
        if isinstance(name, str) and name.strip() in SESSION_ABBREVS:
            out.add(SESSION_ABBREVS[name.strip()])
    return out


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
        with _inference_lock:
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
        store_db, year, round_num, name, session, ordered, features,
        model_fit_at(model_dir, data_dir),
    )
    return {
        "year": year,
        "round": round_num,
        "session": session,
        "event_name": name,
        "n_drivers": len(ordered),
    }


def run_event_predictions(
    model_dir: Path,
    data_dir: Path,
    store_db: Path,
    year: int,
    event_id: int,
    *,
    force: bool = False,
) -> list[dict[str, object]]:
    """Predict every model-target session for the given event and store the
    orderings. Skips sessions already in the DB unless ``force`` is set;
    sessions that can't be predicted (missing preceding data) are logged and
    skipped. Used so the fantasy optimiser sees R + Q + Sprint together."""
    cfg = json.loads((model_dir / "config.json").read_text())
    target_sessions: list[str] = cfg["training"]["target_sessions"]
    on_schedule = scheduled_sessions(year, event_id)
    if on_schedule:
        target_sessions = [s for s in target_sessions if s in on_schedule]
    name = event_name(year, event_id)
    out: list[dict[str, object]] = []
    for session in target_sessions:
        if not force and store.latest_for_race(
            store_db, year, event_id, session
        ) is not None:
            continue
        log.info("predicting %s round %s %s", year, event_id, session)
        try:
            with _inference_lock:
                ordered, features = predict_session(
                    model_dir, year, event_id, session, data_dir, auto_download=True
                )
        except (ValueError, RuntimeError, FileNotFoundError) as e:
            log.info(
                "cannot predict %s round %s %s yet: %s",
                year, event_id, session, e,
            )
            continue
        store.insert_prediction(
            store_db, year, event_id, name, session, ordered, features,
            model_fit_at(model_dir, data_dir),
        )
        out.append({"session": session, "n_drivers": len(ordered)})
    return out


def _existing_session_dirs(year_dir: Path) -> set[Path]:
    if not year_dir.exists():
        return set()
    return {p for p in year_dir.glob("*/*") if p.is_dir()}


def download_new_data(data_dir: Path, cache_dir: Path) -> list[Path]:
    """Pull any newly-completed F1 sessions for the current calendar year and
    return the list of session directories that appeared on disk (empty if
    nothing new). ``skip_existing=True`` keeps re-runs cheap; errors are
    logged and swallowed — the daily job must never crash the scheduler."""
    from datetime import date

    from f1prediction.data.download import download_history

    year = date.today().year
    year_dir = data_dir / str(year)
    before = _existing_session_dirs(year_dir)
    record_progress(f"checking for new {year} data")
    try:
        download_history(
            start_year=year,
            end_year=year,
            data_dir=data_dir,
            cache_dir=cache_dir,
            skip_existing=True,
        )
    except Exception as e:
        log.exception("data sync failed")
        record_progress(f"data check failed: {type(e).__name__}: {e}")
        return []
    new = sorted(_existing_session_dirs(year_dir) - before)
    if new:
        names = ", ".join(p.parent.name + "/" + p.name for p in new[:3])
        suffix = f" (+{len(new) - 3} more)" if len(new) > 3 else ""
        record_progress(f"downloaded {len(new)} new session(s): {names}{suffix}")
    else:
        record_progress("no new sessions")
    return new


def is_full_data(model_dir: Path) -> bool:
    """True if the deployed model's config was trained on the full dataset
    (no held-out split). False both for missing/unreadable configs and for
    val-split runs — i.e. anything where retraining would meaningfully change
    the model the webapp serves."""
    try:
        cfg = json.loads((model_dir / "config.json").read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return bool(cfg.get("training", {}).get("full_data"))


def retrain_and_deploy(model_dir: Path) -> Path:
    """Run a full two-stage refit using ``model_dir/config.json`` as the base
    config (stage 1 picks an epoch count on the val split, stage 2 fits on all
    data), then atomically swap the new ``checkpoint.pt`` and ``config.json``
    into ``model_dir`` so subsequent ``predict_session`` calls use them.
    ``shutil.copy2`` preserves the source mtime, so the deployed checkpoint's
    mtime is the training time — that's what the UI displays as the fit date.
    Returns the run dir the full-data model was saved to."""
    from f1prediction.training.refit import full_refit

    record_progress("starting refit")
    new_run_dir = full_refit(
        model_dir, cross_validation=True, on_phase=record_progress,
    )
    record_progress("deploying new model")
    for name in ("checkpoint.pt", "config.json"):
        src = new_run_dir / name
        dst = model_dir / name
        tmp = dst.with_suffix(dst.suffix + ".new")
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
    record_progress("model deployed")
    return new_run_dir


def event_key(year: int, round_num: int) -> str:
    """Stable, sortable directory name for a per-event model snapshot."""
    return f"{year}_{round_num:02d}"


def per_event_model_dir(models_root: Path, year: int, round_num: int) -> Path:
    """``<models_root>/<year>_<round>/`` — the directory holding a snapshot
    of the model that should be used to predict event ``(year, round_num)``.
    Doesn't create the directory."""
    return models_root / event_key(year, round_num)


def _last_event_on_disk(
    data_dir: Path,
    years: list[int],
    cutoff: tuple[int, int] | None,
) -> tuple[int, int] | None:
    """Max (year, round) event dir on disk that's within ``years``, strictly
    before ``cutoff`` if set, and has at least one real-results parquet
    (filters out fastf1's pre-session placeholder dirs)."""
    from webapp.backtest import _has_real_results

    candidates: list[tuple[int, int]] = []
    for year in years:
        year_dir = data_dir / str(year)
        if not year_dir.is_dir():
            continue
        for event_path in year_dir.iterdir():
            if not event_path.is_dir():
                continue
            try:
                round_num = int(event_path.name.split("_", 1)[0])
            except (ValueError, IndexError):
                continue
            if cutoff is not None and (year, round_num) >= cutoff:
                continue
            if any(
                _has_real_results(event_path / s / "results.parquet")
                for s in ("R", "Sprint", "Q")
            ):
                candidates.append((year, round_num))
    return max(candidates) if candidates else None


def _event_race_date(year: int, round_num: int) -> str | None:
    """Race date (YYYY-MM-DD) of an event from fastf1's schedule; None on
    miss or error."""
    try:
        import fastf1

        schedule = fastf1.get_event_schedule(year, include_testing=False)
        rows = schedule[schedule["RoundNumber"] == round_num]
        if rows.empty:
            return None
        date = rows.iloc[0].get("EventDate")
        if date is None:
            return None
        return str(date)[:10]
    except Exception:
        return None


def model_fit_at(model_dir: Path, data_dir: Path) -> str | None:
    """Date (YYYY-MM-DD) of the last event the model saw during training —
    i.e. the race date of the latest event in scope per the config's years
    and event_cutoff that actually has results on disk. Despite the legacy
    name, this is *not* the training timestamp; the user-facing label is
    'model fit through <last event date>'. Returns None if any lookup
    fails."""
    try:
        cfg = json.loads((model_dir / "config.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None
    years = cfg["training"].get("years", [])
    raw_cutoff = cfg["training"].get("event_cutoff")
    cutoff = tuple(raw_cutoff) if isinstance(raw_cutoff, (list, tuple)) else None
    if cutoff is not None and len(cutoff) != 2:
        cutoff = None
    last = _last_event_on_disk(data_dir, years, cutoff)  # type: ignore[arg-type]
    if last is None:
        return None
    return _event_race_date(*last)


class RetrainState(TypedDict):
    running: bool
    started_at: str | None
    finished_at: str | None
    error: str | None
    run_dir: str | None


class ProgressEntry(TypedDict):
    ts: str
    message: str


_retrain_state: RetrainState = {
    "running": False,
    "started_at": None,
    "finished_at": None,
    "error": None,
    "run_dir": None,
}
_retrain_lock = threading.Lock()

_progress_entries: list[ProgressEntry] = []
_progress_lock = threading.Lock()
_PROGRESS_MAX = 50

# Global mutex around inference. predict_session writes synthetic
# weather/results placeholders into the shared data/ tree for the event it's
# predicting, then unlinks them in its finally block. Another inference
# running concurrently (e.g. backtest job alongside next-event prediction at
# startup) can have its lazy parquet scan pick up one of these placeholders,
# only for the read to error after the first call's cleanup runs. Serialising
# inference is cheaper than per-event temp dirs and avoids that whole class
# of race.
_inference_lock = threading.Lock()


def record_progress(message: str) -> None:
    """Append a user-facing progress line to the rolling retrain/download log.
    Thread-safe; also forwarded to the Python logger so it shows up in
    server logs. Buffer trimmed to ``_PROGRESS_MAX`` entries."""
    ts = datetime.now(timezone.utc).isoformat()
    log.info("[progress] %s", message)
    with _progress_lock:
        _progress_entries.append({"ts": ts, "message": message})
        excess = len(_progress_entries) - _PROGRESS_MAX
        if excess > 0:
            del _progress_entries[:excess]


def get_progress() -> list[ProgressEntry]:
    """Read-only snapshot of recent progress entries (oldest first)."""
    with _progress_lock:
        return list(_progress_entries)


def retrain_state_snapshot() -> RetrainState:
    """Read-only snapshot of the shared retrain state."""
    with _retrain_lock:
        return RetrainState(**_retrain_state)


def _try_start_retrain() -> bool:
    """Atomically claim the retrain slot. Returns False if one is already
    in flight (caller should not start another)."""
    with _retrain_lock:
        if _retrain_state["running"]:
            return False
        _retrain_state["running"] = True
        _retrain_state["started_at"] = datetime.now(timezone.utc).isoformat()
        _retrain_state["finished_at"] = None
        _retrain_state["error"] = None
        _retrain_state["run_dir"] = None
    return True


def _run_retrain(model_dir: Path) -> None:
    """Synchronously run retrain_and_deploy and keep the shared state honest.
    Caller must have acquired the slot via ``_try_start_retrain`` first."""
    try:
        run_dir = retrain_and_deploy(model_dir)
        with _retrain_lock:
            _retrain_state["run_dir"] = str(run_dir)
            _retrain_state["error"] = None
    except Exception as e:
        log.exception("retrain failed")
        with _retrain_lock:
            _retrain_state["error"] = f"{type(e).__name__}: {e}"
    finally:
        with _retrain_lock:
            _retrain_state["running"] = False
            _retrain_state["finished_at"] = datetime.now(timezone.utc).isoformat()


def download_and_maybe_retrain(
    model_dir: Path, data_dir: Path, cache_dir: Path
) -> None:
    """Daily job: pull new sessions; if anything appeared on disk, retrain
    the model and atomically swap it in. If a retrain is already running, the
    download still happens but the retrain step is skipped (next day's run
    will catch up)."""
    new_sessions = download_new_data(data_dir, cache_dir)
    if not new_sessions:
        return
    if not _try_start_retrain():
        record_progress(
            f"{len(new_sessions)} new session(s) but a refit is already in flight"
        )
        return
    record_progress("new data found — triggering automatic refit")
    _run_retrain(model_dir)


def kick_off_retrain(model_dir: Path) -> bool:
    """Synchronously run a refit on the caller's thread iff none is already in
    flight; returns False (no-op) if one was already running. Designed for
    executor dispatch from the on-demand FastAPI endpoint — the scheduler
    path uses ``download_and_maybe_retrain`` instead so it can short-circuit
    on the no-new-data case."""
    if not _try_start_retrain():
        record_progress("on-demand refit ignored: one already in flight")
        return False
    record_progress("on-demand refit triggered")
    _run_retrain(model_dir)
    return True
