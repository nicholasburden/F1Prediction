"""FastAPI application: JSON API + server-rendered podium UI."""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, AsyncIterator, TypedDict

from fastapi import BackgroundTasks, FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from webapp import fantasy, store
from webapp.backtest import actual_top, enumerate_events, run_backtest
from webapp.feature_meta import categorise
from webapp.runner import (
    get_progress,
    is_full_data,
    kick_off_retrain,
    model_fit_at,
    next_target,
    retrain_state_snapshot,
    run_event_predictions,
    run_prediction,
)
from webapp.scheduler import start as scheduler_start, stop as scheduler_stop

SEASON_YEAR = 2026

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("webapp")

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "webapp_config"
MODELS_DIR = ROOT / "webapp_models"
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "cache"
DB_PATH = ROOT / "webapp.db"
WEBAPP_DIR = Path(__file__).resolve().parent
STATIC_DIR = WEBAPP_DIR / "static"
TEMPLATES_DIR = WEBAPP_DIR / "templates"
DRIVERS_JSON = MODEL_DIR / "drivers.json"


class DriverInfo(TypedDict):
    full_name: str | None
    team: str | None
    image: str | None


def _load_drivers() -> dict[str, DriverInfo]:
    return json.loads(DRIVERS_JSON.read_text())


def _predict_all_sessions_of_next_event(
    model_dir: Path, data_dir: Path, db_path: Path
) -> None:
    """Background task at startup: regenerate every target-session prediction
    for the next-upcoming event so the home page reflects the *current*
    deployed model (including its fit-date label) rather than whatever was
    cached from a previous run with an older model."""
    tgt = next_target(model_dir)
    if tgt is None:
        return
    run_event_predictions(
        model_dir, data_dir, db_path, tgt["year"], tgt["event_id"], force=True,
    )


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    store.init_db(DB_PATH)
    scheduler_start(MODEL_DIR, DATA_DIR, CACHE_DIR, DB_PATH)
    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, run_prediction, MODEL_DIR, DATA_DIR, DB_PATH)
    loop.run_in_executor(
        None, _predict_all_sessions_of_next_event, MODEL_DIR, DATA_DIR, DB_PATH
    )
    loop.run_in_executor(
        None,
        lambda: run_backtest(
            MODEL_DIR, DATA_DIR, DB_PATH, SEASON_YEAR,
            models_dir=MODELS_DIR, force=True,
        ),
    )
    log.info(
        "startup complete; one-shot prediction + %d-season backtest running "
        "in background", SEASON_YEAR,
    )
    try:
        yield
    finally:
        scheduler_stop()


app = FastAPI(lifespan=lifespan, title="F1 Prediction")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)
templates.env.globals["asset_version"] = str(
    int((STATIC_DIR / "style.css").stat().st_mtime)
)


def _driver_info(abbrev: str, drivers: dict[str, DriverInfo]) -> dict[str, object]:
    info = drivers.get(abbrev, DriverInfo(full_name=None, team=None, image=None))
    return {
        "abbrev": abbrev,
        "full_name": info.get("full_name") or abbrev,
        "team": info.get("team") or "Unknown",
        "image": info.get("image"),
    }


def _enrich(pred: store.PredictionRow | None) -> list[dict[str, object]]:
    if pred is None:
        return []
    drivers = _load_drivers()
    enriched: list[dict[str, object]] = []
    for rank, (abbrev, p) in enumerate(pred["drivers"], start=1):
        enriched.append({**_driver_info(abbrev, drivers), "rank": rank, "pred": p})
    return enriched


_SESSION_PRIORITY = ("R", "Sprint", "Q", "FP3", "FP2", "FP1", "SQ")


def _default_session(sessions: list[str]) -> str | None:
    if not sessions:
        return None
    for s in _SESSION_PRIORITY:
        if s in sessions:
            return s
    return sessions[0]


def _events_with_sessions(
    history: list[store.RaceSummary],
    target: dict[str, object] | None,
) -> list[dict[str, object]]:
    """Distinct (year, event_id, event_name) groups with their available
    session list, derived from the predictions DB plus the next-upcoming
    target (so a freshly-scheduled event shows up even before any prediction
    exists)."""
    seen: dict[tuple[int, int], dict[str, object]] = {}
    for r in history:
        key = (r["year"], r["event_id"])
        entry = seen.setdefault(key, {
            "year": r["year"],
            "event_id": r["event_id"],
            "event_name": r["event_name"],
            "sessions": [],
        })
        if r["session"] not in entry["sessions"]:  # type: ignore[operator]
            entry["sessions"].append(r["session"])  # type: ignore[union-attr]
    if target is not None:
        key = (target["year"], target["event_id"])  # type: ignore[index]
        entry = seen.setdefault(key, {
            "year": target["year"],
            "event_id": target["event_id"],
            "event_name": target["event_name"],
            "sessions": [],
        })
        if target["session"] not in entry["sessions"]:  # type: ignore[operator]
            entry["sessions"].append(target["session"])  # type: ignore[union-attr]
    return sorted(  # type: ignore[return-value]
        seen.values(),
        key=lambda e: (e["year"], e["event_id"]),  # type: ignore[arg-type]
        reverse=True,
    )


def _actual_podium(
    year: int, event_id: int, session: str, pred: store.PredictionRow | None
) -> list[dict[str, object]]:
    """Top-3 finishers (enriched, with a ``predicted`` flag set if the driver
    was in the model's predicted top 3) for the given session. Returns [] if
    no real results are on disk."""
    for ev in enumerate_events(year, DATA_DIR):
        if ev["event_id"] != event_id or ev["year"] != year:
            continue
        if session not in ev["target_sessions"]:
            return []
        abbrevs = actual_top(ev["event_path"], session, 3)
        if not abbrevs:
            return []
        pred_top3 = (
            {d for d, _ in pred["drivers"][:3]} if pred is not None else set()
        )
        drivers = _load_drivers()
        return [
            {**_driver_info(a, drivers), "rank": i + 1,
             "predicted": a in pred_top3}
            for i, a in enumerate(abbrevs)
        ]
    return []


@app.get("/", response_class=HTMLResponse)
def index(
    request: Request,
    year: Annotated[int | None, Query()] = None,
    event: Annotated[int | None, Query()] = None,
    session: Annotated[str | None, Query()] = None,
) -> HTMLResponse:
    target = next_target(MODEL_DIR)
    target_pred = None
    if target is not None:
        target_pred = store.latest_for_race(
            DB_PATH, target["year"], target["event_id"], target["session"]
        )

    history = store.distinct_races(DB_PATH)
    events = _events_with_sessions(history, target)

    # An event was picked but no session — default to the most "interesting"
    # session of that event (R > Sprint > Q > FP3 > FP2 > FP1 > SQ).
    selected_event_entry = None
    if year is not None and event is not None:
        for e in events:
            if e["year"] == year and e["event_id"] == event:
                selected_event_entry = e
                break
        if session is None and selected_event_entry is not None:
            session = _default_session(selected_event_entry["sessions"])  # type: ignore[arg-type]

    explicit = year is not None and event is not None and session is not None
    pred = (
        store.latest_for_race(DB_PATH, year, event, session)  # type: ignore[arg-type]
        if explicit
        else target_pred
    )
    inputs: list[object] = []
    if pred is not None and pred.get("features"):
        cfg = json.loads((MODEL_DIR / "config.json").read_text())
        inputs = categorise(pred["features"], cfg["training"]["feature_sets"])  # type: ignore[assignment]

    actual_podium: list[dict[str, object]] = []
    if pred is not None:
        actual_podium = _actual_podium(
            pred["year"], pred["event_id"], pred["session"], pred
        )

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "active_page": "predictions",
            "pred": pred,
            "items": _enrich(pred),
            "inputs": inputs,
            "target": target,
            "target_has_pred": target_pred is not None,
            "explicit": explicit,
            "selected_year": year,
            "selected_event": event,
            "selected_session": session,
            "selected_event_entry": selected_event_entry,
            "events": events,
            "actual_podium": actual_podium,
            "model_full_data": is_full_data(MODEL_DIR),
            "model_fit_at": model_fit_at(MODEL_DIR, DATA_DIR),
            "retrain_running": retrain_state_snapshot()["running"],
            "latest_progress": (get_progress()[-1]["message"] if get_progress() else None),
        },
    )


def _event_for_fantasy(
    year: int | None, event_id: int | None
) -> tuple[int, int, str] | None:
    """Resolve which event the fantasy page should optimise for.

    Priority: explicit (year, event_id) from query string > next upcoming
    target session > most recently predicted event in the DB.
    """
    if year is not None and event_id is not None:
        for r in store.distinct_races(DB_PATH):
            if r["year"] == year and r["event_id"] == event_id:
                return year, event_id, r["event_name"]
        return year, event_id, f"Round {event_id}"
    tgt = next_target(MODEL_DIR)
    if tgt is not None:
        return tgt["year"], tgt["event_id"], tgt["event_name"]
    history = store.distinct_races(DB_PATH)
    if history:
        r = history[0]
        return r["year"], r["event_id"], r["event_name"]
    return None


@app.get("/fantasy", response_class=HTMLResponse)
def fantasy_page(
    request: Request,
    background: BackgroundTasks,
    budget: Annotated[float, Query()] = fantasy.DEFAULT_BUDGET,
    year: Annotated[int | None, Query()] = None,
    event: Annotated[int | None, Query()] = None,
    restriction: Annotated[str, Query()] = fantasy.RESTRICTION_NONE,
) -> HTMLResponse:
    if restriction not in fantasy.RESTRICTIONS:
        restriction = fantasy.RESTRICTION_NONE
    target = next_target(MODEL_DIR)
    history = store.distinct_races(DB_PATH)
    events = _events_with_sessions(history, target)  # type: ignore[arg-type]

    resolved = _event_for_fantasy(year, event)
    feed_error: str | None = None
    feed: fantasy.FantasyFeed | None = None
    try:
        feed = fantasy.fetch_fantasy_feed()
    except (OSError, ValueError, json.JSONDecodeError) as e:
        log.warning("could not fetch fantasy feed: %s", e)
        feed_error = str(e)

    pick: fantasy.TeamPick | None = None
    driver_points: dict[str, float] = {}
    driver_points_by_session: dict[str, dict[str, float]] = {}
    constructor_points: dict[str, float] = {}
    sessions_used: list[str] = []
    sessions_missing: list[str] = []
    sel_year: int | None = None
    sel_event_id: int | None = None
    sel_event_name: str | None = None

    if resolved is not None:
        sel_year, sel_event_id, sel_event_name = resolved
        if feed is not None:
            driver_points, driver_points_by_session = fantasy.expected_driver_points(
                DB_PATH, sel_year, sel_event_id
            )
            sessions_used = sorted({
                sess for breakdown in driver_points_by_session.values() for sess in breakdown
            })
            sessions_missing = [
                s for s in fantasy.SESSION_POINTS if s not in sessions_used
            ]
            if sessions_missing:
                background.add_task(
                    run_event_predictions,
                    MODEL_DIR, DATA_DIR, DB_PATH, sel_year, sel_event_id,
                )
            constructor_points = fantasy.expected_constructor_points(
                driver_points, fantasy.drivers_by_team(feed["drivers"]),
            )
            if driver_points:
                pick = fantasy.optimise_team(
                    feed, driver_points, constructor_points, budget,
                    restriction=restriction,
                )

    drivers_meta = _load_drivers() if feed is not None else {}
    driver_rows: list[dict[str, object]] = []
    if feed is not None:
        for d in sorted(feed["drivers"], key=lambda d: -driver_points.get(d["tla"], 0.0)):
            tla = d["tla"]
            in_team = pick is not None and tla in pick["drivers"]
            driver_rows.append({
                **_driver_info(tla, drivers_meta),
                "price": d["price"],
                "points": driver_points.get(tla, 0.0),
                "breakdown": driver_points_by_session.get(tla, {}),
                "in_team": in_team,
                "is_drs": pick is not None and pick["drs_driver"] == tla,
                "value": (
                    driver_points.get(tla, 0.0) / d["price"] if d["price"] > 0 else 0.0
                ),
            })

    constructor_rows: list[dict[str, object]] = []
    if feed is not None:
        for c in sorted(feed["constructors"], key=lambda c: -constructor_points.get(c["name"], 0.0)):
            name = c["name"]
            in_team = pick is not None and name in pick["constructors"]
            constructor_rows.append({
                "name": name,
                "price": c["price"],
                "points": constructor_points.get(name, 0.0),
                "in_team": in_team,
                "value": (
                    constructor_points.get(name, 0.0) / c["price"] if c["price"] > 0 else 0.0
                ),
            })

    return templates.TemplateResponse(
        request,
        "fantasy.html",
        {
            "active_page": "fantasy",
            "budget": budget,
            "feed": feed,
            "feed_error": feed_error,
            "pick": pick,
            "driver_rows": driver_rows,
            "constructor_rows": constructor_rows,
            "sessions_used": sessions_used,
            "sessions_missing": sessions_missing,
            "all_target_sessions": list(fantasy.SESSION_POINTS.keys()),
            "restriction": restriction,
            "restriction_options": [
                (key, fantasy.RESTRICTION_LABELS[key]) for key in fantasy.RESTRICTIONS
            ],
            "premium_threshold": fantasy.PREMIUM_THRESHOLD,
            "selected_year": sel_year,
            "selected_event": sel_event_id,
            "selected_event_name": sel_event_name,
            "events": events,
            "drs_multiplier": fantasy.DRS_MULTIPLIER,
            "n_drivers": fantasy.TEAM_DRIVERS,
            "n_constructors": fantasy.TEAM_CONSTRUCTORS,
            "model_full_data": is_full_data(MODEL_DIR),
            "model_fit_at": model_fit_at(MODEL_DIR, DATA_DIR),
            "retrain_running": retrain_state_snapshot()["running"],
            "latest_progress": (get_progress()[-1]["message"] if get_progress() else None),
        },
    )


@app.post("/api/fantasy/refresh", status_code=202)
def api_fantasy_refresh() -> dict[str, object]:
    try:
        feed = fantasy.fetch_fantasy_feed(force=True)
    except (OSError, ValueError, json.JSONDecodeError) as e:
        return {"status": "error", "error": str(e)}
    return {
        "status": "ok",
        "drivers": len(feed["drivers"]),
        "constructors": len(feed["constructors"]),
    }


@app.get("/api/latest")
def api_latest() -> JSONResponse:
    pred = store.latest(DB_PATH)
    return JSONResponse(pred or {})


@app.get("/api/history")
def api_history() -> JSONResponse:
    return JSONResponse(store.distinct_races(DB_PATH))


@app.post("/api/refresh", status_code=202)
def api_refresh(background: BackgroundTasks) -> dict[str, str]:
    background.add_task(run_prediction, MODEL_DIR, DATA_DIR, DB_PATH)
    return {"status": "scheduled"}


@app.post("/api/retrain")
async def api_retrain() -> JSONResponse:
    """On-demand refit: dispatched to the executor so the request returns
    immediately. The runner's lock protects against double-starts; if a
    retrain (scheduler- or user-triggered) is already in flight, returns
    409. The pill polls ``/api/retrain/status`` to track completion."""
    snap = retrain_state_snapshot()
    if snap["running"]:
        return JSONResponse(
            {"status": "already_running", **dict(snap)}, status_code=409,
        )
    asyncio.get_running_loop().run_in_executor(None, kick_off_retrain, MODEL_DIR)
    return JSONResponse({"status": "scheduled"}, status_code=202)


@app.get("/api/retrain/status")
def api_retrain_status() -> JSONResponse:
    """Polled by the topbar fit-date pill while a retrain is in flight, so
    the UI can refresh once it lands. Also returns the rolling progress log
    so the pill can show what phase the refit is in."""
    state: dict[str, object] = dict(retrain_state_snapshot())
    state["model_full_data"] = is_full_data(MODEL_DIR)
    state["model_fit_at"] = model_fit_at(MODEL_DIR, DATA_DIR)
    state["progress"] = get_progress()
    return JSONResponse(state)
