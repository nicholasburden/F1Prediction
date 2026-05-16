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

from webapp import store
from webapp.feature_meta import categorise
from webapp.runner import next_target, run_prediction
from webapp.scheduler import start as scheduler_start, stop as scheduler_stop

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("webapp")

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "webapp_config"
DATA_DIR = ROOT / "data"
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


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    store.init_db(DB_PATH)
    scheduler_start(MODEL_DIR, DATA_DIR, DB_PATH)
    asyncio.get_running_loop().run_in_executor(
        None, run_prediction, MODEL_DIR, DATA_DIR, DB_PATH
    )
    log.info("startup complete; one-shot prediction running in background")
    try:
        yield
    finally:
        scheduler_stop()


app = FastAPI(lifespan=lifespan, title="F1 Prediction")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


def _enrich(pred: store.PredictionRow | None) -> list[dict[str, object]]:
    if pred is None:
        return []
    drivers = _load_drivers()
    enriched: list[dict[str, object]] = []
    for rank, (abbrev, p) in enumerate(pred["drivers"], start=1):
        info = drivers.get(abbrev, DriverInfo(full_name=None, team=None, image=None))
        enriched.append({
            "rank": rank,
            "abbrev": abbrev,
            "full_name": info.get("full_name") or abbrev,
            "team": info.get("team") or "Unknown",
            "image": info.get("image"),
            "pred": p,
        })
    return enriched


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

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "pred": pred,
            "items": _enrich(pred),
            "inputs": inputs,
            "target": target,
            "target_has_pred": target_pred is not None,
            "explicit": explicit,
            "selected_year": year,
            "selected_event": event,
            "selected_session": session,
            "history": store.distinct_races(DB_PATH),
        },
    )


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
