"""APScheduler wiring for the recurring prediction job."""
from __future__ import annotations

import logging
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler

from webapp.runner import run_prediction

log = logging.getLogger(__name__)
_scheduler: BackgroundScheduler | None = None


def start(
    model_dir: Path,
    data_dir: Path,
    store_db: Path,
    interval_minutes: int = 60,
) -> None:
    global _scheduler
    if _scheduler is not None:
        return
    _scheduler = BackgroundScheduler()
    _scheduler.add_job(
        lambda: run_prediction(model_dir, data_dir, store_db),
        trigger="interval",
        minutes=interval_minutes,
        id="predict_next_session",
        coalesce=True,
        max_instances=1,
    )
    _scheduler.start()
    log.info("scheduler started; interval=%s min", interval_minutes)


def stop() -> None:
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
