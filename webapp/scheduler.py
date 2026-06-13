"""APScheduler wiring for the recurring prediction and data-sync jobs."""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler

from webapp.runner import download_and_maybe_retrain, run_prediction

log = logging.getLogger(__name__)
_scheduler: BackgroundScheduler | None = None


def start(
    model_dir: Path,
    data_dir: Path,
    cache_dir: Path,
    store_db: Path,
    interval_minutes: int = 60,
    download_interval_hours: int = 24,
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
    _scheduler.add_job(
        lambda: download_and_maybe_retrain(model_dir, data_dir, cache_dir, store_db),
        trigger="interval",
        hours=download_interval_hours,
        id="daily_data_download",
        coalesce=True,
        max_instances=1,
        next_run_time=datetime.now(),
    )
    _scheduler.start()
    log.info(
        "scheduler started; predict interval=%s min, download interval=%s h",
        interval_minutes, download_interval_hours,
    )


def stop() -> None:
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
