#!/usr/bin/env python
"""Overnight F1 data download with exponential backoff and rate limiting.

Wraps the core download logic with:
  - Exponential backoff + jitter on failures (per session)
  - Configurable inter-session delay to avoid hammering the API
  - File + console logging so you can check progress in the morning
  - Progress summary at exit (even on Ctrl-C)

Usage:
    uv run python scripts/download_overnight.py
    uv run python scripts/download_overnight.py --start-year 2018 --end-year 2024
    uv run python scripts/download_overnight.py --delay 3.0 --max-retries 5
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from f1prediction.data.download import (
    SESSION_ABBREVS,
    _event_slug,
    _save_laps,
    _save_results,
    _save_weather,
    _session_dir_exists,
)

# ---------------------------------------------------------------------------
# Stats tracking
# ---------------------------------------------------------------------------


@dataclass
class DownloadStats:
    sessions_attempted: int = 0
    sessions_ok: int = 0
    sessions_skipped: int = 0
    sessions_failed: int = 0
    total_retries: int = 0
    failed_sessions: list[str] = field(default_factory=list)

    def report(self, logger: logging.Logger) -> None:
        logger.info("=" * 60)
        logger.info("DOWNLOAD SUMMARY")
        logger.info("  Attempted : %d", self.sessions_attempted)
        logger.info("  Succeeded : %d", self.sessions_ok)
        logger.info("  Skipped   : %d", self.sessions_skipped)
        logger.info("  Failed    : %d", self.sessions_failed)
        logger.info("  Retries   : %d", self.total_retries)
        if self.failed_sessions:
            logger.info("Failed sessions:")
            for s in self.failed_sessions:
                logger.info("  - %s", s)
        logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Backoff helper
# ---------------------------------------------------------------------------


_RATE_LIMIT_MARKERS = ("500 calls", "rate limit", "429", "too many requests")


def _is_rate_limited(exc: Exception) -> bool:
    return any(m in str(exc).lower() for m in _RATE_LIMIT_MARKERS)


def backoff_sleep(attempt: int, base: float = 5.0, cap: float = 300.0) -> float:
    """Exponential backoff with full jitter. Returns actual sleep duration."""
    delay = min(cap, base * (2**attempt))
    jitter = random.uniform(0, delay)
    time.sleep(jitter)
    return jitter


def rate_limit_sleep(wait: float = 900.0) -> None:
    """Sleep after hitting a rate limit, logging progress every minute."""
    logger.warning("Rate limit hit — pausing %.0f minutes before resuming.", wait / 60)
    end = time.monotonic() + wait
    while True:
        remaining = end - time.monotonic()
        if remaining <= 0:
            break
        logger.info("  ... resuming in %.0f minutes", remaining / 60)
        time.sleep(min(60.0, remaining))


# ---------------------------------------------------------------------------
# Session download with retry
# ---------------------------------------------------------------------------

# Re-use column lists and helpers from the library module


logger = logging.getLogger("download_overnight")


def _download_session_with_retry(
    fastf1,
    year: int,
    round_num: int,
    slug: str,
    sess_name: str,
    abbrev: str,
    event_dir: Path,
    *,
    max_retries: int,
    base_backoff: float,
    inter_delay: float,
    stats: DownloadStats,
) -> bool:
    """Try to download one session, with exponential backoff on failure.

    Returns True if successful, False if all retries exhausted.
    """
    out_dir = event_dir / abbrev
    label = f"{year}/{slug}/{abbrev}"

    if _session_dir_exists(out_dir):
        logger.info("    [SKIP] %s (already on disk)", label)
        stats.sessions_skipped += 1
        return True

    stats.sessions_attempted += 1

    for attempt in range(max_retries + 1):
        try:
            logger.info(
                "    [%s] Attempt %d/%d ...",
                label,
                attempt + 1,
                max_retries + 1,
            )
            session = fastf1.get_session(year, round_num, sess_name)
            session.load(laps=True, telemetry=False, weather=True, messages=False)

            _save_results(session, out_dir)
            _save_laps(session, out_dir)
            _save_weather(session, out_dir)

            logger.info("    [OK] %s", label)
            stats.sessions_ok += 1

            if inter_delay > 0:
                time.sleep(inter_delay)

            return True

        except KeyboardInterrupt:
            raise

        except Exception as exc:
            if attempt < max_retries:
                stats.total_retries += 1
                if _is_rate_limited(exc):
                    rate_limit_sleep()
                else:
                    slept = backoff_sleep(attempt, base=base_backoff)
                    logger.warning(
                        "    [RETRY] %s — %s (sleeping %.1fs before retry %d)",
                        label,
                        exc,
                        slept,
                        attempt + 2,
                    )
            else:
                logger.error("    [FAIL] %s — %s (all retries exhausted)", label, exc)
                stats.sessions_failed += 1
                stats.failed_sessions.append(label)
                return False

    return False  # unreachable


# ---------------------------------------------------------------------------
# Main download loop
# ---------------------------------------------------------------------------


def download_overnight(
    start_year: int,
    end_year: int,
    data_dir: Path,
    cache_dir: Path,
    skip_existing: bool = True,
    inter_delay: float = 30.0,
    max_retries: int = 4,
    base_backoff: float = 10.0,
) -> DownloadStats:
    """Download F1 data with backoff, returning a stats summary."""
    import fastf1

    cache_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))

    stats = DownloadStats()

    for year in range(start_year, end_year + 1):
        logger.info("=== Year %d ===", year)

        schedule = None
        for sched_attempt in range(max_retries + 1):
            try:
                schedule = fastf1.get_event_schedule(year, include_testing=False)
                break
            except Exception as exc:
                if sched_attempt < max_retries:
                    if _is_rate_limited(exc):
                        rate_limit_sleep()
                    else:
                        slept = backoff_sleep(sched_attempt, base=base_backoff)
                        logger.warning(
                            "Schedule fetch for %d failed: %s (retry in %.1fs)",
                            year,
                            exc,
                            slept,
                        )
                else:
                    logger.error(
                        "Could not fetch schedule for %d after retries: %s", year, exc
                    )
        if schedule is None:
            continue

        for _, event_row in schedule.iterrows():
            round_num: int = int(event_row.get("RoundNumber") or 0)
            event_name: str = str(event_row.get("EventName", f"round{round_num}"))
            slug = _event_slug(round_num, event_name)
            event_dir = data_dir / str(year) / slug
            logger.info("  Event %02d: %s", round_num, event_name)

            # Determine sessions from schedule columns Session1..Session5
            session_names: list[str] = []
            for i in range(1, 6):
                val = event_row.get(f"Session{i}")
                if val and isinstance(val, str) and val.strip():
                    session_names.append(val.strip())
            if not session_names:
                session_names = [
                    "Practice 1",
                    "Practice 2",
                    "Practice 3",
                    "Qualifying",
                    "Race",
                ]

            for sess_name in session_names:
                abbrev = SESSION_ABBREVS.get(sess_name)
                if abbrev is None:
                    continue
                try:
                    _download_session_with_retry(
                        fastf1,
                        year=year,
                        round_num=round_num,
                        slug=slug,
                        sess_name=sess_name,
                        abbrev=abbrev,
                        event_dir=event_dir,
                        max_retries=max_retries,
                        base_backoff=base_backoff,
                        inter_delay=inter_delay,
                        stats=stats,
                    )
                except KeyboardInterrupt:
                    raise

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _setup_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        datefmt=datefmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode="a", encoding="utf-8"),
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download F1 history overnight with backoff/rate limiting",
    )
    parser.add_argument("--start-year", type=int, default=2018)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument(
        "--delay",
        type=float,
        default=30.0,
        metavar="SECS",
        help="Seconds to wait between successful session downloads (default: 30.0). "
        "The Jolpica API allows 500 calls/hr; each session load uses ~4 calls, "
        "so 30s keeps you around 480 calls/hr.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=4,
        help="Max retry attempts per session before giving up (default: 4)",
    )
    parser.add_argument(
        "--base-backoff",
        type=float,
        default=10.0,
        metavar="SECS",
        help="Base backoff seconds for first retry; doubles each attempt (default: 10)",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("logs/download.log"),
        help="Log file path (default: logs/download.log)",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-download sessions that are already on disk",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _setup_logging(args.log_file)

    logger.info("Starting overnight download")
    logger.info("  Years:       %d – %d", args.start_year, args.end_year)
    logger.info("  Delay:       %.1fs between sessions", args.delay)
    logger.info("  Max retries: %d", args.max_retries)
    logger.info(
        "  Base backoff:%.1fs (doubles each retry, jittered)", args.base_backoff
    )
    logger.info("  Log file:    %s", args.log_file)

    stats = DownloadStats()
    try:
        stats = download_overnight(
            start_year=args.start_year,
            end_year=args.end_year,
            data_dir=args.data_dir,
            cache_dir=args.cache_dir,
            skip_existing=not args.no_skip_existing,
            inter_delay=args.delay,
            max_retries=args.max_retries,
            base_backoff=args.base_backoff,
        )
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        stats.report(logger)


if __name__ == "__main__":
    main()
