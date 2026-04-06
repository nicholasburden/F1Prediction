"""Download historical F1 session data using FastF1.

Saves per-session parquet files under::

    data/<year>/<round_num:02d>_<event_slug>/<session_abbrev>/
        results.parquet
        laps.parquet
        weather.parquet

LapTime and qualifying time columns (Q1, Q2, Q3) are converted from
timedelta to float seconds before saving.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Session abbreviation mapping
# ---------------------------------------------------------------------------

SESSION_ABBREVS: dict[str, str] = {
    "Practice 1": "FP1",
    "Practice 2": "FP2",
    "Practice 3": "FP3",
    "Qualifying": "Q",
    "Sprint Qualifying": "SQ",
    "Sprint Shootout": "SS",
    "Sprint": "Sprint",
    "Race": "R",
}

# Columns to save from session results (those that exist will be saved).
_RESULT_COLUMNS = [
    "Abbreviation",
    "TeamName",
    "Position",
    "GridPosition",
    "Q1",
    "Q2",
    "Q3",
    "Points",
    "Status",
]

# Lap columns to save.
_LAP_COLUMNS = [
    "Driver",
    "LapTime",
    "Stint",
    "Compound",
    "LapNumber",
    "SpeedI1",
    "SpeedI2",
    "SpeedFL",
    "SpeedST",
]

# Weather columns to save.
_WEATHER_COLUMNS = [
    "AirTemp",
    "TrackTemp",
    "Humidity",
    "WindSpeed",
    "Rainfall",
]

# Timedelta columns that should be converted to float seconds.
_TIMEDELTA_RESULT_COLS = ["Q1", "Q2", "Q3"]
_TIMEDELTA_LAP_COLS = ["LapTime"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _slugify(name: str) -> str:
    """Convert an event name to a filesystem-safe slug."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _event_slug(round_num: int, event_name: str) -> str:
    """Build the directory name for an event."""
    return f"{round_num:02d}_{_slugify(event_name)}"


def _timedelta_to_seconds(series: pl.Series) -> pl.Series:
    """Convert a pl.Duration series to float64 seconds.

    If the series is already numeric, return it unchanged.
    """
    if series.dtype == pl.Duration:
        return series.dt.total_microseconds().cast(pl.Float64) / 1_000_000.0
    # Try to cast to float in case it's stored as a string or integer.
    try:
        return series.cast(pl.Float64)
    except Exception:
        return series


def _convert_timedelta_cols(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    """Convert listed timedelta columns in a DataFrame to float seconds in-place."""
    for col in cols:
        if col not in df.columns:
            continue
        converted = _timedelta_to_seconds(df[col])
        df = df.with_columns(converted.alias(col))
    return df


def _select_available_cols(df: pl.DataFrame, wanted: list[str]) -> pl.DataFrame:
    """Select only those columns from ``wanted`` that actually exist in ``df``."""
    available = [c for c in wanted if c in df.columns]
    return df.select(available)


# ---------------------------------------------------------------------------
# Core download logic
# ---------------------------------------------------------------------------


def _save_results(session, out_dir: Path) -> None:
    """Save session results to ``results.parquet``."""
    try:
        results = session.results
        if results is None or results.empty:
            return
        # Convert pandas → polars.
        df = pl.from_pandas(results.reset_index(drop=True))
        df = _select_available_cols(df, _RESULT_COLUMNS)
        df = _convert_timedelta_cols(df, _TIMEDELTA_RESULT_COLS)
        out_dir.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out_dir / "results.parquet")
        logger.debug("    Saved results.parquet (%d rows)", len(df))
    except Exception as exc:
        logger.warning("    Could not save results: %s", exc)


def _save_laps(session, out_dir: Path) -> None:
    """Save session lap data to ``laps.parquet``."""
    try:
        laps = session.laps
        if laps is None or laps.empty:
            return
        df = pl.from_pandas(laps.reset_index(drop=True))
        df = _select_available_cols(df, _LAP_COLUMNS)
        df = _convert_timedelta_cols(df, _TIMEDELTA_LAP_COLS)
        out_dir.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out_dir / "laps.parquet")
        logger.debug("    Saved laps.parquet (%d rows)", len(df))
    except Exception as exc:
        logger.warning("    Could not save laps: %s", exc)


def _save_weather(session, out_dir: Path) -> None:
    """Save session weather data to ``weather.parquet``."""
    try:
        weather = session.weather_data
        if weather is None or weather.empty:
            return
        df = pl.from_pandas(weather.reset_index(drop=True))
        df = _select_available_cols(df, _WEATHER_COLUMNS)
        out_dir.mkdir(parents=True, exist_ok=True)
        df.write_parquet(out_dir / "weather.parquet")
        logger.debug("    Saved weather.parquet (%d rows)", len(df))
    except Exception as exc:
        logger.warning("    Could not save weather: %s", exc)


def _session_dir_exists(out_dir: Path) -> bool:
    """Return True if the session directory already contains any parquet file."""
    if not out_dir.exists():
        return False
    return any(out_dir.glob("*.parquet"))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def download_history(
    start_year: int,
    end_year: int,
    data_dir: Path,
    cache_dir: Path,
    skip_existing: bool = True,
) -> None:
    """Download F1 historical data for a range of seasons using FastF1.

    For each event session, saves three parquet files:
    - ``results.parquet`` — session results (selected columns, timedeltas→seconds).
    - ``laps.parquet`` — lap data with LapTime as float seconds.
    - ``weather.parquet`` — weather data.

    Args:
        start_year: First season to download (inclusive).
        end_year: Last season to download (inclusive).
        data_dir: Root directory where data is stored.
        cache_dir: FastF1 cache directory (avoids re-downloading raw data).
        skip_existing: If True, skip session directories that already contain
            parquet files.
    """
    import fastf1  # local import so module is usable without fastf1 installed

    cache_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))

    for year in range(start_year, end_year + 1):
        logger.info("=== Year %d ===", year)
        try:
            schedule = fastf1.get_event_schedule(year, include_testing=False)
        except Exception as exc:
            logger.error("Could not fetch schedule for %d: %s", year, exc)
            continue

        for _, event_row in schedule.iterrows():
            round_num: int = int(event_row.get("RoundNumber") or 0)
            event_name: str = str(event_row.get("EventName", f"round{round_num}"))
            slug = _event_slug(round_num, event_name)
            event_dir = data_dir / str(year) / slug
            logger.info("  Event %02d: %s → %s", round_num, event_name, slug)

            # Determine which sessions this event has.
            # FastF1 event rows contain columns like Session1, Session2, … Session5.
            session_names: list[str] = []
            for i in range(1, 6):
                sess_col = f"Session{i}"
                sess_name_val = event_row.get(sess_col)
                if (
                    sess_name_val
                    and isinstance(sess_name_val, str)
                    and sess_name_val.strip()
                ):
                    session_names.append(sess_name_val.strip())

            if not session_names:
                # Fallback: use the canonical set for a standard weekend.
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
                    logger.debug("    Unknown session name '%s', skipping", sess_name)
                    continue

                out_dir = event_dir / abbrev

                if skip_existing and _session_dir_exists(out_dir):
                    logger.debug("    Skipping %s (already exists)", abbrev)
                    continue

                logger.info("    Downloading %s (%s)...", abbrev, sess_name)
                try:
                    session = fastf1.get_session(year, round_num, sess_name)
                    session.load(
                        laps=True, telemetry=False, weather=True, messages=False
                    )
                except Exception as exc:
                    logger.warning(
                        "    Could not load session %s/%s/%s: %s",
                        year,
                        slug,
                        abbrev,
                        exc,
                    )
                    continue

                _save_results(session, out_dir)
                _save_laps(session, out_dir)
                _save_weather(session, out_dir)
                logger.info("    Done: %s", abbrev)

    logger.info("Download complete.")
