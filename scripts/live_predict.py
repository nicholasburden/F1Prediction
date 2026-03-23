#!/usr/bin/env python
"""Live F1 race prediction.

Automatically fetches session data and weather forecasts to predict
race finishing positions for the current or specified weekend.

Usage:
    uv run python scripts/live_predict.py                  # all drivers, current weekend
    uv run python scripts/live_predict.py --driver VER     # specific driver
    uv run python scripts/live_predict.py --round 3        # specific round
    uv run python scripts/live_predict.py --no-weather     # skip weather forecast
"""

import argparse
import json
import logging
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import fastf1
import pandas as pd
import polars as pl
import torch

from f1prediction.data.dataset import _convert_durations, find_driver_team, vocab_index
from f1prediction.data.download import SESSION_KEY_MAP, _retry_on_rate_limit, slugify
from f1prediction.data.features import WEEKEND_SESSION_ORDER
from f1prediction.data.history import build_history_table
from f1prediction.data.normalization import NormalizationStats
from f1prediction.data.registry import REGISTRY
from f1prediction.config import ModelConfig
from f1prediction.models import build_model
from f1prediction.training.splits import get_event_order

_TREE_BACKENDS = {"xgboost", "lightgbm"}

# Trigger feature registration
import f1prediction.data.features  # noqa: F401

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Circuit coordinates for weather forecasts (keyed by FastF1 Location field)
CIRCUIT_COORDS: dict[str, tuple[float, float]] = {
    "Sakhir": (26.0325, 50.5106),
    "Jeddah": (21.6319, 39.1044),
    "Melbourne": (-37.8497, 144.9680),
    "Suzuka": (34.8431, 136.5407),
    "Shanghai": (31.3389, 121.2198),
    "Miami": (25.9581, -80.2389),
    "Imola": (44.3439, 11.7167),
    "Monaco": (43.7347, 7.4206),
    "Monte-Carlo": (43.7347, 7.4206),
    "Montréal": (45.5000, -73.5228),
    "Montreal": (45.5000, -73.5228),
    "Barcelona": (41.57, 2.2611),
    "Spielberg": (47.2197, 14.7647),
    "Silverstone": (52.0786, -1.0169),
    "Budapest": (47.5789, 19.2486),
    "Spa-Francorchamps": (50.4372, 5.9714),
    "Stavelot": (50.4372, 5.9714),
    "Zandvoort": (52.3888, 4.5409),
    "Monza": (45.6156, 9.2811),
    "Baku": (40.3725, 49.8533),
    "Marina Bay": (1.2914, 103.8640),
    "Singapore": (1.2914, 103.8640),
    "Austin": (30.1328, -97.6411),
    "Mexico City": (19.4042, -99.0907),
    "São Paulo": (-23.7036, -46.6997),
    "Sao Paulo": (-23.7036, -46.6997),
    "Las Vegas": (36.1162, -115.1745),
    "Lusail": (25.49, 51.4542),
    "Yas Island": (24.4672, 54.6031),
    "Abu Dhabi": (24.4672, 54.6031),
    "Portimão": (37.227, -8.627),
    "Istanbul": (40.9517, 29.4050),
    "Mugello": (43.9975, 11.3719),
    "Nürburg": (50.3356, 6.9475),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live F1 race prediction")
    parser.add_argument(
        "--driver", type=str, default=None,
        help="Driver abbreviation (e.g. VER). Omit for all drivers.",
    )
    parser.add_argument("--year", type=int, default=datetime.now().year)
    parser.add_argument(
        "--round", type=int, default=None,
        help="Round number. Default: current/next race.",
    )
    parser.add_argument("--run-dir", type=Path, default=Path("runs"))
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--no-weather", action="store_true", help="Skip weather forecast")
    parser.add_argument(
        "--target", type=str, default="R", choices=["Q", "Sprint", "R"],
        help="Session to predict results for (default: R)",
    )
    return parser.parse_args()


def get_event(year: int, round_num: int | None = None) -> pd.Series:
    """Get the current/next event, or a specific round."""
    schedule = _retry_on_rate_limit(lambda: fastf1.get_event_schedule(year))
    schedule = schedule[schedule["RoundNumber"] > 0]

    if round_num is not None:
        matches = schedule[schedule["RoundNumber"] == round_num]
        if matches.empty:
            raise ValueError(f"Round {round_num} not found in {year} schedule")
        return matches.iloc[0]

    # Find next upcoming race
    now = datetime.now(timezone.utc)
    for _, event in schedule.iterrows():
        # Check the last session date (Race) to see if the weekend has passed
        for i in range(5, 0, -1):
            date = event.get(f"Session{i}DateUtc")
            if date is not None and not pd.isna(date):
                session_dt = pd.Timestamp(date)
                if session_dt.tzinfo is None:
                    session_dt = session_dt.tz_localize("UTC")
                # Weekend hasn't finished yet (add buffer for race duration)
                if session_dt + pd.Timedelta(hours=3) >= pd.Timestamp(now):
                    return event
                break

    # All events passed — return the last one
    return schedule.iloc[-1]


def get_completed_sessions(event: pd.Series) -> list[tuple[str, str]]:
    """Return (session_name, short_code) for sessions that have finished.

    Excludes Race since that's what we're predicting.
    """
    now = pd.Timestamp(datetime.now(timezone.utc))
    completed = []

    for i in range(1, 6):
        name = event.get(f"Session{i}")
        if not name or name == "None" or name not in SESSION_KEY_MAP:
            continue
        short = SESSION_KEY_MAP[name]
        if short == "R":
            continue

        date = event.get(f"Session{i}DateUtc")
        if date is None or pd.isna(date):
            continue
        session_dt = pd.Timestamp(date)
        if session_dt.tzinfo is None:
            session_dt = session_dt.tz_localize("UTC")
        # 3-hour buffer: session takes ~1-2h, data available ~30min after
        if now > session_dt + pd.Timedelta(hours=3):
            completed.append((name, short))

    return completed


def load_live_session(
    year: int, event_name: str, session_short: str,
) -> dict[str, pl.DataFrame] | None:
    """Load a session via FastF1 and return as dict of polars DataFrames."""
    try:
        session = _retry_on_rate_limit(
            lambda: fastf1.get_session(year, event_name, session_short)
        )
        _retry_on_rate_limit(
            lambda: session.load(telemetry=False, messages=False)
        )
    except Exception as e:
        logger.warning("Could not load %s: %s", session_short, e)
        return None

    data: dict[str, pl.DataFrame] = {}
    if session.laps is not None and not session.laps.empty:
        data["laps"] = _convert_durations(pl.from_pandas(session.laps))
    if session.results is not None and not session.results.empty:
        data["results"] = _convert_durations(pl.from_pandas(session.results))
    if session.weather_data is not None and not session.weather_data.empty:
        data["weather"] = _convert_durations(pl.from_pandas(session.weather_data))

    return data if data else None


def build_event_data(year: int, event: pd.Series) -> dict:
    """Build event_data dict by loading completed sessions via FastF1."""
    event_name = event["EventName"]
    slug = slugify(event_name)
    event_data: dict = {"meta": {"year": year, "event_slug": slug}}

    completed = get_completed_sessions(event)
    if not completed:
        logger.info("No completed sessions yet — predicting from identity/era features only")
        return event_data

    for session_name, short in completed:
        logger.info("Loading %s (%s)...", session_name, short)
        data = load_live_session(year, event_name, short)
        if data:
            event_data[short] = data
            logger.info("  Loaded: %s", ", ".join(data.keys()))

    return event_data


def get_circuit_coords(event: pd.Series) -> tuple[float, float] | None:
    """Look up circuit coordinates from the event's Location field."""
    location = event.get("Location", "")
    if not location:
        return None

    # Direct match
    if location in CIRCUIT_COORDS:
        return CIRCUIT_COORDS[location]

    # Fuzzy match
    loc_lower = location.lower()
    for key, coords in CIRCUIT_COORDS.items():
        if key.lower() in loc_lower or loc_lower in key.lower():
            return coords

    return None


def fetch_weather_forecast(
    lat: float, lon: float, date_str: str,
) -> dict[str, float | bool] | None:
    """Fetch weather forecast from Open-Meteo (free, no API key)."""
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        f"&hourly=temperature_2m,relative_humidity_2m,surface_pressure,"
        f"wind_speed_10m,rain"
        f"&start_date={date_str}&end_date={date_str}"
        f"&timezone=UTC"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "F1Prediction/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())

        hourly = data["hourly"]

        # Average over typical race hours (12:00-17:00 UTC)
        def avg(vals: list, start: int = 12, end: int = 17) -> float:
            window = [v for v in vals[start:end] if v is not None]
            return sum(window) / len(window) if window else 0.0

        air_temp = avg(hourly["temperature_2m"])
        return {
            "AirTemp": air_temp,
            "Humidity": avg(hourly["relative_humidity_2m"]),
            "Pressure": avg(hourly["surface_pressure"]),
            "TrackTemp": air_temp + 15.0,  # rough estimate
            "WindSpeed": avg(hourly["wind_speed_10m"]),
            "Rainfall": any(
                r > 0 for r in hourly["rain"][12:17] if r is not None
            ),
        }
    except Exception as e:
        logger.warning("Weather forecast failed: %s", e)
        return None


def inject_weather(event_data: dict, event: pd.Series) -> None:
    """Fetch weather forecast and inject as event_data['R']['weather']."""
    coords = get_circuit_coords(event)
    if coords is None:
        location = event.get("Location", "unknown")
        logger.info("No coordinates for '%s' — skipping weather", location)
        return

    # Find race date
    race_date_str = None
    for i in range(5, 0, -1):
        name = event.get(f"Session{i}")
        if name == "Race":
            date = event.get(f"Session{i}DateUtc")
            if date is not None and not pd.isna(date):
                race_date_str = pd.Timestamp(date).strftime("%Y-%m-%d")
            break

    if race_date_str is None:
        return

    logger.info("Fetching weather forecast for %s on %s...", event.get("Location"), race_date_str)
    forecast = fetch_weather_forecast(coords[0], coords[1], race_date_str)
    if forecast is None:
        return

    logger.info(
        "  Forecast: %.1f°C air, %.1f°C track, %.0f%% humidity, %s",
        forecast["AirTemp"],
        forecast["TrackTemp"],
        forecast["Humidity"],
        "rain" if forecast["Rainfall"] else "dry",
    )

    weather_df = pl.DataFrame({
        "AirTemp": [forecast["AirTemp"]],
        "Humidity": [forecast["Humidity"]],
        "Pressure": [forecast["Pressure"]],
        "TrackTemp": [forecast["TrackTemp"]],
        "WindSpeed": [forecast["WindSpeed"]],
        "Rainfall": [forecast["Rainfall"]],
    })
    if "R" not in event_data:
        event_data["R"] = {}
    event_data["R"]["weather"] = weather_df


def find_drivers(event_data: dict) -> list[str]:
    """Extract driver abbreviations from loaded session data."""
    drivers: set[str] = set()
    for key, session in event_data.items():
        if key == "meta":
            continue
        results = session.get("results")
        if results is not None and "Abbreviation" in results.columns:
            drivers.update(results["Abbreviation"].drop_nulls().to_list())
    return sorted(drivers)


def get_drivers_from_previous_round(year: int, event: pd.Series) -> list[str]:
    """Load the driver list from the previous round's race results."""
    current_round = event["RoundNumber"]
    if current_round <= 1:
        return []
    try:
        session = _retry_on_rate_limit(
            lambda: fastf1.get_session(year, current_round - 1, "R")
        )
        _retry_on_rate_limit(
            lambda: session.load(telemetry=False, messages=False)
        )
        if session.results is not None and not session.results.empty:
            results = pl.from_pandas(session.results)
            return sorted(results["Abbreviation"].drop_nulls().to_list())
    except Exception as e:
        logger.warning("Could not load previous round: %s", e)
    return []


def _extract_driver_features(
    event_data, driver, max_drivers, driver_vocab, team_vocab,
    target_session, history_table, team_history_table, event_slug, year,
):
    """Extract feature vector and categorical IDs for a single driver."""
    driver_history = history_table.get((year, event_slug, driver))
    team_name = find_driver_team(event_data, driver)
    team_hist = (
        team_history_table.get((year, event_slug, team_name))
        if team_name else None
    )
    feature_vec = REGISTRY.extract(
        event_data, driver,
        max_drivers=max_drivers,
        driver_vocab=driver_vocab,
        team_vocab=team_vocab,
        target_session=target_session,
        driver_history=driver_history,
        team_history=team_hist,
    )
    driver_idx = vocab_index(driver, driver_vocab)
    team_idx = vocab_index(team_name, team_vocab)
    return feature_vec, driver_idx, team_idx


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir

    # Load model artifacts
    vocab_path = run_dir / "vocabs.json"
    if not vocab_path.exists():
        logger.error("vocabs.json not found in %s. Run training first.", run_dir)
        return

    vocabs = json.loads(vocab_path.read_text())
    driver_vocab: list[str] = vocabs["driver_vocab"]
    team_vocab: list[str] = vocabs["team_vocab"]
    max_drivers: int = vocabs["max_drivers"]

    norm_stats = NormalizationStats.load(run_dir / "norm_stats.json")

    model_cfg_path = run_dir / "model_config.json"
    if not model_cfg_path.exists():
        logger.error("model_config.json not found in %s. Run training first.", run_dir)
        return
    mc = json.loads(model_cfg_path.read_text())
    backend = mc.get("backend", "torch")
    is_tree = backend in _TREE_BACKENDS

    # Load model based on backend
    if is_tree:
        from f1prediction.models.gbm import GBMModel
        model = GBMModel.load(run_dir / "best_model.joblib")
    else:
        model_cfg = ModelConfig(
            model_type=mc["model_type"],
            hidden_dims=mc.get("hidden_dims", [128, 64]),
            dropout=mc.get("dropout", 0.1),
            driver_embed_dim=mc.get("driver_embed_dim", 8),
            team_embed_dim=mc.get("team_embed_dim", 4),
            normalize_embeddings=mc.get("normalize_embeddings", False),
        )
        model = build_model(
            mc["continuous_dim"], model_cfg,
            driver_vocab_size=mc["driver_vocab_size"],
            team_vocab_size=mc["team_vocab_size"],
        )
        model.load_state_dict(torch.load(run_dir / "best_model.pt", weights_only=True))
        model.eval()

    # Enable FastF1 cache
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(args.cache_dir))

    # Get event
    logger.info("Looking up %d schedule...", args.year)
    event = get_event(args.year, args.round)
    event_name = event["EventName"]
    location = event.get("Location", "")
    country = event.get("Country", "")
    logger.info(
        "Event: %s (Round %d) — %s, %s",
        event_name, event["RoundNumber"], location, country,
    )

    # Build event data from live sessions
    event_data = build_event_data(args.year, event)

    # Inject weather forecast
    if not args.no_weather:
        inject_weather(event_data, event)

    # Find drivers — fall back to previous round, then training vocabulary
    drivers = find_drivers(event_data)
    if not drivers:
        logger.info("No session data yet — checking previous round...")
        drivers = get_drivers_from_previous_round(args.year, event)
    if not drivers:
        drivers = list(driver_vocab)
        logger.info("Using %d drivers from training vocabulary", len(drivers))

    # Filter to specific driver if requested
    if args.driver:
        abbr = args.driver.upper()
        if abbr not in drivers:
            logger.warning(
                "Driver %s not in session data. Known: %s", abbr, ", ".join(drivers),
            )
        # Predict all drivers so we can report rank, but highlight the requested one
        target_driver = abbr
        if abbr not in drivers:
            drivers.append(abbr)
    else:
        target_driver = None

    logger.info("Predicting for %d driver(s)...", len(drivers))

    # Build history table from on-disk data
    data_dir = Path("data")
    history_table: dict = {}
    team_history_table: dict = {}
    if data_dir.exists():
        all_years = sorted(
            int(d.name) for d in data_dir.iterdir()
            if d.is_dir() and d.name.isdigit()
        )
        if all_years:
            all_events_chrono = get_event_order(data_dir, all_years)
            history_table, team_history_table = build_history_table(data_dir, all_events_chrono)
            logger.info("History table: %d driver, %d team entries from years %s", len(history_table), len(team_history_table), all_years)

    # Get event slug for history lookup
    event_slug = event_data["meta"]["event_slug"]

    # Predict
    available = [s for s in WEEKEND_SESSION_ORDER if s in event_data]
    predictions: list[tuple[str, float]] = []

    extract_kwargs = dict(
        event_data=event_data,
        max_drivers=max_drivers,
        driver_vocab=driver_vocab,
        team_vocab=team_vocab,
        target_session=args.target,
        history_table=history_table,
        team_history_table=team_history_table,
        event_slug=event_slug,
        year=args.year,
    )

    if is_tree:
        # Batch predict for tree models
        all_features = []
        all_cat_ids = []
        for driver in drivers:
            feature_vec, driver_idx, team_idx = _extract_driver_features(
                driver=driver, **extract_kwargs,
            )
            all_features.append(feature_vec)
            all_cat_ids.append([driver_idx, team_idx])

        features_np = np.array(all_features, dtype=np.float32)
        cat_ids_np = np.array(all_cat_ids, dtype=np.int64)
        preds = model.predict(features_np, cat_ids_np)
        for i, driver in enumerate(drivers):
            predictions.append((driver, float(preds[i] * max_drivers)))
    else:
        # Per-driver predict for torch models
        with torch.no_grad():
            for driver in drivers:
                feature_vec, driver_idx, team_idx = _extract_driver_features(
                    driver=driver, **extract_kwargs,
                )
                features = torch.tensor(feature_vec, dtype=torch.float32)
                features = norm_stats.normalize(features)
                cat_ids = torch.tensor([[driver_idx, team_idx]], dtype=torch.long)
                pred = model(features.unsqueeze(0), cat_ids).item()
                predictions.append((driver, pred * max_drivers))

    predictions.sort(key=lambda x: x[1])

    # Display
    has_weather = not args.no_weather and "R" in event_data and "weather" in event_data.get("R", {})
    print()
    print(f"  {event_name} ({args.year} Round {event['RoundNumber']})")
    print(f"  Location: {location}, {country}")
    print(f"  Sessions: {', '.join(available) if available else 'none yet'}")
    if has_weather:
        w = event_data["R"]["weather"]
        print(f"  Weather:  {w['AirTemp'][0]:.0f}°C, {w['Humidity'][0]:.0f}% humidity, "
              f"{'rain' if w['Rainfall'][0] else 'dry'}")
    print()

    if target_driver:
        # Single driver mode
        for rank, (driver, pred_pos) in enumerate(predictions, 1):
            if driver == target_driver:
                print(f"  {target_driver}: predicted P{rank} (score {pred_pos:.2f})")
                if target_driver not in driver_vocab:
                    print(f"  Note: {target_driver} was not in training data — prediction is less reliable")
                break
    else:
        # Full grid
        print(f"  {'Pos':>3}  {'Driver':<6}  {'Predicted':>9}")
        print(f"  {'---':>3}  {'------':<6}  {'---------':>9}")
        for rank, (driver, pred_pos) in enumerate(predictions, 1):
            print(f"  {rank:>3}  {driver:<6}  {pred_pos:>9.2f}")

    print()


if __name__ == "__main__":
    main()
