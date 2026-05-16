"""Weather forecast lookup via Open-Meteo (free, keyless).

Used at inference time to populate weather.parquet for sessions that have not
yet happened, so the model has non-zero weather inputs for pre-weekend
predictions. Open-Meteo's free tier covers up to ~16 days ahead — beyond that
``fetch_forecast`` returns ``None`` and the caller falls back to null-fill.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TypedDict
from urllib.parse import urlencode
from urllib.request import urlopen
from urllib.error import URLError
import json

import polars as pl

log = logging.getLogger(__name__)

_API = "https://api.open-meteo.com/v1/forecast"
_HOURLY_VARS = (
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "soil_temperature_0cm",
)


class WeatherRow(TypedDict):
    AirTemp: float
    TrackTemp: float
    Humidity: float
    WindSpeed: float
    Rainfall: bool


def fetch_forecast(lat: float, lon: float, dt: datetime) -> WeatherRow | None:
    """Return a single-row weather snapshot for the hour matching ``dt``.

    ``dt`` is interpreted as UTC (naive datetimes are assumed UTC). Returns
    ``None`` if Open-Meteo declines (e.g. dt is more than ~16 days away) or
    the request fails.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt_utc = dt.astimezone(timezone.utc)
    date_str = dt_utc.strftime("%Y-%m-%d")
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(_HOURLY_VARS),
        "start_date": date_str,
        "end_date": date_str,
        "timezone": "UTC",
        "wind_speed_unit": "ms",
    }
    url = f"{_API}?{urlencode(params)}"
    try:
        with urlopen(url, timeout=10) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (URLError, TimeoutError, json.JSONDecodeError) as e:
        log.warning("forecast fetch failed for (%s, %s) %s: %s", lat, lon, date_str, e)
        return None

    hourly = payload.get("hourly")
    if not hourly or not hourly.get("time"):
        log.warning("forecast empty for (%s, %s) %s", lat, lon, date_str)
        return None

    target_hour = dt_utc.replace(minute=0, second=0, microsecond=0)
    times = [datetime.fromisoformat(t).replace(tzinfo=timezone.utc) for t in hourly["time"]]
    diffs = [abs((t - target_hour).total_seconds()) for t in times]
    idx = min(range(len(diffs)), key=diffs.__getitem__)

    air = hourly["temperature_2m"][idx]
    soil = hourly.get("soil_temperature_0cm", [None] * len(times))[idx]
    rh = hourly["relative_humidity_2m"][idx]
    wind = hourly["wind_speed_10m"][idx]
    precip = hourly["precipitation"][idx]
    if air is None or rh is None or wind is None:
        return None
    return WeatherRow(
        AirTemp=float(air),
        TrackTemp=float(soil) if soil is not None else float(air),
        Humidity=float(rh),
        WindSpeed=float(wind),
        Rainfall=bool((precip or 0.0) > 0.0),
    )


def write_weather_parquet(row: WeatherRow, path) -> None:
    """Write a single-row weather.parquet matching FastF1's schema."""
    pl.DataFrame(
        [row],
        schema={
            "AirTemp": pl.Float64,
            "TrackTemp": pl.Float64,
            "Humidity": pl.Float64,
            "WindSpeed": pl.Float64,
            "Rainfall": pl.Boolean,
        },
    ).write_parquet(path)
