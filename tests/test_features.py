"""One test per feature — synthetic data in, expected value out."""

from __future__ import annotations

import polars as pl
import pytest

from f1prediction.data.constants import DataTable
from f1prediction.data.features import ALL_FEATURES, grid_position_norm, gap_to_fastest
from f1prediction.data.registry import SESSION_KEYS


def _agg_feature(name: str, table: DataTable, df: pl.DataFrame) -> pl.DataFrame:
    spec = next(s for s in ALL_FEATURES._specs if s.name == name)
    keys = list(SESSION_KEYS[table])
    return df.group_by(keys).agg(spec.expr.alias(name))


# --- Fixtures ---


@pytest.fixture
def results_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Driver": ["VER", "VER", "HAM", "HAM"],
            "Year": [2024, 2024, 2024, 2024],
            "EventId": [1, 1, 1, 1],
            "SessionId": ["FP1", "Q", "FP1", "Q"],
            "TeamName": ["Red Bull Racing", "Red Bull Racing", "Mercedes", "Mercedes"],
            "GridPosition": [None, 1, None, 2],
            "Position": [1, 1, 2, 2],
        }
    )


@pytest.fixture
def laps_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Driver": ["VER", "VER", "VER", "HAM", "HAM"],
            "Year": [2024, 2024, 2024, 2024, 2024],
            "EventId": [1, 1, 1, 1, 1],
            "SessionId": ["FP1", "FP1", "FP1", "FP1", "FP1"],
            "LapTime": [90.0, 89.5, 91.0, 90.2, 90.8],
            "SpeedI1": [300.0, 305.0, 298.0, 302.0, 301.0],
            "SpeedI2": [310.0, 312.0, 308.0, 311.0, 309.0],
            "SpeedFL": [320.0, 322.0, 318.0, 321.0, 319.0],
            "SpeedST": [330.0, 335.0, 328.0, 332.0, 331.0],
        }
    )


@pytest.fixture
def weather_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "Year": [2024, 2024, 2024],
            "EventId": [1, 1, 1],
            "SessionId": ["FP1", "FP1", "FP1"],
            "Rainfall": [False, True, False],
            "AirTemp": [25.0, 26.0, 24.0],
            "TrackTemp": [40.0, 42.0, 38.0],
            "Humidity": [50.0, 55.0, 60.0],
            "WindSpeed": [10.0, 12.0, 8.0],
        }
    )


# --- Era features (event-wide, results table) ---


def test_era_min_weight(results_df: pl.DataFrame) -> None:
    result = _agg_feature("era_min_weight", "results", results_df)
    assert result["era_min_weight"][0] == 798.0


def test_era_fuel_limit(results_df: pl.DataFrame) -> None:
    result = _agg_feature("era_fuel_limit", "results", results_df)
    assert result["era_fuel_limit"][0] == 110.0


def test_era_budget_cap(results_df: pl.DataFrame) -> None:
    result = _agg_feature("era_budget_cap", "results", results_df)
    assert result["era_budget_cap"][0] == 135.0


def test_era_ground_effect(results_df: pl.DataFrame) -> None:
    result = _agg_feature("era_ground_effect", "results", results_df)
    assert result["era_ground_effect"][0] == 1


def test_era_has_drs(results_df: pl.DataFrame) -> None:
    result = _agg_feature("era_has_drs", "results", results_df)
    assert result["era_has_drs"][0] == 1


# --- Embedding features ---


def test_driver_id(results_df: pl.DataFrame) -> None:
    result = _agg_feature("driver_id", "results", results_df)
    values = set(result["driver_id"].to_list())
    assert "VER" in values
    assert "HAM" in values


def test_team_id(results_df: pl.DataFrame) -> None:
    result = _agg_feature("team_id", "results", results_df)
    values = set(result["team_id"].to_list())
    assert "Red Bull Racing" in values
    assert "Mercedes" in values


# --- Session-level results ---


def test_grid_position(results_df: pl.DataFrame) -> None:
    result = _agg_feature("grid_position", "results", results_df)
    q_ver = result.filter((pl.col("Driver") == "VER") & (pl.col("SessionId") == "Q"))
    assert q_ver["grid_position"][0] == 1

    fp1_ver = result.filter(
        (pl.col("Driver") == "VER") & (pl.col("SessionId") == "FP1")
    )
    assert fp1_ver["grid_position"][0] is None  # null, filled later by pipeline


# --- Session-level laps ---


def test_min_lap_time(laps_df: pl.DataFrame) -> None:
    result = _agg_feature("min_lap_time", "laps", laps_df)
    ver = result.filter(pl.col("Driver") == "VER")
    assert ver["min_lap_time"][0] == 89.5


def test_max_speed_i1(laps_df: pl.DataFrame) -> None:
    result = _agg_feature("max_speed_i1", "laps", laps_df)
    ver = result.filter(pl.col("Driver") == "VER")
    assert ver["max_speed_i1"][0] == 305.0


def test_max_speed_i2(laps_df: pl.DataFrame) -> None:
    result = _agg_feature("max_speed_i2", "laps", laps_df)
    ver = result.filter(pl.col("Driver") == "VER")
    assert ver["max_speed_i2"][0] == 312.0


def test_max_speed_fl(laps_df: pl.DataFrame) -> None:
    result = _agg_feature("max_speed_fl", "laps", laps_df)
    ver = result.filter(pl.col("Driver") == "VER")
    assert ver["max_speed_fl"][0] == 322.0


def test_max_speed_st(laps_df: pl.DataFrame) -> None:
    result = _agg_feature("max_speed_st", "laps", laps_df)
    ver = result.filter(pl.col("Driver") == "VER")
    assert ver["max_speed_st"][0] == 335.0


# --- Session-level weather ---


def test_any_rain(weather_df: pl.DataFrame) -> None:
    result = _agg_feature("any_rain", "weather", weather_df)
    assert result["any_rain"][0] == 1.0


def test_mean_rain(weather_df: pl.DataFrame) -> None:
    result = _agg_feature("mean_rain", "weather", weather_df)
    assert abs(result["mean_rain"][0] - 1 / 3) < 1e-5


def test_mean_air_temp(weather_df: pl.DataFrame) -> None:
    result = _agg_feature("mean_air_temp", "weather", weather_df)
    assert result["mean_air_temp"][0] == 25.0


def test_mean_track_temp(weather_df: pl.DataFrame) -> None:
    result = _agg_feature("mean_track_temp", "weather", weather_df)
    assert result["mean_track_temp"][0] == 40.0


def test_mean_humidity(weather_df: pl.DataFrame) -> None:
    result = _agg_feature("mean_humidity", "weather", weather_df)
    assert result["mean_humidity"][0] == 55.0


def test_mean_wind_speed(weather_df: pl.DataFrame) -> None:
    result = _agg_feature("mean_wind_speed", "weather", weather_df)
    assert result["mean_wind_speed"][0] == 10.0


# --- Global features ---


def test_grid_position_norm() -> None:
    df = pl.LazyFrame(
        {
            "Driver": ["VER", "HAM"],
            "Year": [2024, 2024],
            "EventId": [1, 1],
            "SessionId": ["Q", "Q"],
            "grid_position": [1, 2],
        }
    )
    result = grid_position_norm(df).collect()
    assert (
        abs(result.filter(pl.col("Driver") == "VER")["grid_position_norm"][0] - 0.5)
        < 1e-5
    )
    assert (
        abs(result.filter(pl.col("Driver") == "HAM")["grid_position_norm"][0] - 1.0)
        < 1e-5
    )


def test_gap_to_fastest() -> None:
    df = pl.LazyFrame(
        {
            "Driver": ["VER", "HAM"],
            "Year": [2024, 2024],
            "EventId": [1, 1],
            "SessionId": ["FP1", "FP1"],
            "min_lap_time": [89.5, 90.0],
        }
    )
    result = gap_to_fastest(df).collect()
    ver = result.filter(pl.col("Driver") == "VER")
    ham = result.filter(pl.col("Driver") == "HAM")
    assert abs(ver["gap_to_fastest"][0]) < 1e-5  # fastest, gap = 0
    assert abs(ham["gap_to_fastest"][0] - (90.0 - 89.5) / 89.5) < 1e-5
