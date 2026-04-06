from __future__ import annotations

from collections.abc import Mapping

import polars as pl

from f1prediction.data.registry import FeatureRegistry, FeatureSpec, GlobalFeatureSpec

WEEKEND_SESSION_ORDER: tuple[str, ...] = (
    "FP1",
    "FP2",
    "FP3",
    "SQ",
    "SS",
    "Sprint",
    "Q",
)


# Era regulation lookup tables keyed by year breakpoints.
# Values: {year_start: value} — matched by finding the latest year <= sample year.
_ERA_MIN_WEIGHT_KG = {2014: 690, 2022: 798, 2025: 800, 2026: 768}
_ERA_FUEL_LIMIT_KG = {2014: 100, 2022: 110, 2026: 70}
_ERA_BUDGET_CAP_M = {2014: 0, 2021: 145, 2022: 140, 2023: 135, 2026: 215}
_ERA_GROUND_EFFECT = {2014: 0, 2022: 1, 2026: 0}
_ERA_HAS_DRS = {2014: 1, 2026: 0}


def _year_lookup(table: Mapping[int, float]) -> pl.Expr:
    """Map Year to the value from the most recent matching era breakpoint."""
    breakpoints = sorted(table.keys())
    expr = pl.lit(table[breakpoints[0]]).cast(pl.Float32)
    for year in breakpoints:
        expr = (
            pl.when(pl.col("Year").first() >= year)
            .then(pl.lit(table[year]))
            .otherwise(expr)
        )
    return expr


def grid_position_norm(df: pl.LazyFrame) -> pl.LazyFrame:
    """Normalise grid position by the number of drivers in that event."""
    n_drivers = df.group_by(["Year", "EventId"]).agg(
        pl.col("Driver").n_unique().cast(pl.Float32).alias("_n_drivers")
    )
    return (
        df.join(n_drivers, on=["Year", "EventId"])
        .with_columns(
            (pl.col("grid_position") / pl.col("_n_drivers"))
            .fill_null(1.0)
            .alias("grid_position_norm")
        )
        .drop("_n_drivers")
    )


def gap_to_fastest(df: pl.LazyFrame) -> pl.LazyFrame:
    fastest = df.group_by(["Year", "EventId", "SessionId"]).agg(
        pl.col("min_lap_time").min().alias("session_fastest")
    )
    return (
        df.join(fastest, on=["Year", "EventId", "SessionId"])
        .with_columns(
            pl.when(pl.col("min_lap_time").is_not_null())
            .then(
                (pl.col("min_lap_time") - pl.col("session_fastest"))
                / pl.col("session_fastest")
            )
            .otherwise(1.0)
            .alias("gap_to_fastest")
        )
        .drop("session_fastest")
    )


ALL_FEATURES = FeatureRegistry(
    specs=[
        # --- Event-level (results table, event grain) ---
        FeatureSpec(
            "era_min_weight",
            "core",
            _year_lookup(_ERA_MIN_WEIGHT_KG),
            "results",
            event_wide=True,
        ),
        FeatureSpec(
            "era_fuel_limit",
            "core",
            _year_lookup(_ERA_FUEL_LIMIT_KG),
            "results",
            event_wide=True,
        ),
        FeatureSpec(
            "era_budget_cap",
            "core",
            _year_lookup(_ERA_BUDGET_CAP_M),
            "results",
            event_wide=True,
        ),
        FeatureSpec(
            "era_ground_effect",
            "core",
            _year_lookup(_ERA_GROUND_EFFECT),
            "results",
            event_wide=True,
        ),
        FeatureSpec(
            "era_has_drs",
            "core",
            _year_lookup(_ERA_HAS_DRS),
            "results",
            event_wide=True,
        ),
        FeatureSpec(
            "driver_id",
            "core",
            pl.col("Driver").first(),
            "results",
            encoding="embedding",
            event_wide=True,
        ),
        FeatureSpec(
            "team_id",
            "core",
            pl.col("TeamName").first(),
            "results",
            encoding="embedding",
            event_wide=True,
        ),
        # --- Session-level results ---
        FeatureSpec(
            "grid_position",
            "core",
            pl.col("GridPosition").first(ignore_nulls=True),
            "results",
        ),
        # --- Session-level laps ---
        FeatureSpec("min_lap_time", "core", pl.col("LapTime").min(), "laps"),
        FeatureSpec("max_speed_i1", "core", pl.col("SpeedI1").max(), "laps"),
        FeatureSpec("max_speed_i2", "core", pl.col("SpeedI2").max(), "laps"),
        FeatureSpec("max_speed_fl", "core", pl.col("SpeedFL").max(), "laps"),
        FeatureSpec("max_speed_st", "core", pl.col("SpeedST").max(), "laps"),
        # --- Session-level weather ---
        FeatureSpec(
            "any_rain", "core", pl.col("Rainfall").max().cast(pl.Float32), "weather"
        ),
        FeatureSpec(
            "mean_rain", "core", pl.col("Rainfall").mean().cast(pl.Float32), "weather"
        ),
        FeatureSpec("mean_air_temp", "core", pl.col("AirTemp").mean(), "weather"),
        FeatureSpec("mean_track_temp", "core", pl.col("TrackTemp").mean(), "weather"),
        FeatureSpec("mean_humidity", "core", pl.col("Humidity").mean(), "weather"),
        FeatureSpec("mean_wind_speed", "core", pl.col("WindSpeed").mean(), "weather"),
    ],
    global_specs=[
        GlobalFeatureSpec("grid_position_norm", "core", grid_position_norm),
        GlobalFeatureSpec("gap_to_fastest", "core", gap_to_fastest),
    ],
)
