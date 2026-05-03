from __future__ import annotations

from collections.abc import Mapping
from typing import Callable

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

# Chronological session orderings. The correct order depends on both whether
# the weekend has a Sprint and the year (F1 shuffled sprint formats across
# 2021, 2023, and 2024). Each mapping covers the sessions present in that
# format; unknown sessions sort first via replace_strict(default=-1).
#
# Non-sprint weekend (all years): Fri FP1/FP2 — Sat FP3/Q — Sun R
_ORD_STANDARD: dict[str, int] = {
    "FP1": 0, "FP2": 1, "FP3": 2, "Q": 3, "R": 4,
}
# Sprint 2021–2022: Fri FP1 + Q — Sat FP2 + Sprint — Sun R
_ORD_SPRINT_2021_2022: dict[str, int] = {
    "FP1": 0, "Q": 1, "FP2": 2, "Sprint": 3, "R": 4,
}
# Sprint 2023: Fri FP1 + Q — Sat SS + Sprint — Sun R
_ORD_SPRINT_2023: dict[str, int] = {
    "FP1": 0, "Q": 1, "SS": 2, "Sprint": 3, "R": 4,
}
# Sprint 2024+: Fri FP1 + SQ — Sat Sprint + Q — Sun R
_ORD_SPRINT_2024_PLUS: dict[str, int] = {
    "FP1": 0, "SQ": 1, "Sprint": 2, "Q": 3, "R": 4,
}


def _chronological_ord_expr() -> pl.Expr:
    """Per-format chronological session ordinal. Requires ``_has_sprint`` and
    ``Year`` and ``SessionId`` columns to be present."""
    return (
        pl.when(~pl.col("_has_sprint"))
        .then(pl.col("SessionId").replace_strict(_ORD_STANDARD, default=-1))
        .when(pl.col("Year") <= 2022)
        .then(pl.col("SessionId").replace_strict(_ORD_SPRINT_2021_2022, default=-1))
        .when(pl.col("Year") == 2023)
        .then(pl.col("SessionId").replace_strict(_ORD_SPRINT_2023, default=-1))
        .otherwise(
            pl.col("SessionId").replace_strict(_ORD_SPRINT_2024_PLUS, default=-1)
        )
    )


def attach_session_ord(df: pl.LazyFrame) -> pl.LazyFrame:
    """Attach ``_has_sprint`` (per (Year, EventId)) and ``_session_ord`` (the
    chronological ordinal of the session within the weekend) to ``df``. The
    ordinal differs by year because F1 has shuffled sprint formats — see the
    ``_ORD_*`` mappings."""
    has_sprint = df.group_by(["Year", "EventId"]).agg(
        (pl.col("SessionId") == "Sprint").any().alias("_has_sprint")
    )
    return df.join(has_sprint, on=["Year", "EventId"]).with_columns(
        _chronological_ord_expr().alias("_session_ord")
    )


def _sort_chronological(df: pl.LazyFrame) -> pl.LazyFrame:
    return (
        attach_session_ord(df)
        .sort(["Year", "EventId", "_session_ord"])
        .drop(["_session_ord", "_has_sprint"])
    )


# Era regulation lookup tables keyed by year breakpoints.
# Values: {year_start: value} — matched by finding the latest year <= sample year.
_ERA_MIN_WEIGHT_KG = {2014: 690, 2022: 798, 2025: 800, 2026: 768}
_ERA_FUEL_LIMIT_KG = {2014: 100, 2022: 110, 2026: 70}
_ERA_BUDGET_CAP_M = {2014: 0, 2021: 145, 2022: 140, 2023: 135, 2026: 215}
_ERA_GROUND_EFFECT = {2014: 0, 2022: 1, 2026: 0}
_ERA_HAS_DRS = {2014: 1, 2026: 0}

_RACE_POINTS: dict[int, int] = {
    1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1,
}
_SPRINT_POINTS: dict[int, int] = {
    1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1,
}


def _year_lookup(table: Mapping[int, float]) -> pl.Expr:
    """Map Year to the value from the most recent matching era breakpoint."""
    breakpoints = sorted(table.keys())
    expr = pl.lit(table[breakpoints[0]]).cast(pl.Float32)
    for year in breakpoints:
        expr = (
            pl.when(pl.col("Year").first() >= year)
            .then(pl.lit(table[year]).cast(pl.Float32))
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
            (pl.col("grid_position") / pl.col("_n_drivers")).alias("grid_position_norm")
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

def _decay_weights(n: int, alpha: float) -> list[float]:
    raw = [(1 - alpha) ** i for i in reversed(range(n))]
    total = sum(raw)
    return [w * n / total for w in raw]


def _race_rolling_feature(
    df: pl.LazyFrame,
    source_expr: pl.Expr,
    feature_name: str,
    weights: list[float],
    n: int,
    session_filter: tuple[str, ...] = ("Sprint", "R"),
    partition: tuple[str, ...] = ("Driver",),
) -> pl.LazyFrame:
    """Roll ``source_expr`` over race-only rows within ``partition``, then
    forward-fill the result across every session of each weekend so non-race
    rows inherit the most recent value. Polars' weighted rolling_mean panics
    on nulls, so callers must supply a null-free ``source_expr``. ``_row_idx``
    is preserved so we can restore chronological order after polars' join
    reshuffles rows."""
    sorted_df = _sort_chronological(df).with_row_index("_row_idx")
    rolling_expr = source_expr.rolling_mean(
        window_size=n, weights=weights, min_samples=1
    )
    race_rolling = (
        sorted_df.filter(pl.col("SessionId").is_in(list(session_filter)))
        .with_columns(rolling_expr.over(list(partition)).alias(feature_name))
        .select(["Driver", "Year", "EventId", "SessionId", feature_name])
    )
    return (
        sorted_df.join(
            race_rolling,
            on=["Driver", "Year", "EventId", "SessionId"],
            how="left",
        )
        .sort("_row_idx")
        .with_columns(pl.col(feature_name).forward_fill().over(list(partition)))
        .drop("_row_idx")
    )


def avg_grid_pos(n: int, alpha: float) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    weights = _decay_weights(n, alpha)

    def fn(df: pl.LazyFrame) -> pl.LazyFrame:
        return _race_rolling_feature(
            df,
            pl.col("grid_position").fill_null(pl.col("NumDrivers")),
            "avg_grid_pos",
            weights,
            n,
        )

    return fn


def avg_finish_pos(n: int, alpha: float) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    weights = _decay_weights(n, alpha)

    def fn(df: pl.LazyFrame) -> pl.LazyFrame:
        return _race_rolling_feature(
            df,
            pl.col("Position").fill_null(pl.col("NumDrivers")),
            "avg_finish_pos",
            weights,
            n,
        )

    return fn


def dnf_rate(n: int, alpha: float) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Rolling fraction of main races where the driver finished last (a proxy
    for DNF, since pipeline fills null Position with NumDrivers). Sprints are
    excluded — main-race reliability is a distinct signal."""
    weights = _decay_weights(n, alpha)

    def fn(df: pl.LazyFrame) -> pl.LazyFrame:
        return _race_rolling_feature(
            df,
            (pl.col("Position") == pl.col("NumDrivers")).cast(pl.Float32),
            "dnf_rate",
            weights,
            n,
            session_filter=("R",),
        )

    return fn


def track_avg_finish(n: int, alpha: float) -> Callable[[pl.LazyFrame], pl.LazyFrame]:
    """Rolling mean finish position at the current track, over the driver's
    last N visits. Captures persistent track affinities (e.g. Verstappen at
    Zandvoort)."""
    weights = _decay_weights(n, alpha)

    def fn(df: pl.LazyFrame) -> pl.LazyFrame:
        return _race_rolling_feature(
            df,
            pl.col("Position").fill_null(pl.col("NumDrivers")),
            "track_avg_finish",
            weights,
            n,
            session_filter=("R",),
            partition=("Driver", "track_id"),
        )

    return fn

def cum_champ_points(df: pl.LazyFrame) -> pl.LazyFrame:
    position = pl.col("Position").round().cast(pl.Int32)
    return _sort_chronological(df).with_columns(
            pl.when(pl.col("SessionId") == "Sprint")
            .then(position.replace_strict(_SPRINT_POINTS, default=0))
            .when(pl.col("SessionId") == "R")
            .then(position.replace_strict(_RACE_POINTS, default=0))
            .otherwise(0)
            .cum_sum()
            .over(["Driver", "Year"])
            .alias("cum_champ_points")
        )



CORE_FEATURES = FeatureRegistry(
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
        FeatureSpec(
            "event_id",
            "core",
            pl.col("EventId").first(),
            "results",
            event_wide=True,
        ),
        FeatureSpec(
            "track_id",
            "core",
            pl.col("TrackName").first(),
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
        FeatureSpec(
            "min_lap_time",
            "core",
            pl.col("LapTime").min(),
            "laps",
            fill_null=999.0,
        ),
        FeatureSpec(
            "min_lap_time_soft",
            "core",
            pl.col("LapTime").filter(pl.col("Compound") == "SOFT").min(),
            "laps",
            fill_null=999.0,
        ),
        FeatureSpec(
            "min_lap_time_medium",
            "core",
            pl.col("LapTime").filter(pl.col("Compound") == "MEDIUM").min(),
            "laps",
            fill_null=999.0,
        ),
        FeatureSpec(
            "min_lap_time_hard",
            "core",
            pl.col("LapTime").filter(pl.col("Compound") == "HARD").min(),
            "laps",
            fill_null=999.0,
        ),
        FeatureSpec(
            "min_lap_time_wet",
            "core",
            pl.col("LapTime").filter(pl.col("Compound") == "WET").min(),
            "laps",
            fill_null=999.0,
        ),
        FeatureSpec(
            "min_lap_time_intermediate",
            "core",
            pl.col("LapTime").filter(pl.col("Compound") == "INTERMEDIATE").min(),
            "laps",
            fill_null=999.0,
        ),
        FeatureSpec(
            "long_run_pace",
            "core",
            pl.col("LapTime").filter(pl.col("TyreLife") > 5).mean(),
            "laps",
            fill_null=999.0,
        ),
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

LOOKBACK_FEATURES = FeatureRegistry(
    global_specs=[
        GlobalFeatureSpec("avg_grid_pos", "lookback", avg_grid_pos(n=5, alpha=0.5)),
        GlobalFeatureSpec("avg_finish_pos", "lookback", avg_finish_pos(n=5, alpha=0.5)),
        GlobalFeatureSpec("cum_champ_points", "lookback", cum_champ_points),
        GlobalFeatureSpec("dnf_rate", "lookback", dnf_rate(n=10, alpha=0.3)),
        GlobalFeatureSpec(
            "track_avg_finish", "lookback", track_avg_finish(n=5, alpha=0.3)
        ),
    ],
)
