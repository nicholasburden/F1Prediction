"""Integration test — build features from real data and compare against a saved snapshot."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from f1prediction.data.constants import DATA_DIR
from f1prediction.data.features import ALL_FEATURES
from f1prediction.data.pipeline import build_features

SNAPSHOT_PATH = Path(__file__).parent / "snapshots" / "features_2020.parquet"
YEARS = [2020]


@pytest.fixture(scope="module")
def features_df() -> pl.DataFrame:
    df, _ = build_features(DATA_DIR, YEARS, ALL_FEATURES)
    return df


def test_no_nulls(features_df: pl.DataFrame) -> None:
    for col in features_df.columns:
        assert features_df[col].null_count() == 0, (
            f"{col} has {features_df[col].null_count()} nulls"
        )


def test_year_values(features_df: pl.DataFrame) -> None:
    assert features_df["Year"].unique().to_list() == [2020]


def test_expected_event_count(features_df: pl.DataFrame) -> None:
    n_events = features_df.n_unique(subset=["Year", "EventId"])
    assert n_events == 17


def test_drivers_per_event_reasonable(features_df: pl.DataFrame) -> None:
    counts = features_df.group_by(["Year", "EventId", "SessionId"]).agg(
        pl.col("Driver").n_unique().alias("n")
    )
    assert int(counts["n"].min()) >= 15  # type: ignore[arg-type]
    assert int(counts["n"].max()) <= 25  # type: ignore[arg-type]


def test_era_features_constant_for_year(features_df: pl.DataFrame) -> None:
    for col in [
        "era_min_weight",
        "era_fuel_limit",
        "era_budget_cap",
        "era_ground_effect",
        "era_has_drs",
    ]:
        assert features_df[col].n_unique() == 1, (
            f"{col} should be constant for a single year"
        )


def test_lap_times_positive(features_df: pl.DataFrame) -> None:
    valid = features_df.filter(pl.col("min_lap_time") > 0)
    assert (valid["min_lap_time"] > 0).all()
    assert len(valid) > 0


def test_speeds_reasonable(features_df: pl.DataFrame) -> None:
    for col in ["max_speed_i1", "max_speed_i2", "max_speed_fl", "max_speed_st"]:
        valid = features_df.filter(pl.col(col) > 0)
        if len(valid) == 0:
            continue
        assert float(valid[col].min()) > 50, f"{col} has unreasonably low speed"  # type: ignore[arg-type]
        assert float(valid[col].max()) < 400, f"{col} has unreasonably high speed"  # type: ignore[arg-type]


def test_position_present(features_df: pl.DataFrame) -> None:
    assert "Position" in features_df.columns


def test_grid_position_norm_bounded(features_df: pl.DataFrame) -> None:
    valid = features_df.filter(pl.col("grid_position_norm").is_not_null())
    assert float(valid["grid_position_norm"].min()) >= 0  # type: ignore[arg-type]
    assert float(valid["grid_position_norm"].max()) <= 1.0  # type: ignore[arg-type]


def test_gap_to_fastest_nonnegative(features_df: pl.DataFrame) -> None:
    valid = features_df.filter(pl.col("gap_to_fastest").is_not_null())
    assert float(valid["gap_to_fastest"].min()) >= 0.0  # type: ignore[arg-type]


def test_snapshot_matches(features_df: pl.DataFrame) -> None:
    if not SNAPSHOT_PATH.exists():
        pytest.skip("No snapshot found — run with --snapshot to generate")
    expected = pl.read_parquet(SNAPSHOT_PATH)
    assert features_df.schema == expected.schema, "Schema mismatch"
    assert features_df.shape == expected.shape, (
        f"Shape mismatch: {features_df.shape} vs {expected.shape}"
    )
    sorted_actual = features_df.sort(["Driver", "Year", "EventId", "SessionId"])
    sorted_expected = expected.sort(["Driver", "Year", "EventId", "SessionId"])
    assert sorted_actual.equals(sorted_expected), "Feature values differ from snapshot"


def test_generate_snapshot(
    features_df: pl.DataFrame, request: pytest.FixtureRequest
) -> None:
    if not request.config.getoption("--snapshot", default=False):
        pytest.skip("Pass --snapshot to regenerate")
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features_df.sort(["Driver", "Year", "EventId", "SessionId"]).write_parquet(
        SNAPSHOT_PATH
    )
