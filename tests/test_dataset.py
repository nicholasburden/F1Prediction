"""Tensor-level tests — assert exact values from a fixed-seed dataloader run."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import torch
import pytest

from f1prediction.data.constants import DATA_DIR
from f1prediction.data.dataset import _attach_targets
from f1prediction.data.features import CORE_FEATURES
from f1prediction.data.dataloader import get_dataloaders
from f1prediction.data.pipeline import build_features

SNAPSHOT_PATH = Path(__file__).parent / "snapshots" / "dataloader_seed42.pt"
SEED = 42
YEARS = [2020]
BATCH_SIZE = 4


@pytest.fixture(scope="module")
def dataloaders():
    all_data, _, _ = build_features(DATA_DIR, YEARS, CORE_FEATURES)
    train_dl, val_dl, test_dl, _ = get_dataloaders(
        all_data, CORE_FEATURES, seed=SEED, batch_size=BATCH_SIZE
    )
    return train_dl, val_dl, test_dl


@pytest.fixture(scope="module")
def first_batch(dataloaders):
    train_dl, _, _ = dataloaders
    return next(iter(train_dl))


@pytest.fixture(scope="module")
def expected():
    assert SNAPSHOT_PATH.exists(), (
        f"Snapshot not found at {SNAPSHOT_PATH} — run with --snapshot to generate"
    )
    return torch.load(SNAPSHOT_PATH, weights_only=True)


def test_x_values(first_batch, expected) -> None:
    x, _, _, _ = first_batch
    torch.testing.assert_close(x, expected["x"])


def test_cat_ids_values(first_batch, expected) -> None:
    _, cat, _, _ = first_batch
    assert torch.equal(cat, expected["cat"])


def test_y_values(first_batch, expected) -> None:
    _, _, y, _ = first_batch
    torch.testing.assert_close(y, expected["y"])


def test_split_sizes(dataloaders, expected) -> None:
    train_dl, val_dl, test_dl = dataloaders
    assert len(train_dl.dataset) == expected["train_len"]
    assert len(val_dl.dataset) == expected["val_len"]
    assert len(test_dl.dataset) == expected["test_len"]


def test_cutoff_variants_per_target() -> None:
    """For each (Driver, Year, EventId, TargetSession), the partial-weekend
    augmentation should emit exactly ``_target_session_ord + 1`` variants —
    one per cutoff in ``[0, target_ord]``."""
    long_df, _, _ = build_features(DATA_DIR, [2020], CORE_FEATURES)
    masked = CORE_FEATURES.session_specific_features + ["Position"]
    attached = _attach_targets(long_df, ["Sprint", "Q", "R"], masked)

    variant_keys = (
        attached.lazy()
        .select("Driver", "Year", "EventId", "TargetSession", "_cutoff")
        .unique()
        .collect()
    )
    counts = variant_keys.group_by(
        ["Driver", "Year", "EventId", "TargetSession"]
    ).agg(pl.col("_cutoff").n_unique().alias("n_variants"))

    target_ord_lookup = (
        long_df.select("Year", "EventId", "SessionId", "_session_ord")
        .unique()
        .rename({"SessionId": "TargetSession", "_session_ord": "_target_session_ord"})
    )
    expected = counts.join(
        target_ord_lookup, on=["Year", "EventId", "TargetSession"]
    )
    # 2020 is non-sprint: Q at ord 3 → 4 variants, R at ord 4 → 5.
    assert (
        expected["n_variants"] == expected["_target_session_ord"] + 1
    ).all(), expected


def test_cutoff_keeps_weather_for_future_sessions() -> None:
    """A cutoff variant that hides FP3 should still see FP3's weather (the
    inference path provides forecast weather for future sessions, so the
    model must train on this pattern)."""
    long_df, _, _ = build_features(DATA_DIR, [2020], CORE_FEATURES)
    masked = CORE_FEATURES.session_specific_features + ["Position"]
    attached = _attach_targets(long_df, ["R"], masked)

    # Pick a (Driver, Year, EventId, TargetSession=R) and cutoff=1 — only FP1
    # is "visible"; FP2/FP3/Q rows are future. Their weather column should be
    # populated; their pace column (min_lap_time) should be nulled.
    sample = attached.filter(
        (pl.col("TargetSession") == "R") & (pl.col("_cutoff") == 1)
    )
    future_rows = sample.filter(pl.col("SessionId").is_in(["FP2", "FP3", "Q"]))
    assert not future_rows.is_empty()
    # Weather feature retained on future sessions.
    weather_present = future_rows["mean_air_temp"].drop_nulls()
    assert len(weather_present) > 0, "weather should leak through cutoff"
    # Pace feature masked out on future sessions.
    assert future_rows["min_lap_time"].null_count() == len(future_rows)


def test_weather_block_dropout_zeros_all_weather_columns() -> None:
    """With weather_dropout=1.0, every weather feature column (across every
    session) must be jointly zeroed in every sample. Non-weather columns
    must be untouched."""
    from f1prediction.data.dataset import RacePredictionDataset

    all_data, _, _ = build_features(DATA_DIR, [2020], CORE_FEATURES)
    ds, schema = RacePredictionDataset.from_dataframe(
        all_data,
        CORE_FEATURES,
        target_sessions=["R"],
        block_dropout={"weather": 1.0},
    )
    weather_bases = set(CORE_FEATURES.feature_groups["weather"])
    weather_idx = [
        i for i, c in enumerate(schema.numeric_cols)
        if any(
            c.startswith(f"{b}_") or c == b or c == f"target_{b}"
            for b in weather_bases
        )
    ]
    non_weather_idx = [
        i for i in range(len(schema.numeric_cols)) if i not in weather_idx
    ]
    assert weather_idx, "expected at least one weather column in the pivot"

    x, _, _, _ = ds[0]
    assert (x[weather_idx] == 0).all()
    pre_drop_non_weather = ds.X[0][non_weather_idx]
    assert torch.equal(x[non_weather_idx], pre_drop_non_weather)


def test_block_dropout_disabled_by_default() -> None:
    """With no block_dropout config, the block dropout tensors are empty and
    __getitem__ leaves weather columns alone."""
    from f1prediction.data.dataset import RacePredictionDataset

    all_data, _, _ = build_features(DATA_DIR, [2020], CORE_FEATURES)
    ds, _ = RacePredictionDataset.from_dataframe(
        all_data, CORE_FEATURES, target_sessions=["R"],
    )
    assert ds.block_dropout_probs.numel() == 0


def test_target_weather_columns_exist_and_no_target_non_weather_columns() -> None:
    """The pivot should produce ``target_<weather>`` columns (carrying the
    target session's forecastable weather) but NOT per-session columns for
    the target session itself (no ``*_R`` for non-weather features when
    target=R)."""
    from f1prediction.data.dataset import RacePredictionDataset

    all_data, _, _ = build_features(DATA_DIR, [2020], CORE_FEATURES)
    _, schema = RacePredictionDataset.from_dataframe(
        all_data, CORE_FEATURES, target_sessions=["R"],
    )
    weather_bases = CORE_FEATURES.feature_groups["weather"]
    for base in weather_bases:
        assert f"target_{base}" in schema.numeric_cols, (
            f"expected target_{base} in numeric_cols"
        )
    # Non-weather session-specific feature: no R-suffixed column.
    for name in CORE_FEATURES.session_specific_features:
        assert f"{name}_R" not in schema.numeric_cols, (
            f"{name}_R should not be pivoted (target session, non-weather)"
        )


def test_generate_snapshot(
    dataloaders, first_batch, request: pytest.FixtureRequest
) -> None:
    if not request.config.getoption("--snapshot", default=False):
        pytest.skip("Pass --snapshot to regenerate")
    # Reuse first_batch — calling next(iter(train_dl)) again would advance the
    # DataLoader's generator and give a different shuffle than the other tests.
    train_dl, val_dl, test_dl = dataloaders
    x, cat, y, _ = first_batch
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "x": x,
            "cat": cat,
            "y": y,
            "train_len": len(train_dl.dataset),
            "val_len": len(val_dl.dataset),
            "test_len": len(test_dl.dataset),
        },
        SNAPSHOT_PATH,
    )
