"""Feature pipeline — builds a long-format feature DataFrame.

One row per (Driver, Year, EventId, SessionId). Event-level features are
broadcast across sessions.
"""

from __future__ import annotations

from pathlib import Path
import polars as pl

from f1prediction.data.constants import DataTable, Session
from f1prediction.data.features import attach_session_ord
from f1prediction.data.registry import SESSION_KEYS, FeatureRegistry

_COMPOUND_MAP: dict[str, str] = {
    "HYPERSOFT": "SOFT",
    "ULTRASOFT": "SOFT",
    "SUPERSOFT": "SOFT",
    "SOFT": "SOFT",
    "MEDIUM": "MEDIUM",
    "HARD": "HARD",
    "SUPERHARD": "HARD",
    "INTERMEDIATE": "INTERMEDIATE",
    "WET": "WET",
}


def _encode_categoricals(
    df: pl.DataFrame,
    onehot: list[str],
    embedding: list[str],
    mappings: dict[str, dict[object, int]] | None = None,
) -> tuple[pl.DataFrame, dict[str, int], dict[str, dict[object, int]]]:
    """Encode embedding columns to int ids. If ``mappings`` is provided, reuse
    the saved mapping (with ``default=0`` for unseen values); otherwise derive
    a fresh mapping from the data. Returns ``(df, vocab_sizes, mappings)``."""
    if onehot:
        df = df.to_dummies(onehot)
    vocab_size: dict[str, int] = {}
    out_mappings: dict[str, dict[object, int]] = {}
    for col in embedding:
        if mappings is not None and col in mappings:
            mapping = mappings[col]
        else:
            vocab = sorted(df[col].drop_nulls().unique().to_list())
            mapping = {v: i + 1 for i, v in enumerate(vocab)}
        out_mappings[col] = mapping
        vocab_size[col] = len(mapping) + 1
        df = df.with_columns(pl.col(col).replace(mapping, default=0).cast(pl.Int32))
    return df, vocab_size, out_mappings


def _get_track_id(event_dir: Path) -> str:
    return event_dir.name[3:][:-11]


def _load_data(data_dir: Path, years: list[int]) -> dict[DataTable, pl.LazyFrame]:
    """Scan all parquet files for the given years and return lazy frames.

    Returns a dict with keys "laps", "results", "weather", each a LazyFrame
    with added columns: Year (i32), EventId (i32), SessionId (str), TrackName (str).
    Call .collect() on each when you need a materialised DataFrame.
    """
    frames: dict[DataTable, list[pl.LazyFrame]] = {
        "laps": [],
        "results": [],
        "weather": [],
    }

    for year in years:
        year_dir = data_dir / str(year)
        if not year_dir.exists():
            continue
        for event_path in sorted(year_dir.iterdir()):
            if not event_path.is_dir():
                continue
            event_idx = int(event_path.name.split("_")[0])
            track_name = _get_track_id(event_path)
            for session_dir in sorted(event_path.iterdir()):
                if not session_dir.is_dir():
                    continue
                try:
                    session_id = Session(session_dir.name).value
                except ValueError:
                    continue
                for file_type in frames:
                    p = session_dir / f"{file_type}.parquet"
                    if not p.exists():
                        continue
                    lf = pl.scan_parquet(p).with_columns(
                        pl.lit(year).alias("Year"),
                        pl.lit(event_idx).alias("EventId"),
                        pl.lit(session_id).alias("SessionId"),
                        pl.lit(track_name).alias("TrackName"),
                    )
                    if file_type == "results":
                        lf = lf.rename({"Abbreviation": "Driver"})
                    elif file_type == "laps":
                        lf = lf.with_columns(
                            pl.col("Compound").replace_strict(
                                _COMPOUND_MAP, default=None
                            ),
                            (
                                pl.col("LapNumber")
                                - pl.col("LapNumber")
                                .min()
                                .over(["Driver", "Stint"])
                                + 1
                            ).alias("TyreLife"),
                        )
                    frames[file_type].append(lf)

    return {
        file_type: pl.concat(lfs, how="diagonal_relaxed")
        for file_type, lfs in frames.items()
        if lfs
    }


def build_features(
    data_dir: Path,
    years: list[int],
    feature_registry: FeatureRegistry,
    vocab_mappings: dict[str, dict[object, int]] | None = None,
) -> tuple[pl.DataFrame, dict[str, int], dict[str, dict[object, int]]]:
    data = _load_data(data_dir, years)

    features = (
        feature_registry.apply_group("laps", data)
        .join(
            feature_registry.apply_group("results", data),
            on=list(SESSION_KEYS["results"]),
            how="outer_coalesce",  # type: ignore[arg-type]
        )
        .join(
            feature_registry.apply_group("weather", data),
            on=list(SESSION_KEYS["weather"]),
            how="left",
        )
    )

    # Position is joined in here (before apply_global) so global feature
    # functions can reference it (e.g. for cumulative championship points).
    # Selection of feature columns downstream is name-based via the registry,
    # so Position does not leak into the model inputs — but any refactor that
    # iterates all columns as features would leak the target.
    event_keys = ["SessionId", "EventId", "Year"]
    num_drivers = (
        data["results"]
        .group_by(event_keys)
        .agg(pl.col("Driver").n_unique().alias("NumDrivers"))
    )

    features = (
        features.join(
            data["results"].select(SESSION_KEYS["results"] + ("Position",)),
            on=SESSION_KEYS["results"],
        )
        .join(
            num_drivers,
            on=event_keys,
        )
        .with_columns(
            pl.when(pl.col("SessionId").is_in(["Q", "R", "Sprint"]))
            .then(pl.col("Position").fill_null(pl.col("NumDrivers")))
            .otherwise(pl.col("Position").fill_null(0))
            .alias("Position"),
            pl.col("grid_position")
            .fill_null(pl.col("NumDrivers"))
            .alias("grid_position"),
        )
    )

    features = feature_registry.apply_global(features)
    features = attach_session_ord(features)

    fill_map = feature_registry.null_fill_map
    features = features.with_columns(
        [pl.col(name).fill_null(value) for name, value in fill_map.items()]
    )
    result = features.collect()

    onehot = feature_registry.onehot_features
    embedding = feature_registry.embedding_features

    vocab_len_dict: dict[str, int] = {}
    out_mappings: dict[str, dict[object, int]] = {}
    if onehot or embedding:
        result, vocab_len_dict, out_mappings = _encode_categoricals(
            result, onehot, embedding, vocab_mappings
        )

    return result, vocab_len_dict, out_mappings
