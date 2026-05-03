"""Predict driver finishing order for an F1 session.

Loads a trained model from a runs directory, builds features for the target
event (using the same pipeline as training), and produces an ordered list of
drivers from predicted-best to predicted-worst.

If the target session has not happened yet, a placeholder ``results.parquet``
is written so the feature pipeline can join target rows; the placeholder is
deleted on exit. The driver list is taken from that file if present, otherwise
inferred from the most recent preceding session of the same event.

Note that the model only uses features from sessions chronologically *before*
the target session, so weather, lap, and result data of the target session are
never read by the model — only data from preceding sessions is required.

Usage:
    uv run python scripts/inference.py \\
        --run-dir runs/<group>/<run_name> \\
        --year 2026 --event 4 --session R
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl
import torch

from f1prediction.config import Config
from f1prediction.data.dataset import (
    SESSION_ORDER,
    DatasetSchema,
    NormStats,
    _base_name,
    _pivot_df,
)
from f1prediction.data.features import CORE_FEATURES, LOOKBACK_FEATURES
from f1prediction.data.pipeline import build_features
from f1prediction.data.registry import FeatureRegistry
from f1prediction.models.mlp import MLPModel

_FEATURE_SETS: dict[str, FeatureRegistry] = {
    "core": CORE_FEATURES,
    "lookback": LOOKBACK_FEATURES,
}


def _resolve_event(
    data_dir: Path, year: int, event: int | str, allow_create: bool = False
) -> tuple[int, Path]:
    """Locate the event directory for (year, event). When ``allow_create`` and
    the event is not yet on disk, query fastf1 for its slug and return the
    path that should be created.
    """
    year_dir = data_dir / str(year)
    matches: list[tuple[int, Path]] = []
    if year_dir.is_dir():
        for entry in sorted(year_dir.iterdir()):
            if not entry.is_dir():
                continue
            try:
                round_num = int(entry.name.split("_")[0])
            except ValueError:
                continue
            if isinstance(event, int):
                if round_num == event:
                    matches.append((round_num, entry))
            else:
                if event.lower() in entry.name.lower():
                    matches.append((round_num, entry))
    if matches:
        if len(matches) > 1:
            names = [m[1].name for m in matches]
            raise ValueError(f"Multiple events match {event!r}: {names}")
        return matches[0]

    if not allow_create:
        raise ValueError(f"No event matching {event!r} in {year_dir}")

    import fastf1

    from f1prediction.data.download import _event_slug

    schedule = fastf1.get_event_schedule(year, include_testing=False)
    if isinstance(event, int):
        rows = schedule[schedule["RoundNumber"] == event]
    else:
        rows = schedule[
            schedule["EventName"].str.lower().str.contains(event.lower())
        ]
    if rows.empty:
        raise ValueError(f"fastf1 has no event matching {event!r} for {year}")
    if len(rows) > 1:
        raise ValueError(
            f"Multiple fastf1 events match {event!r}: "
            f"{rows['EventName'].tolist()}"
        )
    row = rows.iloc[0]
    round_num = int(row["RoundNumber"])
    return round_num, year_dir / _event_slug(round_num, str(row["EventName"]))


def _drivers_from_preceding(
    event_path: Path, target: str
) -> list[tuple[str, str]]:
    target_idx = SESSION_ORDER.index(target)
    preceding: list[tuple[int, Path]] = []
    for sess in SESSION_ORDER:
        idx = SESSION_ORDER.index(sess)
        if idx >= target_idx:
            continue
        results = event_path / sess / "results.parquet"
        if results.exists():
            preceding.append((idx, results))
    if not preceding:
        raise ValueError(
            f"No preceding session results in {event_path}; "
            f"cannot infer driver list for {target}"
        )
    preceding.sort(reverse=True)
    df = pl.read_parquet(preceding[0][1])
    return [
        (row["Abbreviation"], row["TeamName"])
        for row in df.select("Abbreviation", "TeamName").to_dicts()
    ]


def _synthesise_target_results(
    event_path: Path, session: str, drivers: list[tuple[str, str]]
) -> Path | None:
    """Write a minimal placeholder results.parquet so the pipeline produces a
    target-session row per driver. Returns the file path if written, else None.
    """
    sess_dir = event_path / session
    results_path = sess_dir / "results.parquet"
    if results_path.exists():
        return None
    sess_dir.mkdir(parents=True, exist_ok=True)
    n = len(drivers)
    df = pl.DataFrame(
        {
            "Abbreviation": [d for d, _ in drivers],
            "TeamName": [t for _, t in drivers],
            "Position": [float(i + 1) for i in range(n)],
            "GridPosition": [None] * n,
            "Q1": [None] * n,
            "Q2": [None] * n,
            "Q3": [None] * n,
            "Points": [None] * n,
            "Status": [""] * n,
        },
        schema={
            "Abbreviation": pl.Utf8,
            "TeamName": pl.Utf8,
            "Position": pl.Float64,
            "GridPosition": pl.Float64,
            "Q1": pl.Float64,
            "Q2": pl.Float64,
            "Q3": pl.Float64,
            "Points": pl.Float64,
            "Status": pl.Utf8,
        },
    )
    df.write_parquet(results_path)
    return results_path


def _download_missing_preceding(
    year: int,
    round_num: int,
    event_path: Path,
    target_session: str,
    cache_dir: Path,
) -> None:
    """Download any sessions of (year, round_num) that precede ``target_session``,
    are missing on disk, and have already started. Errors are logged but not
    raised — missing data falls through to schema-fill defaults during inference.
    """
    from datetime import datetime, timezone

    import fastf1
    import pandas as pd

    from f1prediction.data.download import (
        SESSION_ABBREVS,
        _save_laps,
        _save_results,
        _save_weather,
        _session_dir_exists,
    )

    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))

    schedule = fastf1.get_event_schedule(year, include_testing=False)
    rows = schedule[schedule["RoundNumber"] == round_num]
    if rows.empty:
        print(
            f"[inference] fastf1 has no schedule entry for {year} round {round_num}",
            file=sys.stderr,
        )
        return
    event_row = rows.iloc[0]

    now = datetime.now(timezone.utc)
    target_idx = SESSION_ORDER.index(target_session)

    for i in range(1, 6):
        sess_name = event_row.get(f"Session{i}")
        sess_date = event_row.get(f"Session{i}Date")
        if not isinstance(sess_name, str) or not sess_name.strip():
            continue
        abbrev = SESSION_ABBREVS.get(sess_name.strip())
        if abbrev is None or abbrev not in SESSION_ORDER:
            continue
        if SESSION_ORDER.index(abbrev) >= target_idx:
            continue

        out_dir = event_path / abbrev
        if _session_dir_exists(out_dir):
            continue
        if sess_date is None or pd.isna(sess_date):
            continue
        if hasattr(sess_date, "to_pydatetime"):
            sess_date = sess_date.to_pydatetime()
        if sess_date.tzinfo is None:
            sess_date = sess_date.replace(tzinfo=timezone.utc)
        if sess_date > now:
            print(
                f"[inference] skipping {abbrev}: starts {sess_date.isoformat()} "
                "(in the future)",
                file=sys.stderr,
            )
            continue

        print(
            f"[inference] downloading {year}/{event_path.name}/{abbrev}…",
            file=sys.stderr,
        )
        try:
            session = fastf1.get_session(year, round_num, sess_name)
            session.load(laps=True, telemetry=False, weather=True, messages=False)
            _save_results(session, out_dir)
            _save_laps(session, out_dir)
            _save_weather(session, out_dir)
        except Exception as exc:
            print(
                f"[inference] download of {abbrev} failed: {exc}",
                file=sys.stderr,
            )


def _build_inference_inputs(
    all_features: pl.DataFrame,
    feature_registry: FeatureRegistry,
    schema: DatasetSchema,
    year: int,
    event_id: int,
    target_session: str,
) -> tuple[torch.Tensor, torch.Tensor, list[str], int]:
    """Construct (X, cat_ids, drivers, num_drivers) for one (year, event,
    session) prediction. Mirrors the training data flow but bypasses
    ``_attach_targets`` since there is no Target to compute. Uses the
    per-format ``_session_ord`` from build_features rather than a static
    SESSION_ORDER index, so 2021-22 sprint formats correctly exclude sessions
    that happen chronologically after the target."""
    target_rows = all_features.filter(
        (pl.col("Year") == year)
        & (pl.col("EventId") == event_id)
        & (pl.col("SessionId") == target_session)
    )
    if target_rows.is_empty():
        raise ValueError(
            f"No target-session row for year={year} event={event_id} "
            f"session={target_session}; the placeholder synthesis step failed."
        )
    target_drivers = sorted(target_rows["Driver"].unique().to_list())
    target_num_drivers = int(target_rows["NumDrivers"][0])
    target_ord = int(target_rows["_session_ord"][0])

    rows = all_features.filter(
        (pl.col("Year") == year)
        & (pl.col("EventId") == event_id)
        & (pl.col("_session_ord") < target_ord)
    )
    if rows.is_empty():
        raise ValueError(
            f"No preceding session data for year={year} event={event_id} "
            f"target={target_session}; the target session is the first available."
        )

    rows = (
        rows.filter(pl.col("Driver").is_in(target_drivers))
        .drop("_session_ord", "_has_sprint")
        .with_columns(
            pl.lit(target_session).alias("TargetSession"),
            pl.lit(target_num_drivers).cast(pl.UInt32).alias("TargetNumDrivers"),
            pl.lit(0.0).alias("Target"),
        )
    )

    sessions = rows["SessionId"].unique().to_list()
    pivoted = _pivot_df(rows, feature_registry.event_wide_features)

    fill_map = feature_registry.null_fill_map
    pivoted = pivoted.with_columns(
        [
            pl.col(c).fill_null(fill_map.get(_base_name(c, sessions), 0.0))
            for c in pivoted.columns
            if pivoted[c].null_count() > 0
        ]
    )

    for c in schema.numeric_cols:
        if c not in pivoted.columns:
            pivoted = pivoted.with_columns(pl.lit(0.0).alias(c))
    for c in schema.embedding_cols:
        if c not in pivoted.columns:
            pivoted = pivoted.with_columns(pl.lit(0).cast(pl.Int64).alias(c))

    drivers = pivoted["Driver"].to_list()
    X = torch.tensor(
        pivoted.select(schema.numeric_cols).to_numpy(), dtype=torch.float32
    )
    X = schema.norm_stats.apply(X)
    if schema.embedding_cols:
        cat_ids = torch.tensor(
            pivoted.select(schema.embedding_cols).to_numpy(), dtype=torch.long
        )
    else:
        cat_ids = torch.zeros(len(drivers), 0, dtype=torch.long)
    return X, cat_ids, drivers, target_num_drivers


def predict_session(
    run_dir: Path,
    year: int,
    event: int | str,
    session: str,
    data_dir: Path,
    auto_download: bool = True,
    cache_dir: Path = Path("cache"),
) -> list[tuple[str, float]]:
    """Return a list of (driver, predicted_position) sorted best-first."""
    cfg = Config(**json.loads((run_dir / "config.json").read_text()))
    checkpoint_path = run_dir / "checkpoint.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No checkpoint.pt in {run_dir}; the run must have been trained with "
            "save_model=True."
        )
    device = cfg.training.device
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    schema = DatasetSchema(
        numeric_cols=checkpoint["numeric_cols"],
        embedding_cols=checkpoint["embedding_cols"],
        norm_stats=NormStats(mean=checkpoint["norm_mean"], std=checkpoint["norm_std"]),
    )
    vocab_lens: list[int] = checkpoint["vocab_lens"]
    vocab_mappings: dict[str, dict[object, int]] = checkpoint["vocab_mappings"]

    round_num, event_path = _resolve_event(
        data_dir, year, event, allow_create=auto_download
    )
    if auto_download:
        event_path.mkdir(parents=True, exist_ok=True)
        _download_missing_preceding(
            year, round_num, event_path, session, cache_dir
        )
    event_id = round_num

    target_results = event_path / session / "results.parquet"
    synthesised: Path | None = None
    if not target_results.exists():
        drivers_meta = _drivers_from_preceding(event_path, session)
        synthesised = _synthesise_target_results(event_path, session, drivers_meta)
        if synthesised is not None:
            print(
                f"[inference] synthesised placeholder {synthesised.relative_to(data_dir)}"
                " (target session has no on-disk results)",
                file=sys.stderr,
            )

    try:
        feature_registry = sum(
            (_FEATURE_SETS[name] for name in cfg.training.feature_sets[1:]),
            _FEATURE_SETS[cfg.training.feature_sets[0]],
        )
        all_features, _, _ = build_features(
            data_dir,
            cfg.training.years,
            feature_registry,
            vocab_mappings=vocab_mappings,
        )

        model = MLPModel(cfg.model, len(schema.numeric_cols), vocab_lens).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        X, cat_ids, drivers, num_drivers = _build_inference_inputs(
            all_features, feature_registry, schema, year, event_id, session
        )
        X, cat_ids = X.to(device), cat_ids.to(device)

        with torch.no_grad():
            preds = model(X, cat_ids) * num_drivers

        ordered = sorted(zip(drivers, preds.tolist()), key=lambda kv: kv[1])
        return ordered
    finally:
        if synthesised is not None and synthesised.exists():
            synthesised.unlink()
            try:
                synthesised.parent.rmdir()
            except OSError:
                pass


def _parse_event(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


def _next_upcoming_session(target_sessions: list[str]) -> tuple[int, int, str]:
    """Return (year, round_num, session_abbrev) of the soonest upcoming session
    that is in ``target_sessions``, queried from fastf1's schedule.
    """
    from datetime import datetime, timezone

    import fastf1
    import pandas as pd

    from f1prediction.data.download import SESSION_ABBREVS

    now = datetime.now(timezone.utc)
    candidates: list[tuple[datetime, int, int, str]] = []

    for year in (now.year, now.year + 1):
        try:
            schedule = fastf1.get_event_schedule(year, include_testing=False)
        except Exception:
            continue
        for _, event in schedule.iterrows():
            round_num = int(event.get("RoundNumber") or 0)
            if round_num <= 0:
                continue
            for i in range(1, 6):
                sess_name = event.get(f"Session{i}")
                sess_date = event.get(f"Session{i}Date")
                if not isinstance(sess_name, str) or not sess_name.strip():
                    continue
                abbrev = SESSION_ABBREVS.get(sess_name.strip())
                if abbrev is None or abbrev not in target_sessions:
                    continue
                if sess_date is None or pd.isna(sess_date):
                    continue
                if hasattr(sess_date, "to_pydatetime"):
                    sess_date = sess_date.to_pydatetime()
                if sess_date.tzinfo is None:
                    sess_date = sess_date.replace(tzinfo=timezone.utc)
                if sess_date > now:
                    candidates.append((sess_date, year, round_num, abbrev))

    if not candidates:
        raise RuntimeError(
            f"fastf1 returned no upcoming sessions in {target_sessions}"
        )
    candidates.sort()
    return candidates[0][1], candidates[0][2], candidates[0][3]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict driver finishing order for an F1 session."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to the run directory (contains config.json and model.pt).",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Defaults to the year of the next upcoming target session.",
    )
    parser.add_argument(
        "--event",
        type=_parse_event,
        default=None,
        help="Round number or substring of the event slug. "
        "Defaults to the next upcoming target session.",
    )
    parser.add_argument(
        "--session",
        default=None,
        choices=[s for s in SESSION_ORDER],
        help="Target session to predict. Defaults to the next upcoming session "
        "in the model's training target_sessions.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Disable auto-download of missing preceding sessions for the target event.",
    )
    args = parser.parse_args()

    if args.year is None or args.event is None or args.session is None:
        cfg = Config(**json.loads((args.run_dir / "config.json").read_text()))
        year, round_num, session = _next_upcoming_session(cfg.training.target_sessions)
        if args.year is None:
            args.year = year
        if args.event is None:
            args.event = round_num
        if args.session is None:
            args.session = session
        print(
            f"[inference] auto-selected next upcoming session: "
            f"{args.year} round {args.event} {args.session}",
            file=sys.stderr,
        )

    ordered = predict_session(
        args.run_dir,
        args.year,
        args.event,
        args.session,
        args.data_dir,
        auto_download=not args.no_download,
    )
    width = max(len(d) for d, _ in ordered)
    print(f"Predicted finishing order — {args.year} event {args.event} {args.session}:")
    for rank, (driver, pred) in enumerate(ordered, start=1):
        print(f"  {rank:>2}.  {driver:<{width}}  (pred_pos={pred:5.2f})")


if __name__ == "__main__":
    main()
