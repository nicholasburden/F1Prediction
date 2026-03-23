from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from f1prediction.data.features import WEEKEND_SESSION_ORDER

# Sessions whose results can serve as prediction targets.
TARGET_SESSIONS = ("Q", "Sprint", "R")


@dataclass(frozen=True)
class RaceEntry:
    year: int
    event_slug: str
    driver: str  # abbreviation


@dataclass(frozen=True)
class PredictionSample:
    """One training/inference sample.

    Attributes:
        year: Season year.
        event_slug: Directory name for the event.
        driver: Driver abbreviation.
        visible_sessions: Sessions available as features at the prediction point.
        target_session: The session whose result we predict ("Q", "Sprint", or "R").
    """

    year: int
    event_slug: str
    driver: str
    visible_sessions: tuple[str, ...]
    target_session: str  # "Q" | "Sprint" | "R"


def build_vocabularies(data_dir: Path, years: list[int]) -> tuple[list[str], list[str]]:
    """Scan all results.parquet to build sorted driver and team vocabularies."""
    drivers: set[str] = set()
    teams: set[str] = set()

    for year in years:
        year_dir = data_dir / str(year)
        if not year_dir.exists():
            continue
        for results_path in year_dir.rglob("results.parquet"):
            df = pl.read_parquet(results_path, columns=["Abbreviation", "TeamName"])
            drivers.update(
                df["Abbreviation"].drop_nulls().unique().to_list()
            )
            teams.update(
                df["TeamName"].drop_nulls().unique().to_list()
            )

    return sorted(drivers), sorted(teams)


def discover_entries(data_dir: Path, years: list[int]) -> list[RaceEntry]:
    """Scan data dir, return entries with valid race results."""
    entries = []
    for year in years:
        year_dir = data_dir / str(year)
        if not year_dir.exists():
            continue
        for event_dir in sorted(year_dir.iterdir()):
            if not event_dir.is_dir():
                continue
            race_results = event_dir / "R" / "results.parquet"
            if not race_results.exists():
                continue
            df = pl.read_parquet(race_results, columns=["Abbreviation", "Position"])
            # Only include drivers with a valid finishing position
            valid = df.filter(pl.col("Position").is_not_null())
            for abbr in valid["Abbreviation"].to_list():
                entries.append(RaceEntry(year, event_dir.name, abbr))
    return entries


def generate_samples(
    data_dir: Path,
    events: list[tuple[int, str]],
) -> list[PredictionSample]:
    """Generate all prediction samples for the given events.

    For each event, for each target session (Q, Sprint, R) that has results,
    for each prefix of visible sessions leading up to that target, for each
    driver with a valid position in the target session, create a sample.

    NOTE: When historical/rolling features are added, they will use results
    from ALL chronologically preceding events regardless of train/val/test
    split. This means train-set features may contain val/test-set targets
    (a mild form of leakage). If val metrics look suspiciously good, consider
    implementing split-aware history: train samples only look back at train
    events, while val/test samples use full history.
    """
    samples: list[PredictionSample] = []

    for year, event_slug in events:
        event_dir = data_dir / str(year) / event_slug

        # Determine which pre-race sessions exist on disk
        present = tuple(
            s for s in WEEKEND_SESSION_ORDER
            if (event_dir / s).exists()
        )

        for target in TARGET_SESSIONS:
            # Check that the target session has results
            results_path = event_dir / target / "results.parquet"
            if not results_path.exists():
                continue

            df = pl.read_parquet(results_path, columns=["Abbreviation", "Position"])
            valid = df.filter(pl.col("Position").is_not_null())
            drivers = valid["Abbreviation"].to_list()
            if not drivers:
                continue

            # Visible sessions = everything chronologically before the target.
            # R is not in WEEKEND_SESSION_ORDER, so for target=R all present
            # sessions are visible. For Q/Sprint, visible is the prefix
            # before that session in the weekend order.
            if target in present:
                idx = present.index(target)
                max_visible = present[:idx]
            else:
                max_visible = present

            # Generate one sample per visibility prefix per driver.
            # prefix_len=0 → predict from embeddings/era only (pre-weekend).
            for prefix_len in range(len(max_visible) + 1):
                visible = max_visible[:prefix_len]
                for driver in drivers:
                    samples.append(PredictionSample(
                        year=year,
                        event_slug=event_slug,
                        driver=driver,
                        visible_sessions=visible,
                        target_session=target,
                    ))

    return samples


def get_event_order(data_dir: Path, years: list[int]) -> list[tuple[int, str]]:
    """Chronological ordering of events (sorted by year then directory name)."""
    events = []
    for year in sorted(years):
        year_dir = data_dir / str(year)
        if not year_dir.exists():
            continue
        for event_dir in sorted(year_dir.iterdir()):
            if not event_dir.is_dir():
                continue
            if (event_dir / "R" / "results.parquet").exists():
                events.append((year, event_dir.name))
    return events


def split_events(
    event_order: list[tuple[int, str]],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    shuffle: bool = True,
    seed: int = 42,
) -> tuple[list[tuple[int, str]], list[tuple[int, str]], list[tuple[int, str]]]:
    """Split events into train/val/test sets.

    When shuffle=True (default), events are shuffled before splitting so
    that all splits see a representative mix of tracks and conditions.
    """
    events = list(event_order)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(events)

    n_events = len(events)
    n_train = round(n_events * train_frac)
    n_val = round(n_events * val_frac)

    train = events[:n_train]
    val = events[n_train:n_train + n_val]
    test = events[n_train + n_val:]
    return train, val, test


def time_series_split(
    entries: list[RaceEntry],
    event_order: list[tuple[int, str]],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    shuffle: bool = True,
    seed: int = 42,
) -> tuple[list[RaceEntry], list[RaceEntry], list[RaceEntry]]:
    """Split entries by event into train/val/test.

    When shuffle=True (default), events are shuffled before splitting so
    that all splits see a representative mix of tracks and conditions.
    All drivers in the same event always go to the same split.
    """
    train_events, val_events, test_events = split_events(
        event_order, train_frac, val_frac, shuffle, seed,
    )
    train_set = set(train_events)
    val_set = set(val_events)

    train, val, test = [], [], []
    for entry in entries:
        key = (entry.year, entry.event_slug)
        if key in train_set:
            train.append(entry)
        elif key in val_set:
            val.append(entry)
        else:
            test.append(entry)

    return train, val, test
