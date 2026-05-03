from __future__ import annotations

from collections.abc import Iterator
from typing import cast

import torch

from f1prediction.data.constants import EventSample
from f1prediction.data.dataset import DatasetSchema, RacePredictionDataset
from f1prediction.data.registry import FeatureRegistry
from torch.utils.data import DataLoader
import numpy as np
import polars as pl

F1Batch = tuple[torch.Tensor, torch.IntTensor, torch.Tensor, torch.Tensor]


class F1DataLoader(DataLoader[F1Batch]):
    def __iter__(self) -> Iterator[F1Batch]:  # type: ignore[override]
        return cast(Iterator[F1Batch], super().__iter__())


def _get_splits(
    events: list[EventSample],
    seed: int,
    train_weight: float,
    val_weight: float,
) -> tuple[list[EventSample], list[EventSample], list[EventSample]]:
    assert train_weight > 0
    assert val_weight > 0
    assert train_weight + val_weight < 1

    train_end_idx = int(round(train_weight * len(events)))
    val_end_idx = int(round((train_weight + val_weight) * len(events)))

    assert train_end_idx > 0
    assert train_end_idx < val_end_idx
    assert val_end_idx < (len(events) - 1)

    rng = np.random.default_rng(seed)
    shuffled = list(events)
    rng.shuffle(shuffled)  # type: ignore[arg-type]
    return (
        shuffled[:train_end_idx],
        shuffled[train_end_idx:val_end_idx],
        shuffled[val_end_idx:],
    )


def _validate_splits(
    train_split: list[EventSample],
    val_split: list[EventSample],
    test_split: list[EventSample],
) -> None:
    def _assert_unique(split: list[EventSample]) -> None:
        assert len(list(set(split))) == len(split)

    def _assert_no_overlap(splits: list[list[EventSample]]) -> None:
        for i, split1 in enumerate(splits):
            for j, split2 in enumerate(splits):
                if i != j:
                    assert set(split1).isdisjoint(set(split2))

    splits = [train_split, val_split, test_split]
    for split in splits:
        _assert_unique(split)
    _assert_no_overlap([train_split, val_split, test_split])


def _filter_events(df: pl.DataFrame, events: list[EventSample]) -> pl.DataFrame:
    keys = pl.DataFrame(
        [(e.year, e.event_id) for e in events],
        schema=["Year", "EventId"],
        orient="row",
    )
    return df.join(keys, on=["Year", "EventId"])


def _get_kfold_splits(
    events: list[EventSample], k: int, fold_idx: int, seed: int
) -> tuple[list[EventSample], list[EventSample]]:
    """Return (train_events, val_events) for fold ``fold_idx`` of K. Events are
    deterministically shuffled with ``seed`` so all folds see the same global
    ordering."""
    assert k >= 2
    assert 0 <= fold_idx < k

    rng = np.random.default_rng(seed)
    shuffled = list(events)
    rng.shuffle(shuffled)  # type: ignore[arg-type]

    n = len(shuffled)
    fold_starts = [round(i * n / k) for i in range(k + 1)]
    val_start, val_end = fold_starts[fold_idx], fold_starts[fold_idx + 1]
    val = shuffled[val_start:val_end]
    train = shuffled[:val_start] + shuffled[val_end:]
    assert val and train
    return train, val


def _train_loader(
    dataset: RacePredictionDataset, batch_size: int, seed: int
) -> F1DataLoader:
    """Train DataLoader with deterministic per-epoch shuffling."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    return F1DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=gen)


def get_kfold_dataloaders(
    all_data: pl.DataFrame,
    feature_registry: FeatureRegistry,
    k: int,
    fold_idx: int,
    seed: int,
    batch_size: int = 8,
    target_sessions: list[str] = ["Sprint", "Q", "R"],
    training_feature_dropout: dict[str, float] | None = None,
) -> tuple[F1DataLoader, F1DataLoader, DatasetSchema]:
    """Build (train_dl, val_dl, schema) for one fold of K-fold CV at the event
    grain. Shuffle order is fixed by ``seed`` so folds are reproducible."""
    all_events = sorted(
        [
            EventSample(row[0], row[1])
            for row in all_data.select("Year", "EventId").unique().iter_rows()
        ],
        key=lambda es: (es.year, es.event_id),
    )
    train_split, val_split = _get_kfold_splits(all_events, k, fold_idx, seed)

    train_ds, schema = RacePredictionDataset.from_dataframe(
        _filter_events(all_data, train_split),
        feature_registry,
        target_sessions,
        feature_dropout=training_feature_dropout,
    )
    val_ds, _ = RacePredictionDataset.from_dataframe(
        _filter_events(all_data, val_split),
        feature_registry,
        target_sessions,
        schema,
    )
    return (
        _train_loader(train_ds, batch_size, seed),
        F1DataLoader(val_ds, batch_size=1),
        schema,
    )


def get_full_dataloader(
    all_data: pl.DataFrame,
    feature_registry: FeatureRegistry,
    seed: int,
    batch_size: int = 8,
    target_sessions: list[str] = ["Sprint", "Q", "R"],
    training_feature_dropout: dict[str, float] | None = None,
) -> tuple[F1DataLoader, DatasetSchema]:
    """Build a single dataloader over all events — no val/test split. The schema
    (numeric/embedding cols + norm stats) is derived from the full dataset."""
    train_ds, schema = RacePredictionDataset.from_dataframe(
        all_data,
        feature_registry,
        target_sessions,
        feature_dropout=training_feature_dropout,
    )
    return _train_loader(train_ds, batch_size, seed), schema


def get_dataloaders(
    all_data: pl.DataFrame,
    feature_registry: FeatureRegistry,
    seed: int,
    batch_size: int = 8,
    target_sessions: list[str] = ["Sprint", "Q", "R"],
    training_feature_dropout: dict[str, float] | None = None,
) -> tuple[F1DataLoader, F1DataLoader, F1DataLoader, DatasetSchema]:
    all_events = sorted(
        [
            EventSample(row[0], row[1])
            for row in all_data.select("Year", "EventId").unique().iter_rows()
        ],
        key=lambda es: (es.year, es.event_id),
    )
    train_split, val_split, test_split = _get_splits(
        all_events, seed=seed, train_weight=0.7, val_weight=0.15
    )
    _validate_splits(train_split, val_split, test_split)

    train_ds, schema = RacePredictionDataset.from_dataframe(
        _filter_events(all_data, train_split),
        feature_registry,
        target_sessions,
        feature_dropout=training_feature_dropout,
    )
    val_ds, _ = RacePredictionDataset.from_dataframe(
        _filter_events(all_data, val_split), feature_registry, target_sessions, schema
    )
    test_ds, _ = RacePredictionDataset.from_dataframe(
        _filter_events(all_data, test_split), feature_registry, target_sessions, schema
    )

    return (
        _train_loader(train_ds, batch_size, seed),
        F1DataLoader(val_ds, batch_size=1),
        F1DataLoader(test_ds, batch_size=1),
        schema,
    )
