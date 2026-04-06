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
    events: list[EventSample], train_weight: float, val_weight: float
) -> tuple[list[EventSample], list[EventSample], list[EventSample]]:
    assert train_weight > 0
    assert val_weight > 0
    assert train_weight + val_weight < 1

    train_end_idx = int(round(train_weight * len(events)))
    val_end_idx = int(round((train_weight + val_weight) * len(events)))

    assert train_end_idx > 0
    assert train_end_idx < val_end_idx
    assert val_end_idx < (len(events) - 1)

    np.random.shuffle(events)  # type: ignore[arg-type]
    return (
        events[:train_end_idx],
        events[train_end_idx:val_end_idx],
        events[val_end_idx:],
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


def get_dataloaders(
    all_data: pl.DataFrame,
    feature_registry: FeatureRegistry,
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
        all_events, train_weight=0.7, val_weight=0.15
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
        F1DataLoader(train_ds, batch_size=batch_size),
        F1DataLoader(val_ds, batch_size=1),
        F1DataLoader(test_ds, batch_size=1),
        schema,
    )
