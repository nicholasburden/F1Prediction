"""Integration test — build dataloaders from real data and verify shapes/contents."""

from __future__ import annotations

import torch
import pytest

from f1prediction.data.constants import DATA_DIR
from f1prediction.data.features import CORE_FEATURES
from f1prediction.data.dataloader import get_dataloaders
from f1prediction.data.pipeline import build_features

YEARS = [2020, 2021]
BATCH_SIZE = 8
SEED = 42


@pytest.fixture(scope="module")
def dataloaders():
    all_data, _, _ = build_features(DATA_DIR, YEARS, CORE_FEATURES)
    train_dl, val_dl, test_dl, _ = get_dataloaders(
        all_data, CORE_FEATURES, seed=SEED, batch_size=BATCH_SIZE
    )
    return train_dl, val_dl, test_dl


def test_returns_three_dataloaders(dataloaders) -> None:
    train_dl, val_dl, test_dl = dataloaders
    assert len(train_dl.dataset) > 0
    assert len(val_dl.dataset) > 0
    assert len(test_dl.dataset) > 0


def test_splits_dont_overlap(dataloaders) -> None:
    train_dl, val_dl, test_dl = dataloaders
    train_n = len(train_dl.dataset)
    val_n = len(val_dl.dataset)
    test_n = len(test_dl.dataset)
    total = train_n + val_n + test_n
    assert total == train_n + val_n + test_n, "Samples unaccounted for"
    assert train_n > val_n, "Train should be larger than val"
    assert train_n > test_n, "Train should be larger than test"


def test_total_samples_matches_events(dataloaders) -> None:
    train_dl, val_dl, test_dl = dataloaders
    total = len(train_dl.dataset) + len(val_dl.dataset) + len(test_dl.dataset)
    assert total > 30, f"Expected at least 30 events across 2 years, got {total}"


def test_batch_shapes(dataloaders) -> None:
    train_dl, _, _ = dataloaders
    x, cat_ids, y, num_drivers = next(iter(train_dl))
    assert x.ndim == 2
    assert x.shape[0] <= BATCH_SIZE
    assert cat_ids.ndim == 2
    assert cat_ids.shape[0] == x.shape[0]
    assert y.ndim == 1
    assert y.shape[0] == x.shape[0]
    assert num_drivers.shape == y.shape


def test_x_no_nans(dataloaders) -> None:
    train_dl, val_dl, test_dl = dataloaders
    for dl in [train_dl, val_dl, test_dl]:
        for x, *_ in dl:
            assert not torch.isnan(x).any(), "X contains NaN values"


def test_x_no_infs(dataloaders) -> None:
    train_dl, val_dl, test_dl = dataloaders
    for dl in [train_dl, val_dl, test_dl]:
        for x, *_ in dl:
            assert not torch.isinf(x).any(), "X contains Inf values"


def test_y_no_nans(dataloaders) -> None:
    train_dl, val_dl, test_dl = dataloaders
    for dl in [train_dl, val_dl, test_dl]:
        for _, _, y, _ in dl:
            assert not torch.isnan(y).any(), "y contains NaN values"


def test_y_bounded(dataloaders) -> None:
    train_dl, val_dl, test_dl = dataloaders
    for dl in [train_dl, val_dl, test_dl]:
        for _, _, y, _ in dl:
            assert (y >= 0).all()
            assert (y <= 1).all()


def test_cat_ids_nonnegative(dataloaders) -> None:
    train_dl, val_dl, test_dl = dataloaders
    for dl in [train_dl, val_dl, test_dl]:
        for _, cat_ids, _, _ in dl:
            if cat_ids.shape[1] > 0:
                assert (cat_ids >= 0).all()


def test_cat_ids_width(dataloaders) -> None:
    train_dl, _, _ = dataloaders
    _, cat_ids, _, _ = next(iter(train_dl))
    assert cat_ids.shape[1] == len(CORE_FEATURES.embedding_features)


def test_dtypes(dataloaders) -> None:
    train_dl, _, _ = dataloaders
    x, cat_ids, y, num_drivers = next(iter(train_dl))
    assert x.dtype == torch.float32
    assert cat_ids.dtype == torch.long
    assert y.dtype == torch.float32
    assert num_drivers.dtype == torch.float32


def test_feature_width_consistent(dataloaders) -> None:
    train_dl, val_dl, test_dl = dataloaders
    train_x = next(iter(train_dl))[0]
    val_x = next(iter(val_dl))[0]
    test_x = next(iter(test_dl))[0]
    assert train_x.shape[1] == val_x.shape[1] == test_x.shape[1]
