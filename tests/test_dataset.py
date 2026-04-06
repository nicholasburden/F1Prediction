"""Tensor-level tests — assert exact values from a fixed-seed dataloader run."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import pytest

from f1prediction.data.constants import DATA_DIR
from f1prediction.data.features import ALL_FEATURES
from f1prediction.data.dataloader import get_dataloaders
from f1prediction.data.pipeline import build_features

SNAPSHOT_PATH = Path(__file__).parent / "snapshots" / "dataloader_seed42.pt"
SEED = 42
YEARS = [2020]
BATCH_SIZE = 4


@pytest.fixture(scope="module")
def dataloaders():
    np.random.seed(SEED)
    all_data, _ = build_features(DATA_DIR, YEARS, ALL_FEATURES)
    return get_dataloaders(all_data, ALL_FEATURES, batch_size=BATCH_SIZE)


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
    x, _, _ = first_batch
    torch.testing.assert_close(x, expected["x"])


def test_cat_ids_values(first_batch, expected) -> None:
    _, cat, _ = first_batch
    assert torch.equal(cat, expected["cat"])


def test_y_values(first_batch, expected) -> None:
    _, _, y = first_batch
    torch.testing.assert_close(y, expected["y"])


def test_split_sizes(dataloaders, expected) -> None:
    train_dl, val_dl, test_dl = dataloaders
    assert len(train_dl.dataset) == expected["train_len"]
    assert len(val_dl.dataset) == expected["val_len"]
    assert len(test_dl.dataset) == expected["test_len"]


def test_generate_snapshot(dataloaders, request: pytest.FixtureRequest) -> None:
    if not request.config.getoption("--snapshot", default=False):
        pytest.skip("Pass --snapshot to regenerate")
    train_dl, val_dl, test_dl = dataloaders
    x, cat, y = next(iter(train_dl))
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
