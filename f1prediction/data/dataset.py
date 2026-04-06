from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.utils.data import Dataset

from f1prediction.data.constants import Session
from f1prediction.data.registry import FeatureRegistry

import polars as pl

logger = logging.getLogger(__name__)

KEY_COLS = ["Driver", "Year", "EventId"]
SESSION_ORDER = [s.value for s in Session]
KNOWN_CLASH_COLS = {"team_id"}


@dataclass
class NormStats:
    """Per-feature mean and std, computed from the training set."""

    mean: Tensor  # (D,)
    std: Tensor  # (D,)

    @staticmethod
    def from_polars(
        df: pl.DataFrame, all_cols: list[str], normalise_cols: list[str]
    ) -> NormStats:
        normalise_set = set(normalise_cols)
        assert set(normalise_cols).issubset(normalise_set)
        mean = torch.zeros(len(all_cols), dtype=torch.float32)
        std = torch.ones(len(all_cols), dtype=torch.float32)
        norm_indices = [i for i, c in enumerate(all_cols) if c in normalise_set]
        norm_sub = df.select([all_cols[i] for i in norm_indices])
        mean[norm_indices] = torch.tensor(norm_sub.mean().row(0), dtype=torch.float32)
        raw_std = torch.tensor(norm_sub.std().row(0), dtype=torch.float32)
        raw_std[raw_std < 1e-8] = 1.0
        std[norm_indices] = raw_std
        return NormStats(mean=mean, std=std)

    def apply(self, X: Tensor) -> Tensor:
        return (X - self.mean) / self.std


def _base_name(col: str, sessions: list[str]) -> str:
    for s in sessions:
        if col.endswith(f"_{s}"):
            return col[: -(len(s) + 1)]
    return col


def _attach_targets(df: pl.DataFrame, target_sessions: list[str]) -> pl.DataFrame:
    targets = (
        df.filter(pl.col("SessionId").is_in(target_sessions))
        .select(
            "Driver",
            "Year",
            "EventId",
            pl.col("SessionId").alias("TargetSession"),
            (pl.col("Position") / pl.col("NumDrivers")).alias("Target"),
            pl.col("NumDrivers").alias("TargetNumDrivers"),
        )
        .filter(pl.col("Target").is_not_null())
    )
    joined = df.join(targets, on=KEY_COLS)

    session_idx = {s: i for i, s in enumerate(SESSION_ORDER)}
    target_ord = pl.col("TargetSession").replace(session_idx).cast(pl.Int32)
    session_ord = pl.col("SessionId").replace(session_idx).cast(pl.Int32)

    return joined.filter(session_ord < target_ord)


def _pivot_df(df: pl.DataFrame, event_wide_features: list[str]) -> pl.DataFrame:
    exclude = set(
        KEY_COLS
        + ["SessionId", "Target", "TargetSession", "TargetNumDrivers"]
        + event_wide_features
    )
    session_cols = [c for c in df.columns if c not in exclude]
    pivoted = df.pivot(
        on="SessionId",
        index=KEY_COLS + ["TargetSession", "Target", "TargetNumDrivers"],
        values=session_cols,
    )

    event_wide = df.select(KEY_COLS + event_wide_features).unique()
    n_events = df.n_unique(subset=KEY_COLS)

    if len(event_wide) != n_events:
        clash_cols = [
            c
            for c in event_wide_features
            if df.select(KEY_COLS + [c]).unique().n_unique(subset=KEY_COLS)
            < len(df.select(KEY_COLS + [c]).unique())
        ]
        unexpected = set(clash_cols) - KNOWN_CLASH_COLS
        assert not unexpected, f"Unexpected event-wide clashes in: {unexpected}"

        event_wide = df.group_by(KEY_COLS).agg(
            [pl.col(c).mode().first() for c in event_wide_features]
        )

    return pivoted.join(event_wide, on=KEY_COLS)


@dataclass
class DatasetSchema:
    """Column layout and normalisation stats derived from the training split."""

    numeric_cols: list[str]
    embedding_cols: list[str]
    norm_stats: NormStats


class RacePredictionDataset(Dataset):
    def __init__(
        self,
        X: Tensor,
        cat_ids: Tensor,
        y: Tensor,
        num_drivers: Tensor,
        numeric_cols_dropout: Tensor,
        cat_ids_dropout: Tensor,
    ) -> None:
        self.X = X
        self.cat_ids = cat_ids
        self.y = y
        self.num_drivers = num_drivers
        self.numeric_cols_dropout = numeric_cols_dropout
        self.cat_ids_dropout = cat_ids_dropout

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        X = self.X[idx]
        X = X.masked_fill(
            torch.rand_like(X, dtype=torch.float) < self.numeric_cols_dropout, 0
        )
        cat_ids = self.cat_ids[idx]
        cat_ids = cat_ids.masked_fill(
            torch.rand_like(cat_ids, dtype=torch.float) < self.cat_ids_dropout, 0
        )
        return X, cat_ids, self.y[idx], self.num_drivers[idx]

    @classmethod
    def from_dataframe(
        cls,
        df: pl.DataFrame,
        feature_registry: FeatureRegistry,
        target_sessions: list[str],
        schema: DatasetSchema | None = None,
        feature_dropout: dict[str, float] | None = None,
    ) -> tuple[RacePredictionDataset, DatasetSchema]:
        event_wide_features = feature_registry.event_wide_features
        embedding_cols = feature_registry.embedding_features

        df = _attach_targets(df, target_sessions)
        sessions = df["SessionId"].unique().to_list()
        pivoted = _pivot_df(df, event_wide_features)

        fill_map = feature_registry.null_fill_map
        pivoted = pivoted.with_columns(
            [
                pl.col(c).fill_null(fill_map.get(_base_name(c, sessions), 0.0))
                for c in pivoted.columns
                if pivoted[c].null_count() > 0
            ]
        )

        if schema is None:
            feature_cols = [
                c
                for c in pivoted.columns
                if c not in KEY_COLS + ["Target", "TargetSession", "TargetNumDrivers"]
            ]
            numeric_cols = [c for c in feature_cols if c not in embedding_cols]
            normalise_cols = [
                c for c in numeric_cols if c not in feature_registry.onehot_features
            ]
            norm_stats = NormStats.from_polars(pivoted, numeric_cols, normalise_cols)
            schema = DatasetSchema(numeric_cols, embedding_cols, norm_stats)
        else:
            for c in schema.numeric_cols:
                if c not in pivoted.columns:
                    pivoted = pivoted.with_columns(pl.lit(0.0).alias(c))
            for c in schema.embedding_cols:
                if c not in pivoted.columns:
                    pivoted = pivoted.with_columns(pl.lit(0).cast(pl.Int64).alias(c))

        X = torch.tensor(
            pivoted.select(schema.numeric_cols).to_numpy(), dtype=torch.float32
        )
        X = schema.norm_stats.apply(X)
        y = torch.tensor(pivoted["Target"].to_numpy(), dtype=torch.float32)
        num_drivers = torch.tensor(
            pivoted["TargetNumDrivers"].to_numpy(), dtype=torch.float32
        )

        if schema.embedding_cols:
            cat_ids = torch.tensor(
                pivoted.select(schema.embedding_cols).to_numpy(), dtype=torch.long
            )
        else:
            cat_ids = torch.zeros(len(y), 0, dtype=torch.long)

        numeric_cols_dropout = []
        cat_ids_dropout = []
        for col in schema.numeric_cols:
            prob = feature_dropout.get(col, 0) if feature_dropout else 0
            numeric_cols_dropout.append(prob)
        for col in schema.embedding_cols:
            prob = feature_dropout.get(col, 0) if feature_dropout else 0
            cat_ids_dropout.append(prob)

        return cls(
            X,
            cat_ids,
            y,
            num_drivers,
            Tensor(numeric_cols_dropout),
            Tensor(cat_ids_dropout),
        ), schema
