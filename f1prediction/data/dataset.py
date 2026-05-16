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


def _base_name(col: str) -> str:
    """Strip a trailing ``_<session>`` suffix from a pivoted feature column,
    using the canonical ``SESSION_ORDER`` rather than whichever sessions
    happen to appear in the current data. This matters at inference for
    events that lack some sessions (e.g. SQ on a non-sprint event) — without
    a full session list, ``min_lap_time_wet_SQ`` would not get stripped and
    its registry ``fill_null`` (999.0) would be missed, leaving the column
    silently zero-filled and the model fed wildly OOD inputs."""
    for s in SESSION_ORDER:
        if col.endswith(f"_{s}"):
            return col[: -(len(s) + 1)]
    return col


TARGET_FEATURE_PREFIX = "target_"


def _attach_targets(
    df: pl.DataFrame,
    target_sessions: list[str],
    masked_cols: list[str],
    target_feature_cols: list[str] | None = None,
) -> pl.DataFrame:
    """Attach a (Target, TargetSession, _cutoff) tuple to each preceding row.

    For each (Driver, Year, EventId, TargetSession), emit ``target_ord + 1``
    variants — one per cutoff ``k`` in ``[0, target_ord]``. Within a variant
    of cutoff ``k``, sessions with ``_session_ord >= k`` are "future" relative
    to the cutoff: their feature columns named in ``masked_cols`` are nulled
    out (so they fill to their sentinels post-pivot), while columns not in
    ``masked_cols`` (event-wide, known-at-inference like weather, and
    lookback features) keep their real values. The PRE row (``_session_ord =
    -1``) is always visible.

    Columns named in ``target_feature_cols`` are pulled from the target
    session's row and carried into every variant as ``target_<col>``, so the
    model gets target-session weather (which we forecast at inference) as a
    single set of columns rather than per-session-suffixed bloat.
    """
    target_alias = [
        pl.col(c).alias(f"{TARGET_FEATURE_PREFIX}{c}")
        for c in (target_feature_cols or [])
    ]
    targets = (
        df.filter(pl.col("SessionId").is_in(target_sessions))
        .select(
            "Driver",
            "Year",
            "EventId",
            pl.col("SessionId").alias("TargetSession"),
            (pl.col("Position") / pl.col("NumDrivers")).alias("Target"),
            pl.col("NumDrivers").alias("TargetNumDrivers"),
            pl.col("_session_ord").alias("_target_session_ord"),
            *target_alias,
        )
        .filter(pl.col("Target").is_not_null())
        .with_columns(
            pl.int_ranges(0, pl.col("_target_session_ord") + 1).alias("_cutoff")
        )
        .explode("_cutoff")
    )
    joined = df.join(targets, on=KEY_COLS).filter(
        pl.col("_session_ord") < pl.col("_target_session_ord")
    )
    is_future = pl.col("_session_ord") >= pl.col("_cutoff")
    existing_masked = [c for c in masked_cols if c in joined.columns]
    return (
        joined.with_columns(
            [
                pl.when(is_future).then(None).otherwise(pl.col(c)).alias(c)
                for c in existing_masked
            ]
        )
        .drop("_session_ord", "_has_sprint", "_target_session_ord")
    )


def _pivot_df(df: pl.DataFrame, event_wide_features: list[str]) -> pl.DataFrame:
    # ``Position`` is carried in the long frame only so ``_attach_targets``
    # can derive ``Target``; it is not a registered feature. Pivoting it
    # leaks the qualifying / sprint result into the model under any cutoff
    # variant and produces a fill-rule mismatch with the inference path,
    # so drop it before pivoting.
    # ``target_*`` columns hold the target session's known-at-inference
    # values; they're constant per (KEY, TargetSession, _cutoff) so we put
    # them in the pivot index to carry them through unchanged rather than
    # smearing them across per-session pivoted columns.
    target_cols = [c for c in df.columns if c.startswith(TARGET_FEATURE_PREFIX)]
    exclude = set(
        KEY_COLS
        + ["SessionId", "Target", "TargetSession", "TargetNumDrivers",
           "_cutoff", "Position"]
        + event_wide_features
        + target_cols
    )
    session_cols = [c for c in df.columns if c not in exclude]
    pivoted = df.pivot(
        on="SessionId",
        index=KEY_COLS + ["TargetSession", "Target", "TargetNumDrivers", "_cutoff"]
              + target_cols,
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
        block_dropout_masks: Tensor,
        block_dropout_probs: Tensor,
    ) -> None:
        self.X = X
        self.cat_ids = cat_ids
        self.y = y
        self.num_drivers = num_drivers
        self.numeric_cols_dropout = numeric_cols_dropout
        self.cat_ids_dropout = cat_ids_dropout
        # (K, D) bool — which numeric columns belong to each block group
        self.block_dropout_masks = block_dropout_masks
        # (K,) float — bernoulli probability of dropping each group
        self.block_dropout_probs = block_dropout_probs

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        X = self.X[idx]
        X = X.masked_fill(
            torch.rand_like(X, dtype=torch.float) < self.numeric_cols_dropout, 0
        )
        if self.block_dropout_probs.numel() > 0:
            triggered = (
                torch.rand_like(self.block_dropout_probs) < self.block_dropout_probs
            )
            block_mask = (
                self.block_dropout_masks & triggered.unsqueeze(1)
            ).any(dim=0)
            X = X.masked_fill(block_mask, 0)
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
        block_dropout: dict[str, float] | None = None,
    ) -> tuple[RacePredictionDataset, DatasetSchema]:
        event_wide_features = feature_registry.event_wide_features
        embedding_cols = feature_registry.embedding_features
        masked_cols = feature_registry.session_specific_features
        # Per-session features flagged ``known_at_inference`` (i.e. the
        # target session's value is forecastable/announceable, e.g.
        # weather) are pulled from the target row and carried through as
        # ``target_<col>`` columns.
        target_feature_cols = feature_registry.target_session_features

        df = _attach_targets(df, target_sessions, masked_cols, target_feature_cols)
        pivoted = _pivot_df(df, event_wide_features)

        # Polars pivot/unique row order is hash-based and varies across
        # processes — sort here so the dataset is reproducible.
        pivoted = pivoted.sort(KEY_COLS + ["TargetSession", "_cutoff"]).drop("_cutoff")

        fill_map = feature_registry.null_fill_map
        pivoted = pivoted.with_columns(
            [
                pl.col(c).fill_null(fill_map.get(_base_name(c), 0.0))
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
                    fill = fill_map.get(_base_name(c), 0.0)
                    pivoted = pivoted.with_columns(pl.lit(fill).alias(c))
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

        block_groups = feature_registry.feature_groups
        active_blocks = [
            (name, prob) for name, prob in (block_dropout or {}).items()
            if prob > 0 and name in block_groups
        ]
        if active_blocks:
            block_masks = torch.zeros(
                (len(active_blocks), len(schema.numeric_cols)), dtype=torch.bool
            )
            for k, (name, _) in enumerate(active_blocks):
                base_names = set(block_groups[name])
                for j, col in enumerate(schema.numeric_cols):
                    base = _base_name(col)
                    if col.startswith(TARGET_FEATURE_PREFIX):
                        base = col[len(TARGET_FEATURE_PREFIX):]
                    if base in base_names:
                        block_masks[k, j] = True
            block_probs = torch.tensor(
                [prob for _, prob in active_blocks], dtype=torch.float32
            )
        else:
            block_masks = torch.zeros((0, len(schema.numeric_cols)), dtype=torch.bool)
            block_probs = torch.zeros(0, dtype=torch.float32)

        return cls(
            X,
            cat_ids,
            y,
            num_drivers,
            Tensor(numeric_cols_dropout),
            Tensor(cat_ids_dropout),
            block_masks,
            block_probs,
        ), schema
