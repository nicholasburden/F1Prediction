from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable, Literal

import polars as pl

from f1prediction.data.constants import DataTable

Encoding = Literal["numeric", "onehot", "embedding"]

SESSION_KEYS: dict[DataTable, tuple[str, ...]] = {
    "laps": ("Driver", "Year", "EventId", "SessionId"),
    "results": ("Driver", "Year", "EventId", "SessionId"),
    "weather": ("Year", "EventId", "SessionId"),
}
SAMPLE_KEYS: tuple[str, ...] = ("Driver", "Year", "EventId", "SessionId")


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    category: str
    expr: pl.Expr
    table: DataTable
    encoding: Encoding = "numeric"
    event_wide: bool = False
    fill_null: float = 0.0
    # When True, the value is treated as available at inference for any future
    # session (e.g. weather, where the inference path fills future sessions
    # from a forecast). Such features are NOT masked out by the partial-
    # weekend cutoff augmentation in the dataset.
    known_at_inference: bool = False
    # Specs sharing a non-None ``dropout_group`` are dropped together as a
    # block during training (one bernoulli per group per sample), matching
    # inference paths where a whole class of features may be unavailable
    # (e.g. weather forecasts too far ahead to be reliable).
    dropout_group: str | None = None


@dataclass(frozen=True)
class GlobalFeatureSpec:
    name: str
    category: str
    fn: Callable[[pl.LazyFrame], pl.LazyFrame]
    fill_null: float = 0.0
    # See FeatureSpec.known_at_inference. Lookback-style globals that only
    # depend on prior events should set this — the cutoff augmentation then
    # keeps them visible across all partial-weekend variants.
    known_at_inference: bool = False


class FeatureRegistry:
    __slots__ = ("_specs", "_global_specs")

    def __init__(
        self,
        specs: Sequence[FeatureSpec] = (),
        global_specs: Sequence[GlobalFeatureSpec] = (),
    ) -> None:
        self._specs = tuple(specs)
        self._global_specs = tuple(global_specs)

    def __len__(self) -> int:
        return len(self._specs) + len(self._global_specs)

    def __repr__(self) -> str:
        return (
            f"FeatureRegistry({len(self._specs)} table specs, "
            f"{len(self._global_specs)} global specs)"
        )

    def __add__(self, other: FeatureRegistry) -> FeatureRegistry:
        own_names = {s.name for s in self._specs} | {s.name for s in self._global_specs}
        other_names = {s.name for s in other._specs} | {
            s.name for s in other._global_specs
        }
        dupes = own_names & other_names
        if dupes:
            raise ValueError(f"Duplicate feature names: {sorted(dupes)}")
        return FeatureRegistry(
            specs=self._specs + other._specs,
            global_specs=self._global_specs + other._global_specs,
        )

    def filter(self, categories: set[str]) -> FeatureRegistry:
        return FeatureRegistry(
            specs=tuple(s for s in self._specs if s.category in categories),
            global_specs=tuple(
                s for s in self._global_specs if s.category in categories
            ),
        )

    def exclude(self, names: set[str]) -> FeatureRegistry:
        return FeatureRegistry(
            specs=tuple(s for s in self._specs if s.name not in names),
            global_specs=tuple(s for s in self._global_specs if s.name not in names),
        )

    @property
    def all_features(self) -> list[str]:
        return [s.name for s in self._specs]

    @property
    def onehot_features(self) -> list[str]:
        return [s.name for s in self._specs if s.encoding == "onehot"]

    @property
    def embedding_features(self) -> list[str]:
        return [s.name for s in self._specs if s.encoding == "embedding"]

    @property
    def features_to_normalise(self) -> list[str]:
        return sorted(
            set(self.all_features)
            - set(self.onehot_features)
            - set(self.embedding_features)
        )

    @property
    def event_wide_features(self) -> list[str]:
        return [s.name for s in self._specs if s.event_wide]

    @property
    def known_at_inference_features(self) -> list[str]:
        return [s.name for s in self._specs if s.known_at_inference] + [
            s.name for s in self._global_specs if s.known_at_inference
        ]

    @property
    def target_session_features(self) -> list[str]:
        """Per-session FeatureSpec entries whose target-session value is
        legitimately known at inference (e.g. weather via forecast). The
        dataset carries their target-session value through as
        ``target_<col>``. Setting ``known_at_inference=True`` on a per-
        session feature whose value is NOT actually known at inference
        is a lookahead bug."""
        return [
            s.name for s in self._specs
            if s.known_at_inference and not s.event_wide
        ]

    @property
    def feature_groups(self) -> dict[str, list[str]]:
        """Map ``dropout_group`` → list of spec names in that group."""
        groups: dict[str, list[str]] = {}
        for s in self._specs:
            if s.dropout_group is not None:
                groups.setdefault(s.dropout_group, []).append(s.name)
        return groups

    @property
    def session_specific_features(self) -> list[str]:
        """Features that vary by session and are not 'known at inference' —
        these get masked out for future sessions in the partial-weekend
        cutoff augmentation."""
        return [
            s.name for s in self._specs
            if not s.event_wide and not s.known_at_inference
        ] + [
            s.name for s in self._global_specs if not s.known_at_inference
        ]

    @property
    def null_fill_map(self) -> dict[str, float]:
        return {
            **{s.name: s.fill_null for s in self._specs},
            **{s.name: s.fill_null for s in self._global_specs},
        }

    def apply_group(
        self,
        table: DataTable,
        data: dict[DataTable, pl.LazyFrame],
    ) -> pl.LazyFrame:
        matching = [s for s in self._specs if s.table == table]
        if not matching:
            raise ValueError(f"No features registered for table={table!r}")
        return (
            data[table]
            .group_by(SESSION_KEYS[table])
            .agg([s.expr.alias(s.name) for s in matching])
        )

    def apply_global(self, df: pl.LazyFrame) -> pl.LazyFrame:
        # Materialise between specs: each rolling/lookback feature stacks a
        # join + sort + with_row_index, and chaining several lazily blows up
        # the polars physical-plan builder (recursive LP traversal hangs
        # indefinitely and consumes tens of GB before any execution starts).
        for spec in self._global_specs:
            df = spec.fn(df).collect().lazy()
        return df
