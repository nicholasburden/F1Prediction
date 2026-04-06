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


@dataclass(frozen=True)
class GlobalFeatureSpec:
    name: str
    category: str
    fn: Callable[[pl.LazyFrame], pl.LazyFrame]


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
    def null_fill_map(self) -> dict[str, float]:
        return {s.name: s.fill_null for s in self._specs}

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
        for spec in self._global_specs:
            df = spec.fn(df)
        return df
