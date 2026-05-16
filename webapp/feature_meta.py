"""Map pivoted feature columns back to (base, session, category) and group
them for display."""
from __future__ import annotations

import functools
from typing import TypedDict

from f1prediction.data.features import CORE_FEATURES, LOOKBACK_FEATURES
from f1prediction.data.registry import FeatureRegistry

# All session ids that may appear as a `_<SESSION>` suffix on a pivoted
# column.
KNOWN_SESSIONS: tuple[str, ...] = (
    "FP1", "FP2", "FP3", "SQ", "SS", "Sprint", "Q", "R",
)
_SESSIONS_BY_LEN_DESC: tuple[str, ...] = tuple(
    sorted(KNOWN_SESSIONS, key=len, reverse=True)
)

_FEATURE_SETS: dict[str, FeatureRegistry] = {
    "core": CORE_FEATURES,
    "lookback": LOOKBACK_FEATURES,
}

CATEGORY_LABEL: dict[str, str] = {
    "regulations": "Regulations",
    "identity": "Identity",
    "qualifying": "Qualifying",
    "pace": "Pace",
    "speed": "Speed",
    "weather": "Weather",
    "form": "Form",
    "results": "Session results",
    "other": "Other",
}
CATEGORY_ORDER: tuple[str, ...] = (
    "weather", "qualifying", "results", "pace", "speed", "form",
    "regulations", "identity", "other",
)

# Columns that aren't declared as FeatureSpecs but flow through the dataset
# pipeline (carried from the raw results parquet) and end up in
# numeric_cols. Categorised here so they don't fall through to "Other".
_IMPLICIT_BASE_CATEGORY: dict[str, str] = {
    "Position": "results",
    "NumDrivers": "results",
    "grid_position": "qualifying",  # already a spec, kept for safety
}


class StatBlock(TypedDict):
    min: float | None
    max: float | None
    mean: float | None
    n_null: int
    n_total: int


class SessionStat(TypedDict):
    session: str | None
    stats: StatBlock


class BaseFeature(TypedDict):
    base: str
    sessions: list[SessionStat]


class CategoryView(TypedDict):
    name: str
    label: str
    n_base: int
    n_columns: int
    bases: list[BaseFeature]


def split_column(col: str) -> tuple[str, str | None]:
    """Split ``base_SESSION`` into (base, session). Event-wide columns return
    (col, None). Suffixes are matched longest-first so 'min_lap_time_SQ' picks
    SQ rather than Q."""
    for s in _SESSIONS_BY_LEN_DESC:
        if col.endswith(f"_{s}"):
            return col[: -(len(s) + 1)], s
    return col, None


@functools.cache
def _registry_for(feature_set_names: tuple[str, ...]) -> FeatureRegistry:
    if not feature_set_names:
        raise ValueError("feature_sets is empty")
    head = _FEATURE_SETS[feature_set_names[0]]
    for name in feature_set_names[1:]:
        head = head + _FEATURE_SETS[name]
    return head


@functools.cache
def _base_to_category(feature_set_names: tuple[str, ...]) -> dict[str, str]:
    registry = _registry_for(feature_set_names)
    out: dict[str, str] = dict(_IMPLICIT_BASE_CATEGORY)
    for s in tuple(registry._specs) + tuple(registry._global_specs):  # type: ignore[attr-defined]
        out[s.name] = s.category
    return out


def categorise(
    features_by_driver: dict[str, dict[str, float | int | None]],
    feature_set_names: list[str],
) -> list[CategoryView]:
    """Aggregate per-driver feature values into a per-category, per-base,
    per-session view with min/mean/max/null-count stats across drivers."""
    if not features_by_driver:
        return []
    cat_lookup = _base_to_category(tuple(feature_set_names))
    sample = next(iter(features_by_driver.values()))
    columns = list(sample.keys())

    # category -> base -> list of (session, [values across drivers])
    grouped: dict[str, dict[str, list[tuple[str | None, list[float | int | None]]]]] = {}
    for col in columns:
        base, sess = split_column(col)
        category = cat_lookup.get(base, "other")
        values: list[float | int | None] = [
            features_by_driver[d].get(col) for d in features_by_driver
        ]
        grouped.setdefault(category, {}).setdefault(base, []).append((sess, values))

    out: list[CategoryView] = []
    for cat in CATEGORY_ORDER:
        if cat not in grouped:
            continue
        bases: list[BaseFeature] = []
        for base in sorted(grouped[cat]):
            sessions: list[SessionStat] = []
            for sess, vals in sorted(
                grouped[cat][base], key=lambda kv: kv[0] or ""
            ):
                non_null = [v for v in vals if v is not None]
                stats: StatBlock = {
                    "min": min(non_null) if non_null else None,
                    "max": max(non_null) if non_null else None,
                    "mean": (sum(non_null) / len(non_null)) if non_null else None,
                    "n_null": len(vals) - len(non_null),
                    "n_total": len(vals),
                }
                sessions.append({"session": sess, "stats": stats})
            bases.append({"base": base, "sessions": sessions})
        n_columns = sum(len(b["sessions"]) for b in bases)
        out.append({
            "name": cat,
            "label": CATEGORY_LABEL.get(cat, cat.title()),
            "n_base": len(bases),
            "n_columns": n_columns,
            "bases": bases,
        })
    return out
