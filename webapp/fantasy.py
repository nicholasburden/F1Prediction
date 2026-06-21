"""F1 Fantasy team optimiser.

Pulls live driver/constructor prices from the official F1 Fantasy JSON feed,
maps the model's predicted finish positions to fantasy points (position-only
scoring), and enumerates valid (5 drivers + 2 constructors + 1 DRS boost)
teams to find the highest expected-points team within a budget cap.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import TypedDict

from webapp import store

log = logging.getLogger(__name__)

# The trailing integer is the game-period (race-weekend) id. Prices change
# every round, so we must resolve the latest available period rather than
# hardcode one — period 1 is the season opener and freezes prices in March.
FEED_URL_TEMPLATE = "https://fantasy.formula1.com/feeds/drivers/{period}_en.json"
FIRST_PERIOD = 1
MAX_PERIOD = 40
FEED_CACHE_TTL_SECONDS = 1800
FEED_HTTP_TIMEOUT = 10.0

TEAM_DRIVERS = 5
TEAM_CONSTRUCTORS = 2
DEFAULT_BUDGET = 100.0
DRS_MULTIPLIER = 2.0
PREMIUM_THRESHOLD = 20.0

RESTRICTION_NONE = "none"
RESTRICTION_MAX_ONE_PREMIUM_DRIVER = "max_one_premium_driver"
RESTRICTION_MAX_ONE_PREMIUM_CONSTRUCTOR = "max_one_premium_constructor"
RESTRICTIONS = (
    RESTRICTION_NONE,
    RESTRICTION_MAX_ONE_PREMIUM_DRIVER,
    RESTRICTION_MAX_ONE_PREMIUM_CONSTRUCTOR,
)
RESTRICTION_LABELS: dict[str, str] = {
    RESTRICTION_NONE: "No restrictions",
    RESTRICTION_MAX_ONE_PREMIUM_DRIVER: f"Max 1 premium driver (>${PREMIUM_THRESHOLD:g}M)",
    RESTRICTION_MAX_ONE_PREMIUM_CONSTRUCTOR: f"Max 1 premium constructor (>${PREMIUM_THRESHOLD:g}M)",
}

# F1 Fantasy position-points tables (race / qualifying / sprint). Index i ->
# points awarded for finishing position i+1; positions beyond the list score 0.
RACE_POINTS: tuple[int, ...] = (25, 18, 15, 12, 10, 8, 6, 4, 2, 1)
QUALI_POINTS: tuple[int, ...] = (10, 9, 8, 7, 6, 5, 4, 3, 2, 1)
SPRINT_POINTS: tuple[int, ...] = (8, 7, 6, 5, 4, 3, 2, 1)

# Session code -> points table. Position-only scoring across the model's
# target sessions; whatever subset is predicted gets summed.
SESSION_POINTS: dict[str, tuple[int, ...]] = {
    "R": RACE_POINTS,
    "Q": QUALI_POINTS,
    "Sprint": SPRINT_POINTS,
}


class FantasyDriver(TypedDict):
    tla: str
    full_name: str
    team: str
    price: float


class FantasyConstructor(TypedDict):
    name: str
    price: float


class FantasyFeed(TypedDict):
    fetched_at: float
    period: int
    feed_time: str | None
    drivers: list[FantasyDriver]
    constructors: list[FantasyConstructor]


class TeamPick(TypedDict):
    drivers: list[str]
    constructors: list[str]
    drs_driver: str
    total_cost: float
    total_points: float
    driver_points: dict[str, float]
    constructor_points: dict[str, float]


_cache_lock = threading.Lock()
_cache: FantasyFeed | None = None


def _parse_feed_time(feed_time: object) -> str | None:
    """Normalise the feed's ``FeedTime`` to an ISO-8601 UTC string. The feed
    now nests times by zone (``{"UTCTime": "6/21/2026 4:57:07 PM", ...}``);
    older feeds used a bare string. Returns ``None`` if neither parses."""
    raw = feed_time.get("UTCTime") if isinstance(feed_time, dict) else feed_time
    if not isinstance(raw, str):
        return None
    try:
        dt = datetime.strptime(raw.strip(), "%m/%d/%Y %I:%M:%S %p")
    except ValueError:
        return raw.strip()
    return dt.replace(tzinfo=timezone.utc).isoformat()


def _parse_feed(raw: dict[str, object], period: int) -> FantasyFeed:
    data = raw.get("Data", {})
    if not isinstance(data, dict):
        raise ValueError("fantasy feed missing 'Data' object")
    values = data.get("Value", [])
    if not isinstance(values, list):
        raise ValueError("fantasy feed 'Data.Value' is not a list")
    drivers: list[FantasyDriver] = []
    constructors: list[FantasyConstructor] = []
    for entry in values:
        if not isinstance(entry, dict):
            continue
        if entry.get("IsActive") != "1":
            continue
        pos = entry.get("PositionName")
        if pos == "DRIVER":
            tla = entry.get("DriverTLA")
            name = entry.get("FUllName")
            team = entry.get("TeamName") or ""
            price = entry.get("Value")
            if not isinstance(tla, str) or not isinstance(name, str):
                continue
            try:
                price_f = float(price)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
            drivers.append(FantasyDriver(
                tla=tla, full_name=name, team=str(team), price=price_f,
            ))
        elif pos == "CONSTRUCTOR":
            name = entry.get("FUllName")
            price = entry.get("Value")
            if not isinstance(name, str):
                continue
            try:
                price_f = float(price)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
            constructors.append(FantasyConstructor(name=name, price=price_f))
    feed_time = _parse_feed_time(data.get("FeedTime"))
    return FantasyFeed(
        fetched_at=time.time(),
        period=period,
        feed_time=feed_time,
        drivers=drivers,
        constructors=constructors,
    )


def _fetch_period(period: int) -> dict[str, object] | None:
    """Fetch the raw feed for ``period``, or ``None`` if it does not exist
    yet (future periods return HTTP 403/404)."""
    url = FEED_URL_TEMPLATE.format(period=period)
    try:
        with urllib.request.urlopen(url, timeout=FEED_HTTP_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code in (403, 404):
            return None
        raise


def _resolve_latest_feed(start: int) -> tuple[dict[str, object], int]:
    """Walk upward from ``start`` to find the newest available game period,
    returning its raw feed and period id. Prices roll over each round, so the
    highest period that still resolves is the live one."""
    raw = _fetch_period(start)
    if raw is None:
        # The hint is stale-high (e.g. a new season reset the numbering);
        # fall back to scanning from the first period.
        start = FIRST_PERIOD
        raw = _fetch_period(start)
        if raw is None:
            raise ValueError("no fantasy feed available from the F1 feed")
    period = start
    while period < MAX_PERIOD:
        nxt = _fetch_period(period + 1)
        if nxt is None:
            break
        raw, period = nxt, period + 1
    return raw, period


def fetch_fantasy_feed(*, force: bool = False) -> FantasyFeed:
    """Return the cached fantasy feed, refreshing if older than the TTL or
    ``force`` is set. Network call is at most one in flight at a time."""
    global _cache
    with _cache_lock:
        if (
            not force
            and _cache is not None
            and time.time() - _cache["fetched_at"] < FEED_CACHE_TTL_SECONDS
        ):
            return _cache
        start = _cache["period"] if _cache is not None else FIRST_PERIOD
        log.info("resolving latest fantasy feed period (from %d)", start)
        raw, period = _resolve_latest_feed(start)
        _cache = _parse_feed(raw, period)
        log.info(
            "fantasy feed parsed: period %d, %d drivers, %d constructors",
            period, len(_cache["drivers"]), len(_cache["constructors"]),
        )
        return _cache


def _points_for_position(table: tuple[int, ...], position: float) -> float:
    """Linearly interpolate between the two integer positions bracketing
    ``position``; positions outside [1, len(table)] score 0 (positions below
    the last paying place earn nothing in F1 Fantasy)."""
    if position <= 0:
        return 0.0
    floor = int(position)
    ceil = floor + 1
    if floor < 1:
        return 0.0
    if floor > len(table):
        return 0.0
    lo = table[floor - 1]
    hi = table[ceil - 1] if 1 <= ceil <= len(table) else 0
    frac = position - floor
    return lo * (1.0 - frac) + hi * frac


def _predicted_positions_for_event(
    db_path: Path, year: int, event_id: int
) -> dict[str, dict[str, float]]:
    """Return {tla: {session: predicted_position}} drawn from the latest
    stored predictions for each session of the event."""
    out: dict[str, dict[str, float]] = {}
    for sess in SESSION_POINTS:
        pred = store.latest_for_race(db_path, year, event_id, sess)
        if pred is None:
            continue
        for tla, value in pred["drivers"]:
            out.setdefault(tla, {})[sess] = value
    return out


def expected_driver_points(
    db_path: Path, year: int, event_id: int
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """For each driver TLA with a stored prediction, sum expected fantasy
    points across all available sessions. Returns (totals, per_session)."""
    per_session: dict[str, dict[str, float]] = {}
    totals: dict[str, float] = {}
    positions = _predicted_positions_for_event(db_path, year, event_id)
    for tla, sess_positions in positions.items():
        breakdown: dict[str, float] = {}
        for sess, pos in sess_positions.items():
            table = SESSION_POINTS[sess]
            breakdown[sess] = _points_for_position(table, pos)
        per_session[tla] = breakdown
        totals[tla] = sum(breakdown.values())
    return totals, per_session


def expected_constructor_points(
    driver_points: dict[str, float],
    drivers_by_team: dict[str, list[str]],
) -> dict[str, float]:
    """Constructor expected points = sum of points for both of its drivers
    (position-only approximation; F1 Fantasy adds pit/overtake bonuses we
    can't predict)."""
    return {
        team: sum(driver_points.get(tla, 0.0) for tla in tlas)
        for team, tlas in drivers_by_team.items()
    }


def drivers_by_team(drivers: list[FantasyDriver]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for d in drivers:
        out.setdefault(d["team"], []).append(d["tla"])
    return out


class _DriverEntry(TypedDict):
    tla: str
    price: float
    points: float


class _ConstructorEntry(TypedDict):
    name: str
    price: float
    points: float


def optimise_team(
    feed: FantasyFeed,
    driver_points: dict[str, float],
    constructor_points: dict[str, float],
    budget: float,
    *,
    n_drivers: int = TEAM_DRIVERS,
    n_constructors: int = TEAM_CONSTRUCTORS,
    drs_multiplier: float = DRS_MULTIPLIER,
    restriction: str = RESTRICTION_NONE,
    premium_threshold: float = PREMIUM_THRESHOLD,
) -> TeamPick | None:
    """Brute-force search over (n_drivers, n_constructors) selections within
    ``budget``, picking the highest expected-points team. DRS boost is always
    optimally placed on the in-team driver with the highest base points.

    ``restriction`` caps how many >${premium_threshold}M picks may appear:
    ``RESTRICTION_MAX_ONE_PREMIUM_DRIVER`` allows at most one driver above the
    threshold; ``RESTRICTION_MAX_ONE_PREMIUM_CONSTRUCTOR`` does the same for
    constructors; ``RESTRICTION_NONE`` disables both filters.
    """
    drivers: list[_DriverEntry] = [
        _DriverEntry(
            tla=d["tla"], price=d["price"], points=driver_points.get(d["tla"], 0.0),
        )
        for d in feed["drivers"]
    ]
    constructors: list[_ConstructorEntry] = [
        _ConstructorEntry(
            name=c["name"], price=c["price"],
            points=constructor_points.get(c["name"], 0.0),
        )
        for c in feed["constructors"]
    ]

    if len(drivers) < n_drivers or len(constructors) < n_constructors:
        return None

    cap_premium_drivers = restriction == RESTRICTION_MAX_ONE_PREMIUM_DRIVER
    cap_premium_constructors = restriction == RESTRICTION_MAX_ONE_PREMIUM_CONSTRUCTOR

    # Pre-sort constructor combinations by total cost so we can skip early
    # when the cheapest constructor pair already busts the remaining budget.
    constructor_combos: list[tuple[float, float, tuple[_ConstructorEntry, ...]]] = []
    for combo in combinations(constructors, n_constructors):
        if cap_premium_constructors and sum(
            1 for c in combo if c["price"] > premium_threshold
        ) > 1:
            continue
        constructor_combos.append((
            sum(c["price"] for c in combo),
            sum(c["points"] for c in combo),
            combo,
        ))
    constructor_combos.sort(key=lambda x: x[0])

    best: TeamPick | None = None
    best_score = float("-inf")

    for d_combo in combinations(drivers, n_drivers):
        if cap_premium_drivers and sum(
            1 for d in d_combo if d["price"] > premium_threshold
        ) > 1:
            continue
        d_cost = sum(d["price"] for d in d_combo)
        if d_cost > budget:
            continue
        d_base_points = sum(d["points"] for d in d_combo)
        drs_bonus = max(d["points"] for d in d_combo) * (drs_multiplier - 1.0)
        d_points = d_base_points + drs_bonus
        remaining = budget - d_cost
        for c_cost, c_points, c_combo in constructor_combos:
            if c_cost > remaining:
                break
            total_points = d_points + c_points
            if total_points > best_score:
                best_score = total_points
                drs_driver = max(d_combo, key=lambda d: d["points"])["tla"]
                best = TeamPick(
                    drivers=[d["tla"] for d in d_combo],
                    constructors=[c["name"] for c in c_combo],
                    drs_driver=drs_driver,
                    total_cost=d_cost + c_cost,
                    total_points=total_points,
                    driver_points={d["tla"]: d["points"] for d in d_combo},
                    constructor_points={c["name"]: c["points"] for c in c_combo},
                )
    return best
