#!/usr/bin/env python
"""Predictions for China 2026 using best model (SingleLayerMLP lb3).

Predicts:
  1. Sprint  — using FP1 + SQ (sessions available before sprint)
  2. Qualifying — using FP1 + SQ + Sprint (sessions available before qualifying)
  3. Race — using FP1 + SQ + Sprint + Q (all pre-race sessions)

Uses all historical data up to (but not including) China 2026.
"""

import json
import logging
from pathlib import Path

import polars as pl
import torch

from f1prediction.config import ModelConfig
from f1prediction.data.dataset import (
    find_event_drivers,
    load_event_data,
    vocab_index,
    find_driver_team,
)
from f1prediction.data.history import build_history_table
from f1prediction.data.normalization import NormalizationStats
from f1prediction.data.registry import REGISTRY
from f1prediction.models import build_model
from f1prediction.training.splits import get_event_order

import f1prediction.data.features  # noqa: F401
from f1prediction.data.features import WEEKEND_SESSION_ORDER

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

RUN_DIR = Path("runs/pca_improved_all")
DATA_DIR = Path("data")
TARGET_YEAR = 2026
TARGET_EVENT = "chinese_grand_prix"
TOP_N = 20

# Sessions available before each target, in the order they happen
PREDICTIONS = [
    ("Sprint", ["FP1", "SQ"]),
    ("Q", ["FP1", "SQ", "Sprint"]),
    ("R", ["FP1", "SQ", "Sprint", "Q"]),
]


def build_visible_event_data(full_event_data: dict, visible_sessions: list[str]) -> dict:
    """Build event_data dict with only the specified sessions visible."""
    data = {"meta": full_event_data["meta"]}
    for session in visible_sessions:
        if session in full_event_data:
            data[session] = full_event_data[session]
    return data


def main() -> None:
    # Load model artifacts
    vocabs = json.loads((RUN_DIR / "vocabs.json").read_text())
    driver_vocab = vocabs["driver_vocab"]
    team_vocab = vocabs["team_vocab"]
    max_drivers = vocabs["max_drivers"]

    norm_stats = NormalizationStats.load(RUN_DIR / "norm_stats.json")
    expected_feature_order = norm_stats.feature_names
    logger.info("Model expects %d features", len(expected_feature_order))

    mc = json.loads((RUN_DIR / "model_config.json").read_text())
    model = build_model(
        mc["continuous_dim"],
        ModelConfig(
            model_type=mc["model_type"],
            hidden_dims=mc.get("hidden_dims", [128, 64]),
            dropout=mc.get("dropout", 0.1),
            driver_embed_dim=mc.get("driver_embed_dim", 8),
            team_embed_dim=mc.get("team_embed_dim", 4),
            normalize_embeddings=mc.get("normalize_embeddings", False),
        ),
        driver_vocab_size=mc["driver_vocab_size"],
        team_vocab_size=mc["team_vocab_size"],
    )
    model.load_state_dict(torch.load(RUN_DIR / "best_model.pt", weights_only=True))
    model.eval()
    logger.info("Loaded model: %s (%d params)", model.name, sum(p.numel() for p in model.parameters()))

    # Build history from all data up to (but not including) China 2026
    all_years = sorted(int(d.name) for d in DATA_DIR.iterdir() if d.is_dir() and d.name.isdigit())
    all_events = get_event_order(DATA_DIR, all_years)
    filtered = []
    for year, slug in all_events:
        if year == TARGET_YEAR and slug == TARGET_EVENT:
            break
        filtered.append((year, slug))
    history_table, team_history_table = build_history_table(DATA_DIR, filtered, lookback=3)
    logger.info("History: %d events, %d driver / %d team entries", len(filtered), len(history_table), len(team_history_table))

    # Get drivers and full event data
    drivers = find_event_drivers(DATA_DIR, TARGET_YEAR, TARGET_EVENT)
    logger.info("Drivers: %d", len(drivers))
    event_data_full = load_event_data(DATA_DIR, TARGET_YEAR, TARGET_EVENT)

    for target, visible_sessions in PREDICTIONS:
        # Build event data with only sessions visible at prediction time
        visible = [s for s in visible_sessions if s in event_data_full]
        event_data = build_visible_event_data(event_data_full, visible)

        predictions: list[tuple[str, str, float]] = []
        with torch.no_grad():
            for driver in drivers:
                driver_history = history_table.get((TARGET_YEAR, TARGET_EVENT, driver))

                # Look up team history
                team = find_driver_team(event_data_full, driver) or "Unknown"
                team_hist = team_history_table.get((TARGET_YEAR, TARGET_EVENT, team))

                # Extract ALL features as named dict
                named_features = REGISTRY.extract_named(
                    event_data, driver,
                    max_drivers=max_drivers,
                    driver_history=driver_history,
                    team_history=team_hist,
                    target_session=target,
                )

                # Reorder to match model's expected feature order
                feature_vec = [named_features.get(name, 0.0) for name in expected_feature_order]
                features_t = torch.tensor(feature_vec, dtype=torch.float32)

                # Normalize using training stats
                normalized = norm_stats.normalize(features_t)

                # Categorical IDs
                team = find_driver_team(event_data_full, driver) or "Unknown"
                driver_idx = vocab_index(driver, driver_vocab)
                team_idx = vocab_index(team, team_vocab)
                cat_ids = torch.tensor([driver_idx, team_idx], dtype=torch.long)

                # Predict
                pred = model(normalized.unsqueeze(0), cat_ids.unsqueeze(0)).item()
                pred_pos = pred * max_drivers
                predictions.append((driver, team, pred_pos))

        predictions.sort(key=lambda x: x[2])

        # Get actual results if available
        actual_results = event_data_full.get(target, {}).get("results")
        actual = {}
        if actual_results is not None:
            for row in actual_results.iter_rows(named=True):
                abbr = row.get("Abbreviation")
                pos = row.get("Position")
                if abbr is not None and pos is not None:
                    actual[abbr] = pos

        target_label = {"Sprint": "Sprint Race", "Q": "Qualifying", "R": "Race"}[target]
        has_actual = bool(actual)

        print()
        print(f"  China 2026 — {target_label} (Top {TOP_N})")
        print(f"  Sessions used: {', '.join(visible) if visible else 'none'}")
        if not has_actual:
            print(f"  (Race has not taken place yet — prediction only)")
        print()

        header = f"  {'Pos':>3}  {'Driver':<5}  {'Team':<18}  {'Pred':>6}"
        if has_actual:
            header += f"  {'Actual':>6}  {'Err':>4}"
        print(header)
        print(f"  {'---':>3}  {'-----':<5}  {'------------------':<18}  {'------':>6}", end="")
        if has_actual:
            print(f"  {'------':>6}  {'----':>4}", end="")
        print()

        for rank, (driver, team, pred_pos) in enumerate(predictions[:TOP_N], 1):
            line = f"  {rank:>3}  {driver:<5}  {team:<18}  {pred_pos:>6.2f}"
            if has_actual:
                act = actual.get(driver)
                if act is not None:
                    err = rank - int(act)
                    line += f"  {int(act):>6}  {err:>+4d}"
                else:
                    line += f"  {'DNF':>6}      "
            print(line)

        if has_actual:
            total_err = 0.0
            n = 0
            for rank, (driver, _, _) in enumerate(predictions, 1):
                act = actual.get(driver)
                if act is not None:
                    total_err += abs(rank - act)
                    n += 1
            if n > 0:
                print(f"\n  Full-field MAE: {total_err / n:.2f} positions ({n} drivers)")

    print()


if __name__ == "__main__":
    main()
