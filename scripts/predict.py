#!/usr/bin/env python
"""Predict race finishing positions for a given event."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import polars as pl
import torch

from f1prediction.data.dataset import F1RaceDataset, find_event_drivers, load_event_data
from f1prediction.data.extraction import extract_samples
from f1prediction.data.features import WEEKEND_SESSION_ORDER
from f1prediction.data.history import build_history_table
from f1prediction.data.normalization import NormalizationStats
from f1prediction.data.registry import REGISTRY
from f1prediction.config import ModelConfig
from f1prediction.models import build_model
from f1prediction.training.splits import PredictionSample, get_event_order

# Import features to trigger registration
import f1prediction.data.features  # noqa: F401

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_TREE_BACKENDS = {"xgboost", "lightgbm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict F1 race finishing positions"
    )
    parser.add_argument(
        "--run-dir", type=Path, default=Path("runs"),
        help="Directory containing model artifacts",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument(
        "--event", type=str, required=True,
        help="Event slug (directory name under data/<year>/)",
    )
    parser.add_argument(
        "--sessions", type=str, nargs="*", default=None,
        help="Sessions available at prediction time (e.g. FP1 FP2 Q). "
             "Default: use all sessions present on disk.",
    )
    parser.add_argument(
        "--target", type=str, default="R", choices=["Q", "Sprint", "R"],
        help="Session to predict results for (default: R)",
    )
    return parser.parse_args()


def get_actual_results(event_data: dict, target_session: str = "R") -> dict[str, float | None]:
    """Get actual positions for the target session, for comparison."""
    results = event_data.get(target_session, {}).get("results")
    if results is None:
        return {}
    actual = {}
    for row in results.iter_rows(named=True):
        abbr = row.get("Abbreviation")
        pos = row.get("Position")
        if abbr is not None:
            actual[abbr] = pos
    return actual


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir

    # Load artifacts
    vocab_path = run_dir / "vocabs.json"
    if not vocab_path.exists():
        logger.error("vocabs.json not found in %s. Re-run training to generate it.", run_dir)
        return

    vocabs = json.loads(vocab_path.read_text())
    driver_vocab = vocabs["driver_vocab"]
    team_vocab = vocabs["team_vocab"]
    max_drivers = vocabs["max_drivers"]

    norm_stats = NormalizationStats.load(run_dir / "norm_stats.json")

    model_cfg_path = run_dir / "model_config.json"
    if not model_cfg_path.exists():
        logger.error("model_config.json not found in %s. Re-run training.", run_dir)
        return
    mc = json.loads(model_cfg_path.read_text())
    backend = mc.get("backend", "torch")

    # Validate event exists
    event_dir = args.data_dir / str(args.year) / args.event
    if not event_dir.exists():
        logger.error("Event directory not found: %s", event_dir)
        return

    # Discover drivers and sessions
    drivers = find_event_drivers(args.data_dir, args.year, args.event)
    logger.info("Found %d drivers", len(drivers))

    # Log session info
    event_data = load_event_data(args.data_dir, args.year, args.event)
    available = [s for s in WEEKEND_SESSION_ORDER if s in event_data]
    logger.info("Sessions on disk: %s", available)

    # Determine which sessions to use
    if args.sessions is not None:
        used = [s for s in WEEKEND_SESSION_ORDER if s in args.sessions and s in event_data]
        logger.info("Using sessions: %s", used if used else "(pre-weekend, no sessions)")
    else:
        used = available
        logger.info("Using all available sessions: %s", used)

    # Build history table from all available years
    all_years = sorted(
        int(d.name) for d in args.data_dir.iterdir()
        if d.is_dir() and d.name.isdigit()
    )
    all_events_chrono = get_event_order(args.data_dir, all_years)
    history_table, team_history_table = build_history_table(args.data_dir, all_events_chrono)
    logger.info("History table: %d driver, %d team entries from years %s", len(history_table), len(team_history_table), all_years)

    # Build samples
    visible = tuple(used)
    samples = [
        PredictionSample(args.year, args.event, d, visible, args.target)
        for d in drivers
    ]

    if backend in _TREE_BACKENDS:
        predictions = _predict_tree(run_dir, mc, samples, args, drivers, max_drivers,
                                    driver_vocab, team_vocab, history_table, team_history_table)
    else:
        predictions = _predict_torch(run_dir, mc, norm_stats, samples, args, drivers, max_drivers,
                                     driver_vocab, team_vocab, history_table, team_history_table)

    predictions.sort(key=lambda x: x[1])

    # Get actual results for comparison
    actual = get_actual_results(event_data, args.target)

    # Display
    _display_results(predictions, actual, args, used)


def _predict_tree(run_dir, mc, samples, args, drivers, max_drivers,
                  driver_vocab, team_vocab, history_table, team_history_table):
    """Run predictions using a gradient boosting model."""
    from f1prediction.models.gbm import GBMModel

    model = GBMModel.load(run_dir / "best_model.joblib")

    # Extract features without normalization
    data = extract_samples(
        args.data_dir,
        samples,
        REGISTRY,
        max_drivers=max_drivers,
        driver_vocab=driver_vocab,
        team_vocab=team_vocab,
        history_table=history_table,
        team_history_table=team_history_table,
    )

    preds = model.predict(data.features, data.cat_ids)
    return [(drivers[i], float(preds[i] * max_drivers)) for i in range(len(drivers))]


def _predict_torch(run_dir, mc, norm_stats, samples, args, drivers, max_drivers,
                   driver_vocab, team_vocab, history_table, team_history_table):
    """Run predictions using a PyTorch neural network model."""
    model_cfg = ModelConfig(
        model_type=mc["model_type"],
        hidden_dims=mc.get("hidden_dims", [128, 64]),
        dropout=mc.get("dropout", 0.1),
        driver_embed_dim=mc.get("driver_embed_dim", 8),
        team_embed_dim=mc.get("team_embed_dim", 4),
        normalize_embeddings=mc.get("normalize_embeddings", False),
    )
    model = build_model(
        mc["continuous_dim"], model_cfg,
        driver_vocab_size=mc["driver_vocab_size"],
        team_vocab_size=mc["team_vocab_size"],
    )
    model.load_state_dict(
        torch.load(run_dir / "best_model.pt", weights_only=True)
    )
    model.eval()

    ds = F1RaceDataset(
        args.data_dir,
        samples,
        REGISTRY,
        norm_stats=norm_stats,
        max_drivers=max_drivers,
        driver_vocab=driver_vocab,
        team_vocab=team_vocab,
        history_table=history_table,
        team_history_table=team_history_table,
    )

    predictions = []
    with torch.no_grad():
        for i, sample in enumerate(samples):
            features, cat_ids, _ = ds[i]
            pred = model(features.unsqueeze(0), cat_ids.unsqueeze(0)).item()
            predictions.append((sample.driver, pred * max_drivers))
    return predictions


def _display_results(predictions, actual, args, used):
    """Display prediction results."""
    print()
    print(f"  Predictions for {args.year} {args.event} ({args.target})")
    print(f"  Sessions used: {', '.join(used) if used else 'none (pre-weekend)'}")
    print()

    header = f"  {'Pos':>3}  {'Driver':<6}  {'Predicted':>9}"
    if actual:
        header += f"  {'Actual':>6}  {'Error':>5}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    total_error = 0.0
    n_compared = 0
    for rank, (driver, pred_pos) in enumerate(predictions, 1):
        line = f"  {rank:>3}  {driver:<6}  {pred_pos:>9.2f}"
        if actual:
            act = actual.get(driver)
            if act is not None:
                err = abs(rank - act)
                total_error += err
                n_compared += 1
                line += f"  {act:>6.0f}  {err:>+5.0f}"
            else:
                line += f"  {'DNF':>6}      "
        print(line)

    if n_compared > 0:
        mae = total_error / n_compared
        print()
        print(f"  MAE: {mae:.2f} positions ({n_compared} drivers)")
    print()


if __name__ == "__main__":
    main()
