#!/usr/bin/env python
"""Prepare shared data artifacts (normalization stats, vocabularies, event split).

Artifacts are saved to a directory keyed by years, so different year
combinations each get their own cached stats.  Multiple training runs
can then share these without recomputing.

Usage:
    uv run python scripts/prepare_data.py --years 2020 2021 2022 2023 2024
    uv run python scripts/prepare_data.py --years 2024 --seed 123
"""

import argparse
import json
import logging
from pathlib import Path

import torch

from f1prediction.data.dataset import F1RaceDataset
from f1prediction.data.history import build_history_table
from f1prediction.data.normalization import compute_stats
from f1prediction.data.registry import REGISTRY
from f1prediction.training.splits import (
    build_vocabularies,
    generate_samples,
    get_event_order,
    split_events,
)

# Import features to trigger registration
import f1prediction.data.features  # noqa: F401

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare shared data artifacts for training",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--years", type=int, nargs="+", required=True)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--max-drivers", type=int, default=22)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("prepared"),
        help="Base directory for prepared data (a year-keyed subdirectory is created)",
    )
    parser.add_argument(
        "--norm-samples", type=int, default=500,
        help="Max training samples to use for normalization stats (0=all)",
    )
    return parser.parse_args()


def years_key(years: list[int]) -> str:
    """Create a directory-safe key from sorted years."""
    return "_".join(str(y) for y in sorted(years))


def main() -> None:
    args = parse_args()
    years = sorted(args.years)

    out_dir = args.output_dir / years_key(years)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Discover events and split
    logger.info("Discovering events for years %s...", years)
    event_order = get_event_order(args.data_dir, years)
    logger.info("Found %d events", len(event_order))

    train_events, val_events, test_events = split_events(
        event_order, args.train_frac, args.val_frac, seed=args.seed,
    )
    logger.info(
        "Event split: %d train, %d val, %d test",
        len(train_events), len(val_events), len(test_events),
    )

    # 2. Generate training samples
    train_samples = generate_samples(args.data_dir, train_events)
    val_count = len(generate_samples(args.data_dir, val_events))
    test_count = len(generate_samples(args.data_dir, test_events))
    logger.info(
        "Samples: %d train, %d val, %d test",
        len(train_samples), val_count, test_count,
    )

    # 3. Build vocabularies
    driver_vocab, team_vocab = build_vocabularies(args.data_dir, years)
    logger.info(
        "Driver vocab: %d (+UNK), Team vocab: %d (+UNK)",
        len(driver_vocab), len(team_vocab),
    )

    # 4. Build history table and compute normalization stats
    logger.info("Building history table from %d events...", len(event_order))
    history_table, team_history_table = build_history_table(args.data_dir, event_order)
    logger.info("History table: %d driver, %d team entries", len(history_table), len(team_history_table))

    norm_subset = train_samples
    if args.norm_samples > 0 and len(train_samples) > args.norm_samples:
        import random as _rng
        _rng.seed(args.seed)
        norm_subset = _rng.sample(train_samples, args.norm_samples)
    logger.info("Computing normalization stats from %d/%d training samples...",
                len(norm_subset), len(train_samples))
    raw_train = F1RaceDataset(
        args.data_dir,
        norm_subset,
        REGISTRY,
        max_drivers=args.max_drivers,
        driver_vocab=driver_vocab,
        team_vocab=team_vocab,
        history_table=history_table,
        team_history_table=team_history_table,
    )
    train_features = raw_train.get_all_features()
    feature_names = REGISTRY.feature_names
    norm_stats = compute_stats(train_features, feature_names)

    # 5. Save artifacts
    norm_stats.save(out_dir / "norm_stats.json")
    logger.info("Saved norm_stats.json")

    (out_dir / "vocabs.json").write_text(json.dumps({
        "driver_vocab": driver_vocab,
        "team_vocab": team_vocab,
        "max_drivers": args.max_drivers,
    }, indent=2))
    logger.info("Saved vocabs.json")

    (out_dir / "split.json").write_text(json.dumps({
        "train_events": train_events,
        "val_events": val_events,
        "test_events": test_events,
    }, indent=2))
    logger.info("Saved split.json")

    (out_dir / "meta.json").write_text(json.dumps({
        "years": years,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "max_drivers": args.max_drivers,
        "seed": args.seed,
        "feature_dim": REGISTRY.total_dim,
        "feature_names": feature_names,
        "n_train_samples": len(train_samples),
        "n_val_samples": val_count,
        "n_test_samples": test_count,
    }, indent=2))
    logger.info("Saved meta.json")

    logger.info("Done — artifacts in %s", out_dir)


if __name__ == "__main__":
    main()
