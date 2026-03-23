#!/usr/bin/env python
"""CLI entry point for training the F1 race position prediction model.

Uses Hydra for configuration management. Examples:

    # Train XGBoost (default):
    uv run python scripts/train.py

    # Train MLP:
    uv run python scripts/train.py model=mlp

    # Override hyperparameters:
    uv run python scripts/train.py model=xgboost model.learning_rate=0.01 model.max_depth=8

    # Change data years and output directory:
    uv run python scripts/train.py data.years=[2023,2024] training.output_dir=runs/experiment1

    # LightGBM with custom estimators:
    uv run python scripts/train.py model=lightgbm model.n_estimators=1000
"""

import json
import logging
import random
from pathlib import Path

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from f1prediction.data.dataset import F1RaceDataset
from f1prediction.data.fast_extraction import fast_extract_samples, precompute_summaries
from f1prediction.data.history import build_history_table
from f1prediction.data.normalization import NormalizationStats, compute_stats_numpy
from f1prediction.data.registry import REGISTRY
from f1prediction.models import build_model
from f1prediction.config import ModelConfig
from f1prediction.training.splits import (
    build_vocabularies,
    generate_samples,
    get_event_order,
    split_events,
)
from f1prediction.training.trainer import Trainer

# Import features to trigger registration
import f1prediction.data.features  # noqa: F401

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_TREE_BACKENDS = {"xgboost", "lightgbm"}


def _build_model_config(model_cfg: DictConfig) -> ModelConfig:
    """Convert Hydra model config to ModelConfig dataclass."""
    backend = model_cfg.backend

    if backend in _TREE_BACKENDS:
        return ModelConfig(
            backend=backend,
            model_type=backend,
            n_estimators=model_cfg.n_estimators,
            max_depth=model_cfg.max_depth,
            gbm_learning_rate=model_cfg.learning_rate,
            subsample=model_cfg.subsample,
            colsample_bytree=model_cfg.colsample_bytree,
            min_child_weight=model_cfg.min_child_weight,
            reg_alpha=model_cfg.reg_alpha,
            reg_lambda=model_cfg.reg_lambda,
            early_stopping_rounds=model_cfg.early_stopping_rounds,
        )
    else:
        return ModelConfig(
            backend="torch",
            model_type=model_cfg.model_type,
            hidden_dims=list(model_cfg.hidden_dims),
            dropout=model_cfg.dropout,
            driver_embed_dim=model_cfg.driver_embed_dim,
            team_embed_dim=model_cfg.team_embed_dim,
            normalize_embeddings=model_cfg.normalize_embeddings,
        )


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    # Log the full resolved config
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    model_cfg = _build_model_config(cfg.model)
    backend = model_cfg.backend
    is_tree = backend in _TREE_BACKENDS

    # Initialize Weights & Biases
    if is_tree:
        run_name = (
            f"{model_cfg.model_type}"
            f"_lr{cfg.model.learning_rate}"
            f"_d{cfg.model.max_depth}"
            f"_n{cfg.model.n_estimators}"
            f"_ss{cfg.model.subsample}"
        )
    else:
        dims = "x".join(str(d) for d in model_cfg.hidden_dims) if model_cfg.hidden_dims else "lin"
        run_name = (
            f"{model_cfg.model_type}_{dims}"
            f"_lr{cfg.model.learning_rate}"
            f"_do{cfg.model.dropout}"
        )
    wandb_group = cfg.wandb.group if cfg.get("wandb", {}).get("group") else None
    wandb.init(
        project="f1prediction",
        name=run_name,
        group=wandb_group,
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=[backend, model_cfg.model_type],
    )

    data_dir = Path(cfg.data.data_dir)
    output_dir = Path(cfg.training.output_dir)
    years = list(cfg.data.years)
    seed = cfg.training.seed
    lookback = cfg.data.lookback
    feature_tiers = list(cfg.data.feature_tiers) if cfg.data.feature_tiers is not None else None
    prepared_dir = Path(cfg.data.prepared_dir) if cfg.data.prepared_dir is not None else None
    max_drivers = cfg.data.max_drivers

    # Enable only requested feature tiers (for ablation studies)
    if feature_tiers is not None:
        REGISTRY.enable_categories(feature_tiers)
        logger.info("Feature tiers: %s", feature_tiers)
    else:
        REGISTRY.enable_all()
        logger.info("Feature tiers: all")

    # Seed all RNGs for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Build historical stats from ALL events in chronological order
    all_events_chrono = get_event_order(data_dir, years)
    logger.info("Building history table from %d events...", len(all_events_chrono))
    history_table, team_history_table = build_history_table(data_dir, all_events_chrono, lookback=lookback)
    logger.info("History table: %d driver, %d team entries (lookback=%d)", len(history_table), len(team_history_table), lookback)

    # Pre-compute all data summaries for fast extraction
    logger.info("Pre-computing data summaries for fast extraction...")
    precomputed = precompute_summaries(data_dir, years)

    # 1. Load or compute split, vocabs, and norm stats
    if prepared_dir is not None:
        logger.info("Loading prepared data from %s", prepared_dir)
        split_data = json.loads((prepared_dir / "split.json").read_text())
        train_events = [tuple(e) for e in split_data["train_events"]]
        val_events = [tuple(e) for e in split_data["val_events"]]
        test_events = [tuple(e) for e in split_data["test_events"]]

        vocabs = json.loads((prepared_dir / "vocabs.json").read_text())
        driver_vocab = vocabs["driver_vocab"]
        team_vocab = vocabs["team_vocab"]
        max_drivers = vocabs["max_drivers"]

        norm_stats = NormalizationStats.load(prepared_dir / "norm_stats.json")
        logger.info(
            "Loaded: %d/%d/%d events, %d drivers, %d teams, %d features",
            len(train_events), len(val_events), len(test_events),
            len(driver_vocab), len(team_vocab), len(norm_stats.feature_names),
        )
    else:
        logger.info("Discovering events for years %s...", years)
        event_order = get_event_order(data_dir, years)
        logger.info("Found %d events", len(event_order))

        train_events, val_events, test_events = split_events(
            event_order, cfg.data.train_frac, cfg.data.val_frac,
            seed=seed,
        )

        driver_vocab, team_vocab = build_vocabularies(data_dir, years)

        # Compute normalization stats from a training subset
        train_samples_raw = generate_samples(data_dir, train_events)
        norm_subset = train_samples_raw
        if len(train_samples_raw) > 500:
            norm_subset = random.sample(train_samples_raw, 500)
        logger.info("Computing normalization stats from %d/%d training samples...",
                     len(norm_subset), len(train_samples_raw))

        norm_extracted = fast_extract_samples(
            data_dir,
            norm_subset,
            REGISTRY,
            max_drivers=max_drivers,
            driver_vocab=driver_vocab,
            team_vocab=team_vocab,
            history_table=history_table,
            team_history_table=team_history_table,
            _precomputed=precomputed,
        )
        norm_stats = compute_stats_numpy(norm_extracted.features, REGISTRY.feature_names)

        # Save artifacts for future runs
        output_dir.mkdir(parents=True, exist_ok=True)
        norm_stats.save(output_dir / "norm_stats.json")
        logger.info("Saved normalization stats to %s", output_dir / "norm_stats.json")

    driver_vocab_size = len(driver_vocab) + 1
    team_vocab_size = len(team_vocab) + 1
    logger.info(
        "Event split: %d train, %d val, %d test",
        len(train_events), len(val_events), len(test_events),
    )
    logger.info("Driver vocab: %d (+UNK), Team vocab: %d (+UNK)", len(driver_vocab), len(team_vocab))

    # 2. Generate samples
    train_samples = generate_samples(data_dir, train_events)
    val_samples = generate_samples(data_dir, val_events)
    test_samples = generate_samples(data_dir, test_events)
    logger.info(
        "Samples: %d train, %d val, %d test",
        len(train_samples), len(val_samples), len(test_samples),
    )

    continuous_dim = REGISTRY.total_dim

    if is_tree:
        return _train_tree(
            cfg, model_cfg, train_samples, val_samples, test_samples,
            driver_vocab, team_vocab, driver_vocab_size, team_vocab_size,
            continuous_dim, norm_stats, history_table, team_history_table,
            data_dir, output_dir, max_drivers, seed, lookback, feature_tiers,
            precomputed,
        )
    else:
        return _train_torch(
            cfg, model_cfg, train_samples, val_samples, test_samples,
            driver_vocab, team_vocab, driver_vocab_size, team_vocab_size,
            continuous_dim, norm_stats, history_table, team_history_table,
            data_dir, output_dir, max_drivers, seed, lookback, feature_tiers,
            precomputed,
        )


def _train_tree(
    cfg, model_cfg, train_samples, val_samples, test_samples,
    driver_vocab, team_vocab, driver_vocab_size, team_vocab_size,
    continuous_dim, norm_stats, history_table, team_history_table,
    data_dir, output_dir, max_drivers, seed, lookback, feature_tiers,
    precomputed,
) -> None:
    """Training path for gradient boosting models."""
    from f1prediction.training.gbm_trainer import GBMTrainer

    shared_kwargs = dict(
        feature_registry=REGISTRY,
        max_drivers=max_drivers,
        driver_vocab=driver_vocab,
        team_vocab=team_vocab,
        history_table=history_table,
        team_history_table=team_history_table,
    )

    logger.info("Extracting features for tree model (no normalization)...")
    train_data = fast_extract_samples(data_dir, train_samples, _precomputed=precomputed, **shared_kwargs)
    val_data = fast_extract_samples(data_dir, val_samples, _precomputed=precomputed, **shared_kwargs)
    test_data = fast_extract_samples(data_dir, test_samples, _precomputed=precomputed, **shared_kwargs)
    logger.info("Feature dim: %d + 2 categorical", continuous_dim)

    model = build_model(
        continuous_dim, model_cfg,
        driver_vocab_size=driver_vocab_size,
        team_vocab_size=team_vocab_size,
        seed=seed,
    )
    logger.info("Model: %s", model.name)

    # Save inference artifacts
    output_dir.mkdir(parents=True, exist_ok=True)
    norm_stats.save(output_dir / "norm_stats.json")
    (output_dir / "vocabs.json").write_text(json.dumps({
        "driver_vocab": driver_vocab,
        "team_vocab": team_vocab,
        "max_drivers": max_drivers,
    }, indent=2))
    (output_dir / "model_config.json").write_text(json.dumps({
        "backend": model_cfg.backend,
        "model_type": model_cfg.model_type,
        "continuous_dim": continuous_dim,
        "driver_vocab_size": driver_vocab_size,
        "team_vocab_size": team_vocab_size,
        "n_estimators": model_cfg.n_estimators,
        "max_depth": model_cfg.max_depth,
        "gbm_learning_rate": model_cfg.gbm_learning_rate,
        "subsample": model_cfg.subsample,
        "colsample_bytree": model_cfg.colsample_bytree,
    }, indent=2))
    # Save full Hydra config for reproducibility
    (output_dir / "hydra_config.yaml").write_text(OmegaConf.to_yaml(cfg))

    trainer = GBMTrainer(
        model=model,
        train_data=train_data,
        val_data=val_data,
        output_dir=output_dir,
        max_drivers=max_drivers,
        feature_names=REGISTRY.feature_names,
    )
    best_val = trainer.fit()
    logger.info("Best validation: %s", best_val)

    test_metrics = trainer.evaluate(test_data)
    _log_test_results(test_metrics)
    _save_results(cfg, model_cfg, test_metrics, best_val, continuous_dim,
                  lookback, feature_tiers, output_dir, n_params=None)
    return best_val["mae_positions"]


def _train_torch(
    cfg, model_cfg, train_samples, val_samples, test_samples,
    driver_vocab, team_vocab, driver_vocab_size, team_vocab_size,
    continuous_dim, norm_stats, history_table, team_history_table,
    data_dir, output_dir, max_drivers, seed, lookback, feature_tiers,
    precomputed,
) -> None:
    """Training path for PyTorch neural network models."""
    unk_dropout = cfg.model.unk_dropout
    batch_size = cfg.data.batch_size

    train_ds = F1RaceDataset(
        data_dir, train_samples, REGISTRY,
        norm_stats=norm_stats, max_drivers=max_drivers,
        driver_vocab=driver_vocab, team_vocab=team_vocab,
        unk_dropout=unk_dropout,
        history_table=history_table, team_history_table=team_history_table,
        _precomputed=precomputed,
    )
    val_ds = F1RaceDataset(
        data_dir, val_samples, REGISTRY,
        norm_stats=norm_stats, max_drivers=max_drivers,
        driver_vocab=driver_vocab, team_vocab=team_vocab,
        history_table=history_table, team_history_table=team_history_table,
        _precomputed=precomputed,
    )
    test_ds = F1RaceDataset(
        data_dir, test_samples, REGISTRY,
        norm_stats=norm_stats, max_drivers=max_drivers,
        driver_vocab=driver_vocab, team_vocab=team_vocab,
        history_table=history_table, team_history_table=team_history_table,
        _precomputed=precomputed,
    )
    logger.info("Feature dim: %d", continuous_dim)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_model(
        continuous_dim, model_cfg,
        driver_vocab_size=driver_vocab_size,
        team_vocab_size=team_vocab_size,
    )

    # Save inference artifacts
    output_dir.mkdir(parents=True, exist_ok=True)
    norm_stats.save(output_dir / "norm_stats.json")
    (output_dir / "vocabs.json").write_text(json.dumps({
        "driver_vocab": driver_vocab,
        "team_vocab": team_vocab,
        "max_drivers": max_drivers,
    }, indent=2))
    (output_dir / "model_config.json").write_text(json.dumps({
        "backend": "torch",
        "model_type": model_cfg.model_type,
        "hidden_dims": model_cfg.hidden_dims,
        "dropout": model_cfg.dropout,
        "continuous_dim": continuous_dim,
        "driver_vocab_size": driver_vocab_size,
        "team_vocab_size": team_vocab_size,
        "driver_embed_dim": model_cfg.driver_embed_dim,
        "team_embed_dim": model_cfg.team_embed_dim,
        "normalize_embeddings": model_cfg.normalize_embeddings,
    }, indent=2))
    # Save full Hydra config for reproducibility
    (output_dir / "hydra_config.yaml").write_text(OmegaConf.to_yaml(cfg))

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "Model: %s | continuous_dim=%d | embed_dim=%d | params=%d",
        model.name, continuous_dim, model.embed_dim, n_params,
    )

    lr = cfg.model.learning_rate
    weight_decay = cfg.model.weight_decay
    embed_l2 = cfg.model.embed_l2
    epochs = cfg.model.epochs
    patience = cfg.model.patience

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=lr,
        weight_decay=weight_decay,
        embed_l2=embed_l2,
        epochs=epochs,
        patience=patience,
        output_dir=output_dir,
        max_drivers=max_drivers,
    )
    best_val = trainer.fit()
    logger.info("Best validation: %s", best_val)

    test_metrics = trainer.evaluate(test_loader)
    _log_test_results(test_metrics)
    _save_results(cfg, model_cfg, test_metrics, best_val, continuous_dim,
                  lookback, feature_tiers, output_dir, n_params=n_params)
    return best_val["mae_positions"]


def _log_test_results(test_metrics: dict[str, float]) -> None:
    logger.info("=" * 60)
    logger.info("TEST RESULTS")
    if "loss" in test_metrics:
        logger.info("  Loss:          %.5f", test_metrics["loss"])
    logger.info("  MAE positions: %.2f", test_metrics["mae_positions"])
    logger.info("  RMSE:          %.2f", test_metrics["rmse_positions"])
    logger.info("  Top-3 acc:     %.2f", test_metrics["top3_accuracy"])
    logger.info("  Exact acc:     %.2f", test_metrics["exact_accuracy"])
    logger.info("=" * 60)

    baseline_mae = 5.5
    if test_metrics["mae_positions"] < baseline_mae:
        logger.info("Model beats constant baseline (MAE %.2f < %.2f)",
                     test_metrics["mae_positions"], baseline_mae)
    else:
        logger.info("Model does NOT beat constant baseline (MAE %.2f >= %.2f)",
                     test_metrics["mae_positions"], baseline_mae)


def _save_results(
    cfg, model_cfg, test_metrics, best_val, continuous_dim,
    lookback, feature_tiers, output_dir, n_params,
) -> None:
    results = {
        "backend": model_cfg.backend,
        "model_type": model_cfg.model_type,
        "lookback": lookback,
        "feature_tiers": feature_tiers or ["core", "form", "derived"],
        "feature_dim": continuous_dim,
        "test_mae": test_metrics["mae_positions"],
        "test_rmse": test_metrics["rmse_positions"],
        "test_top3_acc": test_metrics["top3_accuracy"],
        "test_exact_acc": test_metrics["exact_accuracy"],
        "val_mae": best_val["mae_positions"],
    }
    if "loss" in test_metrics:
        results["test_loss"] = test_metrics["loss"]
    if n_params is not None:
        results["n_params"] = n_params

    # Include all model hyperparameters from the Hydra config
    results["model_config"] = OmegaConf.to_container(cfg.model, resolve=True)

    results_path = output_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2))
    logger.info("Saved results to %s", results_path)

    # Log metrics to W&B
    wandb.log({
        "test/mae": test_metrics["mae_positions"],
        "test/rmse": test_metrics["rmse_positions"],
        "test/top3_accuracy": test_metrics["top3_accuracy"],
        "test/exact_accuracy": test_metrics["exact_accuracy"],
        "val/mae": best_val["mae_positions"],
    })
    wandb.finish()


if __name__ == "__main__":
    main()
