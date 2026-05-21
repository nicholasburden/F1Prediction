"""K-fold cross-validation. Trains K models — each on K-1 folds with the
held-out fold as validation — and returns per-fold val MAE and best stopping
epoch. Use the mean ``best_epoch`` as ``num_epochs`` for a full-data refit.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

import polars as pl

from f1prediction.config import Config
from f1prediction.data.dataloader import get_kfold_dataloaders
from f1prediction.data.features import CORE_FEATURES, LOOKBACK_FEATURES
from f1prediction.data.pipeline import apply_event_cutoff, build_features
from f1prediction.data.registry import FeatureRegistry
from f1prediction.training.train import train_with_dataloaders

_FEATURE_SETS: dict[str, FeatureRegistry] = {
    "core": CORE_FEATURES,
    "lookback": LOOKBACK_FEATURES,
}


@dataclass
class CVResult:
    fold_losses: list[float]
    fold_epochs: list[int]

    @property
    def mean_loss(self) -> float:
        return statistics.mean(self.fold_losses)

    @property
    def std_loss(self) -> float:
        return statistics.stdev(self.fold_losses) if len(self.fold_losses) > 1 else 0.0

    @property
    def mean_epoch(self) -> float:
        return statistics.mean(self.fold_epochs)

    @property
    def std_epoch(self) -> float:
        return statistics.stdev(self.fold_epochs) if len(self.fold_epochs) > 1 else 0.0


def _run_fold(
    config: Config,
    all_data: pl.DataFrame,
    features: FeatureRegistry,
    vocab_dict: dict[str, int],
    fold_idx: int,
    k: int,
) -> tuple[float, int]:
    cfg = config.training
    train_dl, val_dl, schema = get_kfold_dataloaders(
        all_data,
        features,
        k=k,
        fold_idx=fold_idx,
        seed=cfg.seed,
        batch_size=cfg.batch_size,
        target_sessions=cfg.target_sessions,
        training_feature_dropout={"driver_id": cfg.driver_dropout},
        training_block_dropout={"weather": cfg.weather_dropout},
    )
    vocab_lens = [vocab_dict[col] for col in features.embedding_features]
    best_loss, best_epoch, _ = train_with_dataloaders(
        cfg, config.model, train_dl, val_dl, schema, vocab_lens,
        wandb_run=None, should_log=False,
    )
    return best_loss, best_epoch


def kfold_cv(config: Config, k: int = 5, should_log: bool = True) -> CVResult:
    """Run K-fold CV with the given config. Features are built once and reused
    across folds. Returns per-fold val MAE and best epoch."""
    if config.training.full_data:
        raise ValueError("kfold_cv requires full_data=False (need a held-out fold)")

    features = sum(
        (_FEATURE_SETS[name] for name in config.training.feature_sets[1:]),
        _FEATURE_SETS[config.training.feature_sets[0]],
    )
    if should_log:
        print("Building features…")
    all_data, vocab_dict, _ = build_features(
        config.training.data_dir, config.training.years, features
    )
    all_data = apply_event_cutoff(all_data, config.training.event_cutoff)

    losses: list[float] = []
    epochs: list[int] = []
    for fold_idx in range(k):
        if should_log:
            print(f"\n=== Fold {fold_idx + 1}/{k} ===")
        loss, epoch = _run_fold(config, all_data, features, vocab_dict, fold_idx, k)
        if should_log:
            print(f"Fold {fold_idx + 1}: val MAE {loss:.4f}, best epoch {epoch}")
        losses.append(loss)
        epochs.append(epoch)

    return CVResult(fold_losses=losses, fold_epochs=epochs)
