import copy
import itertools
import time
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
from torch import nn

import wandb

from f1prediction.config import Config, MLPModelConfig, ModelConfig, TrainingConfig
from f1prediction.data.dataloader import (
    F1DataLoader,
    get_dataloaders,
    get_full_dataloader,
)
from f1prediction.data.features import CORE_FEATURES, LOOKBACK_FEATURES
from f1prediction.data.registry import FeatureRegistry

_FEATURE_SETS: dict[str, FeatureRegistry] = {
    "core": CORE_FEATURES,
    "lookback": LOOKBACK_FEATURES,
}
from f1prediction.data.pipeline import apply_event_cutoff, build_features
from f1prediction.models.mlp import MLPModel
from f1prediction.training.metrics import Metric


def build_model(
    cfg: ModelConfig, num_numeric: int, vocab_lens: list[int], device: str
) -> nn.Module:
    match cfg:
        case MLPModelConfig():
            return MLPModel(cfg, num_numeric, vocab_lens).to(device)


def build_loss(cfg: TrainingConfig) -> nn.Module:
    if cfg.loss == "mae":
        return nn.L1Loss()
    elif cfg.loss == "mse":
        return nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss: {cfg.loss}")


def build_optimizer(cfg: TrainingConfig, model: nn.Module) -> torch.optim.Optimizer:
    if cfg.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), cfg.lr)
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")


def train_epoch(
    dataloader: F1DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    metrics: list[Metric],
    cfg: TrainingConfig,
    should_log: bool = True,
) -> dict[str, float]:
    model.train()
    for m in metrics:
        m.reset()
    total_loss = 0.0
    num_batches = 0
    for batch_id, (X, cat_ids, y, num_drivers) in enumerate(dataloader):
        X, cat_ids, y, num_drivers = (
            X.to(cfg.device),
            cat_ids.to(cfg.device),
            y.to(cfg.device),
            num_drivers.to(cfg.device),
        )
        num_batches += 1
        pred: torch.Tensor = model(X, cat_ids)
        loss: torch.Tensor = loss_fn(pred, y) / cfg.gradient_accumulation
        total_loss += loss.item() * cfg.gradient_accumulation

        loss.backward()

        if (batch_id + 1) % cfg.gradient_accumulation == 0:
            optimizer.step()
            optimizer.zero_grad()

        with torch.no_grad():
            pred_pos = pred * num_drivers
            actual_pos = y * num_drivers
            for m in metrics:
                m.update(pred_pos, actual_pos)

    if num_batches % cfg.gradient_accumulation != 0:
        optimizer.step()
        optimizer.zero_grad()

    results = {m.name(): m.compute() for m in metrics}
    if should_log:
        parts = [f"{k}: {v:.4f}" for k, v in results.items()]
        print(f"Train | {', '.join(parts)}")
    return results


def evaluate(
    dataloader: F1DataLoader,
    model: nn.Module,
    metrics: list[Metric],
    device: str,
    name: str,
    should_log: bool = True,
) -> dict[str, float]:
    model.eval()
    for m in metrics:
        m.reset()
    with torch.no_grad():
        for X, cat_ids, y, num_drivers in dataloader:
            X, cat_ids, y, num_drivers = (
                X.to(device),
                cat_ids.to(device),
                y.to(device),
                num_drivers.to(device),
            )
            pred: torch.Tensor = model(X, cat_ids)
            pred_pos = pred * num_drivers
            actual_pos = y * num_drivers
            for m in metrics:
                m.update(pred_pos, actual_pos)

    results = {m.name(): m.compute() for m in metrics}
    if should_log:
        parts = [f"{k}: {v:.4f}" for k, v in results.items()]
        print(f"{name} | {', '.join(parts)}")
    return results


def train_with_dataloaders(
    cfg: TrainingConfig,
    mcfg: ModelConfig,
    train_dl: F1DataLoader,
    val_dl: F1DataLoader | None,
    schema,
    vocab_lens: list[int],
    wandb_run=None,
    should_log: bool = True,
) -> tuple[float, int, nn.Module]:
    """Inner training loop. Builds model/loss/optimizer/metrics from ``cfg``,
    iterates over ``train_dl`` (and optionally ``val_dl``), and returns
    ``(best_loss, best_epoch, model)``.

    With a ``val_dl``: val-based early stopping; the returned model has the
    best-epoch weights restored. Without one: requires ``cfg.num_epochs`` and
    runs that many epochs end-to-end (no early stopping); returned best_loss is
    final train MAE and best_epoch is ``cfg.num_epochs``.
    """
    if val_dl is None and cfg.num_epochs is None:
        raise ValueError("Either val_dl or cfg.num_epochs must be set")

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    model = build_model(mcfg, len(schema.numeric_cols), vocab_lens, cfg.device)
    loss_fn = build_loss(cfg)
    optimizer = build_optimizer(cfg, model)
    train_metrics = [mc.build() for mc in cfg.train_metrics]
    val_metrics = [mc.build() for mc in cfg.val_metrics]

    best_loss = float("inf")
    best_epoch = -1
    best_weights = None
    patience_counter = 0

    epoch_iter = (
        range(cfg.num_epochs) if cfg.num_epochs is not None else itertools.count()
    )

    train_results: dict[str, float] = {}
    for t in epoch_iter:
        epoch_start = time.perf_counter()
        if should_log:
            print(f"Epoch {t + 1}\n-------------------------------")
        train_results = train_epoch(
            train_dl, model, loss_fn, optimizer, train_metrics, cfg, should_log
        )
        log_payload: dict[str, float] = {
            f"train/{k}": v for k, v in train_results.items()
        }

        if val_dl is not None:
            val_results = evaluate(
                val_dl, model, val_metrics, cfg.device, "Val", should_log
            )
            log_payload.update({f"val/{k}": v for k, v in val_results.items()})
        if wandb_run is not None:
            wandb_run.log(log_payload)

        epoch_time = time.perf_counter() - epoch_start
        if should_log:
            print(f"Epoch time: {epoch_time:.3f}s")

        if val_dl is None:
            continue

        val_loss = val_results["mae"]
        if val_loss < best_loss - cfg.min_delta:
            best_loss = val_loss
            best_epoch = t + 1
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= cfg.patience:
            if should_log:
                print(
                    f"Early stopping, best val loss {best_loss}, epoch {best_epoch}"
                )
            break

    if val_dl is None:
        assert cfg.num_epochs is not None
        return train_results["mae"], cfg.num_epochs, model

    assert best_weights is not None
    model.load_state_dict(best_weights)
    return best_loss, best_epoch, model


def train_model(
    config: Config, should_log: bool = True, save_model: bool = True
) -> tuple[float, int, Path]:
    """Train a model with the given config. Returns
    ``(best_loss, best_epoch, run_dir)``: val MAE + 1-indexed epoch in normal
    mode, or final train MAE + ``num_epochs`` in ``full_data`` mode."""
    cfg = config.training
    mcfg = config.model

    if cfg.full_data and cfg.num_epochs is None:
        raise ValueError("full_data=True requires num_epochs to be set (no early stopping)")

    run_dir = config.run_dir()
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(config.model_dump_json(indent=2))

    features = sum(
        (_FEATURE_SETS[name] for name in cfg.feature_sets[1:]),
        _FEATURE_SETS[cfg.feature_sets[0]],
    )
    all_data, vocab_dict, vocab_mappings = build_features(
        cfg.data_dir, cfg.years, features
    )
    all_data = apply_event_cutoff(all_data, cfg.event_cutoff)
    vocab_lens = [vocab_dict[col] for col in features.embedding_features]

    training_feature_dropout = {"driver_id": cfg.driver_dropout}
    training_block_dropout = {"weather": cfg.weather_dropout}
    if cfg.full_data:
        train_dl, schema = get_full_dataloader(
            all_data,
            features,
            seed=cfg.seed,
            batch_size=cfg.batch_size,
            target_sessions=cfg.target_sessions,
            training_feature_dropout=training_feature_dropout,
            training_block_dropout=training_block_dropout,
        )
        val_dl: F1DataLoader | None = None
        test_dl: F1DataLoader | None = None
    else:
        train_dl, val_dl, test_dl, schema = get_dataloaders(
            all_data,
            features,
            seed=cfg.seed,
            batch_size=cfg.batch_size,
            target_sessions=cfg.target_sessions,
            training_feature_dropout=training_feature_dropout,
            training_block_dropout=training_block_dropout,
        )

    with wandb.init(
        project="f1prediction",
        name=run_dir.name,
        group=cfg.wandb_group,
        config=config.to_flat_dict(),
    ) as run:
        best_loss, best_epoch, model = train_with_dataloaders(
            cfg, mcfg, train_dl, val_dl, schema, vocab_lens,
            wandb_run=run, should_log=should_log,
        )

        if test_dl is not None:
            val_metrics = [mc.build() for mc in cfg.val_metrics]
            test_results = evaluate(
                test_dl, model, val_metrics, cfg.device, "Test", should_log
            )
            run.log({f"test/{k}": v for k, v in test_results.items()})

        if save_model:
            checkpoint_path = run_dir / "checkpoint.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "numeric_cols": schema.numeric_cols,
                    "embedding_cols": schema.embedding_cols,
                    "norm_mean": schema.norm_stats.mean,
                    "norm_std": schema.norm_stats.std,
                    "vocab_lens": vocab_lens,
                    "vocab_mappings": vocab_mappings,
                },
                checkpoint_path,
            )
            if should_log:
                print(f"Checkpoint saved to {checkpoint_path}")

    if should_log:
        print("Done!")
    return best_loss, best_epoch, run_dir


@hydra.main(config_path="../../configs", config_name="train", version_base=None)
def run_training(dict_config: DictConfig) -> None:
    raw = OmegaConf.to_container(dict_config, resolve=True)
    config: Config = Config(**raw)  # type: ignore
    train_model(config)


if __name__ == "__main__":
    run_training()
