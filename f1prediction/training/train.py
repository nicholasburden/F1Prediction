import copy
import itertools
import time

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
from torch import nn

import wandb

from f1prediction.config import Config, MLPModelConfig, ModelConfig, TrainingConfig
from f1prediction.data.dataloader import F1DataLoader, get_dataloaders
from f1prediction.data.features import ALL_FEATURES
from f1prediction.data.pipeline import build_features
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


def train_model(
    config: Config, should_log: bool = True, save_model: bool = True
) -> float:
    """Train a model with the given config. Returns best validation MAE."""
    cfg = config.training
    mcfg = config.model

    run_dir = config.run_dir()
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(config.model_dump_json(indent=2))

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    all_data, vocab_dict = build_features(cfg.data_dir, cfg.years, ALL_FEATURES)
    vocab_lens = [vocab_dict[col] for col in ALL_FEATURES.embedding_features]

    training_feature_dropout = {"driver_id": cfg.driver_dropout}
    train_dl, val_dl, test_dl, schema = get_dataloaders(
        all_data,
        ALL_FEATURES,
        cfg.batch_size,
        cfg.target_sessions,
        training_feature_dropout,
    )

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

    with wandb.init(
        project="f1prediction",
        name=run_dir.name,
        group=cfg.wandb_group,
        config=config.to_flat_dict(),
    ) as run:
        for t in epoch_iter:
            epoch_start = time.perf_counter()
            if should_log:
                print(f"Epoch {t + 1}\n-------------------------------")
            train_results = train_epoch(
                train_dl, model, loss_fn, optimizer, train_metrics, cfg, should_log
            )
            val_results = evaluate(
                val_dl, model, val_metrics, cfg.device, "Val", should_log
            )
            run.log(
                {
                    **{f"train/{k}": v for k, v in train_results.items()},
                    **{f"val/{k}": v for k, v in val_results.items()},
                }
            )

            val_loss = val_results["mae"]
            epoch_time = time.perf_counter() - epoch_start
            if should_log:
                print(f"Epoch time: {epoch_time:.3f}s")

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

        assert best_weights is not None
        model.load_state_dict(best_weights)

        test_results = evaluate(
            test_dl, model, val_metrics, cfg.device, "Test", should_log
        )
        run.log({f"test/{k}": v for k, v in test_results.items()})

        if save_model:
            model_path = run_dir / "model.pt"
            torch.save(model.state_dict(), model_path)
            if should_log:
                print(f"Model saved to {model_path}")

    if should_log:
        print("Done!")
    return best_loss


@hydra.main(config_path="../../configs", config_name="train", version_base=None)
def run_training(dict_config: DictConfig) -> None:
    raw = OmegaConf.to_container(dict_config, resolve=True)
    config: Config = Config(**raw)  # type: ignore
    train_model(config)


if __name__ == "__main__":
    run_training()
