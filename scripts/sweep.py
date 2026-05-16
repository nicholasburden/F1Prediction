"""Optuna hyperparameter sweep for F1 prediction models."""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import optuna

from f1prediction.config import Config, MLPModelConfig, TrainingConfig
from f1prediction.training.refit import full_refit
from f1prediction.training.train import train_model

STORAGE = "sqlite:///sweep.db"


def suggest_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "embedding_dim": trial.suggest_int("embedding_dim", 2, 16),
        "hidden_dim": trial.suggest_int("hidden_dim", 32, 512, step=32),
        "num_hidden_layers": trial.suggest_int("num_hidden_layers", 1, 16),
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
        "gradient_accumulation": trial.suggest_categorical(
            "gradient_accumulation", [2**i for i in range(5)]
        ),
        "driver_dropout": trial.suggest_float("driver_dropout", 0.0, 0.5),
    }


def build_config(params: dict[str, Any], wandb_group: str | None) -> Config:
    model_cfg = MLPModelConfig(
        type="mlp",
        embedding_dim=params["embedding_dim"],
        hidden_dim=params["hidden_dim"],
        num_hidden_layers=params["num_hidden_layers"],
    )
    training_cfg = TrainingConfig(
        seed=42,
        lr=params["lr"],
        patience=10,
        min_delta=0.0001,
        batch_size=params["batch_size"],
        gradient_accumulation=params["gradient_accumulation"],
        driver_dropout=params["driver_dropout"],
        loss="mae",
        optimizer="adam",
        device="cpu",
        target_sessions=["Sprint", "Q", "R"],
        years=[2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026],
        data_dir=Path("data"),
        output_dir=Path("runs"),
        wandb_group=wandb_group,
    )
    return Config(training=training_cfg, model=model_cfg)


def objective(trial: optuna.Trial) -> float:
    config = build_config(suggest_params(trial), wandb_group=SWEEP_WANDB_GROUP)
    best_loss, _, run_dir = train_model(config, should_log=False, save_model=False)
    trial.set_user_attr("run_dir", str(run_dir))
    return best_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of Optuna trials.",
    )
    parser.add_argument(
        "--refit",
        action="store_true",
        help="After the sweep, refit the best trial on every event (no val/test).",
    )
    epoch_group = parser.add_mutually_exclusive_group()
    epoch_group.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="Epochs for the full-data refit. Requires --refit.",
    )
    epoch_group.add_argument(
        "--cross-validation",
        action="store_true",
        help="Use 5-fold CV mean best-epoch for the full-data refit. Requires --refit.",
    )
    args = parser.parse_args()

    if (args.num_epochs is not None or args.cross_validation) and not args.refit:
        parser.error("--num-epochs/--cross-validation require --refit")

    SWEEP_WANDB_GROUP = (
        f"mlp_sweep_{args.n_trials}runs_{datetime.now():%Y%m%d_%H%M%S}"
    )

    study = optuna.create_study(
        study_name=SWEEP_WANDB_GROUP,
        storage=STORAGE,
        direction="minimize",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=args.n_trials)

    print("\nBest trial:")
    print(f"Value (MAE): {study.best_trial.value:.4f}")
    print("Params:")
    for k, v in study.best_trial.params.items():
        print(f"{k}: {v}")

    sweep_dir = Path("runs") / SWEEP_WANDB_GROUP
    sweep_dir.mkdir(parents=True, exist_ok=True)
    (sweep_dir / "sweep_meta.json").write_text(
        json.dumps(
            {"study_name": SWEEP_WANDB_GROUP, "storage": STORAGE},
            indent=2,
        )
    )
    print(f"\nWrote sweep metadata to {sweep_dir / 'sweep_meta.json'}")

    if args.refit:
        best_run_dir = Path(study.best_trial.user_attrs["run_dir"])
        print(f"\n=== Refitting best trial on full data (base run dir: {best_run_dir}) ===")
        full_refit(
            best_run_dir,
            num_epochs=args.num_epochs,
            cross_validation=args.cross_validation,
        )
