"""Optuna hyperparameter sweep for F1 prediction models."""

from pathlib import Path

import optuna

from f1prediction.config import Config, MLPModelConfig, TrainingConfig
from f1prediction.training.train import train_model


def objective(trial: optuna.Trial) -> float:
    model_cfg = MLPModelConfig(
        type="mlp",
        embedding_dim=trial.suggest_int("embedding_dim", 2, 16),
        hidden_dim=trial.suggest_int("hidden_dim", 32, 512, step=32),
        num_hidden_layers=trial.suggest_int("num_hidden_layers", 1, 16),
    )
    training_cfg = TrainingConfig(
        seed=42,
        lr=trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        patience=10,
        min_delta=0.0001,
        batch_size=trial.suggest_categorical("batch_size", [64, 128, 256, 512]),  # type: ignore
        gradient_accumulation=trial.suggest_categorical(
            "gradient_accumulation", [2**i for i in range(5)]
        ),  # type: ignore
        driver_dropout=trial.suggest_float("driver_dropout", 0.0, 0.5),
        loss="mae",
        optimizer="adam",
        device="cpu",
        target_sessions=["Sprint", "Q", "R"],
        years=[2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026],
        data_dir=Path("data"),
        output_dir=Path("runs"),
        wandb_group="mlp_sweep_250runs_20260406",
    )
    config = Config(training=training_cfg, model=model_cfg)
    return train_model(config, should_log=False, save_model=False)


if __name__ == "__main__":
    study = optuna.create_study(
        study_name="f1prediction",
        direction="minimize",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=250)

    print("\nBest trial:")
    print(f"Value (MAE): {study.best_trial.value:.4f}")
    print("Params:")
    for k, v in study.best_trial.params.items():
        print(f"{k}: {v}")
