"""Two-stage full-data refit driven by a saved run's config.

Stage 1: validated run on the seeded 70/15/15 split, used only to find the
optimal stopping epoch via early stopping. No model saved.
Stage 2: refit on every event for that many epochs, no held-out set.

Without a val set you can't see overfitting from inside the loop, so the
val-derived epoch count is the proxy for "stop here". Caller can override by
passing ``num_epochs`` directly (e.g. from a CV mean).
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from f1prediction.config import Config
from f1prediction.training.cv import kfold_cv
from f1prediction.training.train import train_model


def full_refit(
    base_run_dir: Path,
    num_epochs: int | None = None,
    cross_validation: bool = False,
    cv_k: int = 5,
    val_wandb_group: str | None = None,
    full_wandb_group: str | None = None,
) -> Path:
    """Load ``base_run_dir/config.json`` as the base config and run the refit.
    Returns the run dir where the full-data model was saved.

    Stopping epoch is picked in priority order: ``num_epochs`` if provided;
    otherwise K-fold CV mean if ``cross_validation``; otherwise stage-1 early
    stopping. ``num_epochs`` and ``cross_validation`` are mutually exclusive.
    """
    if num_epochs is not None and cross_validation:
        raise ValueError("num_epochs and cross_validation are mutually exclusive")

    today = date.today().strftime("%Y%m%d")
    val_group = val_wandb_group or f"mlp_val_for_full_{today}"
    full_group = full_wandb_group or f"mlp_full_{today}"

    base_config = Config(**json.loads((base_run_dir / "config.json").read_text()))
    print(f"Loaded base config from {base_run_dir}")

    if cross_validation:
        print(f"\n=== {cv_k}-fold CV to find optimal stopping epoch ===")
        cv_result = kfold_cv(base_config, k=cv_k, should_log=True)
        num_epochs = round(cv_result.mean_epoch)
        print(
            f"\nCV done — val MAE {cv_result.mean_loss:.4f} ± "
            f"{cv_result.std_loss:.4f}, best epoch "
            f"{cv_result.mean_epoch:.1f} ± {cv_result.std_epoch:.1f}"
        )
        print(f"Using num_epochs = {num_epochs}")
    elif num_epochs is None:
        print("\n=== Stage 1: validated run to find optimal stopping epoch ===")
        val_config = base_config.model_copy(deep=True)
        val_config.training.wandb_group = val_group
        val_loss, num_epochs, _ = train_model(
            val_config, should_log=True, save_model=False
        )
        print(f"\nStage 1 done — best val MAE {val_loss:.4f} at epoch {num_epochs}")
    else:
        print(f"\nUsing num_epochs={num_epochs}; skipping stage-1 validated run.")

    print(f"\n=== Full-data refit for {num_epochs} epochs (no early stopping) ===")
    full_config = base_config.model_copy(deep=True)
    full_config.training.full_data = True
    full_config.training.num_epochs = num_epochs
    full_config.training.wandb_group = full_group
    _, _, run_dir = train_model(full_config, should_log=True, save_model=True)
    return run_dir
