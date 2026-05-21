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
from typing import Callable

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
    on_phase: Callable[[str], None] | None = None,
) -> Path:
    """Load ``base_run_dir/config.json`` as the base config and run the refit.
    Returns the run dir where the full-data model was saved.

    Stopping epoch is picked in priority order: ``num_epochs`` if provided;
    otherwise K-fold CV mean if ``cross_validation``; otherwise stage-1 early
    stopping. ``num_epochs`` and ``cross_validation`` are mutually exclusive.

    ``on_phase`` is invoked with a short user-facing string at each phase
    boundary so callers (e.g. the webapp) can surface progress.
    """
    if num_epochs is not None and cross_validation:
        raise ValueError("num_epochs and cross_validation are mutually exclusive")

    def _phase(msg: str) -> None:
        if on_phase is not None:
            on_phase(msg)

    today = date.today().strftime("%Y%m%d")
    val_group = val_wandb_group or f"mlp_val_for_full_{today}"
    full_group = full_wandb_group or f"mlp_full_{today}"

    config = Config(**json.loads((base_run_dir / "config.json").read_text()))
    print(f"Loaded base config from {base_run_dir}")

    if cross_validation:
        _phase(f"refitting with {cv_k}-fold cross-validation")
        print(f"\n=== {cv_k}-fold CV to find optimal stopping epoch ===")
        config.training.full_data = False
        cv_result = kfold_cv(config, k=cv_k, should_log=True)
        num_epochs = round(cv_result.mean_epoch)
        print(
            f"\nCV done — val MAE {cv_result.mean_loss:.4f} ± "
            f"{cv_result.std_loss:.4f}, best epoch "
            f"{cv_result.mean_epoch:.1f} ± {cv_result.std_epoch:.1f}"
        )
        print(f"Using num_epochs = {num_epochs}")
        _phase(f"CV done — using {num_epochs} epochs")
    elif num_epochs is None:
        _phase("refitting on val split to find best epoch (stage 1)")
        print("\n=== Stage 1: validated run to find optimal stopping epoch ===")
        config.training.wandb_group = val_group
        config.training.full_data = False
        val_loss, num_epochs, _ = train_model(
            config, should_log=True, save_model=False
        )
        print(f"\nStage 1 done — best val MAE {val_loss:.4f} at epoch {num_epochs}")
        _phase(f"stage 1 done — best epoch {num_epochs} (val MAE {val_loss:.3f})")
    else:
        print(f"\nUsing num_epochs={num_epochs}; skipping stage-1 validated run.")
        _phase(f"using preset {num_epochs} epochs; skipping val stage")

    _phase(f"refit with all data for {num_epochs} epochs (stage 2)")
    print(f"\n=== Full-data refit for {num_epochs} epochs (no early stopping) ===")
    config.training.full_data = True
    config.training.num_epochs = num_epochs
    config.training.wandb_group = full_group
    _, _, run_dir = train_model(config, should_log=True, save_model=True)
    return run_dir
