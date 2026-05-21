"""Build a fresh deployable model from scratch.

Pipeline: Optuna sweep over ``--n-trials`` configs → full-data refit on the
best trial (with k-fold CV picking the epoch count) → atomic copy of the
final ``config.json`` + ``checkpoint.pt`` into ``--target-dir``.

This is the bootstrap workflow — produces the artifacts the webapp serves
from ``webapp_config/``. Manually-curated metadata (drivers.json,
track_locations.json) in the target dir is left untouched.

Reuses :func:`scripts.sweep.build_config` and :func:`scripts.sweep.suggest_params`
so the search space stays in one place.
"""

from __future__ import annotations

import argparse
import os
import shutil
from datetime import datetime
from pathlib import Path

import optuna

from f1prediction.training.refit import full_refit
from f1prediction.training.train import train_model
from scripts.sweep import STORAGE, build_config, suggest_params


def run_sweep(n_trials: int, wandb_group: str) -> Path:
    """Run an Optuna study of ``n_trials`` and return the run dir of the
    best (lowest val-MAE) trial."""

    def objective(trial: optuna.Trial) -> float:
        config = build_config(suggest_params(trial), wandb_group=wandb_group)
        best_loss, _, run_dir = train_model(
            config, should_log=False, save_model=False
        )
        trial.set_user_attr("run_dir", str(run_dir))
        return best_loss

    study = optuna.create_study(
        study_name=wandb_group,
        storage=STORAGE,
        direction="minimize",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=n_trials)

    print(f"\nBest trial: val MAE {study.best_trial.value:.4f}")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")

    run_dir = study.best_trial.user_attrs.get("run_dir")
    if run_dir is None:
        raise RuntimeError(
            f"Best trial #{study.best_trial.number} has no 'run_dir' user attr."
        )
    return Path(run_dir)


def deploy(run_dir: Path, target_dir: Path) -> None:
    """Atomically copy config.json + checkpoint.pt from ``run_dir`` into
    ``target_dir`` so any concurrent reader sees a consistent pair."""
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "checkpoint.pt"):
        src = run_dir / name
        dst = target_dir / name
        tmp = dst.with_suffix(dst.suffix + ".new")
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
    print(f"Deployed {run_dir} → {target_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep → full-data refit → deploy into a target dir.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of Optuna trials in the sweep (default: 100).",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=Path("webapp_config"),
        help="Directory to deploy the final config.json + checkpoint.pt into "
        "(default: webapp_config).",
    )
    epoch_group = parser.add_mutually_exclusive_group()
    epoch_group.add_argument(
        "--cv-k",
        type=int,
        default=5,
        help="K for k-fold CV picking the epoch count for the full-data refit "
        "(default: 5). Mutually exclusive with --no-cv.",
    )
    epoch_group.add_argument(
        "--no-cv",
        action="store_true",
        help="Skip k-fold CV; use a single 70/15/15 val split (faster, noisier).",
    )
    args = parser.parse_args()

    wandb_group = f"mlp_build_{args.n_trials}runs_{datetime.now():%Y%m%d_%H%M%S}"

    print(f"=== Sweep: {args.n_trials} trials, group {wandb_group} ===")
    best_run_dir = run_sweep(args.n_trials, wandb_group)

    print(f"\n=== Refitting best trial (base: {best_run_dir}) ===")
    full_run_dir = full_refit(
        best_run_dir,
        cross_validation=not args.no_cv,
        cv_k=args.cv_k,
    )

    print(f"\n=== Deploying {full_run_dir} → {args.target_dir} ===")
    deploy(full_run_dir, args.target_dir)


if __name__ == "__main__":
    main()
