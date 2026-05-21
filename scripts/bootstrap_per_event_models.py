"""Bootstrap a per-event model dictionary from the deployed global config.

For every event in the current season up to the next-upcoming round, trains a
model with ``event_cutoff = (year, round)`` so it has only seen data strictly
before that event. Each model is deployed into ``<models-dir>/<year>_<round>/``
with the same atomic ``config.json`` + ``checkpoint.pt`` swap pattern the
webapp uses. Hyperparameters come from ``<base-config>/config.json``; only the
cutoff (and CV-derived num_epochs) differ per event.

After bootstrap the webapp's backtest path can look up a per-event snapshot
instead of using the all-data global model, giving honest "before this event"
predictions.

This is a one-time operation — the daily auto-refit hook keeps the dict in
sync as the season progresses.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

from f1prediction.config import Config
from f1prediction.training.refit import full_refit


def _next_upcoming_round(base_config: Config) -> tuple[int, int]:
    """Resolve the (year, round_num) of the next-upcoming session of any
    target type. Used to decide how many per-event models to produce."""
    from scripts.inference import _next_upcoming_session

    year, round_num, _ = _next_upcoming_session(
        base_config.training.target_sessions
    )
    return year, round_num


def _refit_with_cutoff(
    base_config: Config,
    cutoff: tuple[int, int],
    wandb_group: str,
) -> Path:
    """Run ``full_refit`` with the given (year, round) cutoff. Returns the
    run dir where the full-data model was saved."""
    cfg = base_config.model_copy(deep=True)
    cfg.training.event_cutoff = cutoff
    cfg.training.wandb_group = wandb_group
    cfg.training.full_data = False
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "config.json").write_text(cfg.model_dump_json())
        return full_refit(tmp_path, cross_validation=True)


def _deploy(run_dir: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "checkpoint.pt"):
        src = run_dir / name
        dst = target_dir / name
        tmp = dst.with_suffix(dst.suffix + ".new")
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)


def bootstrap(
    base_config_dir: Path,
    models_dir: Path,
    year: int | None,
    end_round: int | None,
    force: bool,
) -> None:
    base_config = Config(
        **json.loads((base_config_dir / "config.json").read_text())
    )

    if year is None or end_round is None:
        auto_year, auto_round = _next_upcoming_round(base_config)
        year = year if year is not None else auto_year
        end_round = end_round if end_round is not None else auto_round
    print(f"Bootstrapping per-event models for {year}, rounds 1..{end_round}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for round_num in range(1, end_round + 1):
        key = f"{year}_{round_num:02d}"
        target = models_dir / key
        if (target / "checkpoint.pt").exists() and not force:
            print(f"  [{key}] already exists, skipping (use --force to redo)")
            continue
        print(f"  [{key}] training with cutoff=({year}, {round_num})…")
        wandb_group = f"per_event_{key}_{timestamp}"
        run_dir = _refit_with_cutoff(base_config, (year, round_num), wandb_group)
        _deploy(run_dir, target)
        print(f"  [{key}] deployed -> {target}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build per-event 'before this event' model snapshots from the "
            "deployed global config."
        ),
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("webapp_config"),
        help="Directory holding the source config.json (default: webapp_config).",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("webapp_models"),
        help="Target root for per-event model dirs (default: webapp_models).",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Season to bootstrap (default: next-upcoming session's year).",
    )
    parser.add_argument(
        "--end-round",
        type=int,
        default=None,
        help="Last round to bootstrap inclusive (default: next-upcoming round).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-train events that already have a checkpoint on disk.",
    )
    args = parser.parse_args()

    bootstrap(
        base_config_dir=args.base_config,
        models_dir=args.models_dir,
        year=args.year,
        end_round=args.end_round,
        force=args.force,
    )


if __name__ == "__main__":
    main()
