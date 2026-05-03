"""Full-data refit driven by an Optuna sweep's parent dir.

Reads ``sweep_meta.json`` from ``--sweep-dir`` to discover the study name and
storage URL (no need to import the sweep script). Loads the study, finds the
best trial, looks up that trial's run dir from its user attrs, and delegates to
``full_refit``.
"""

import argparse
import json
from pathlib import Path

import optuna

from f1prediction.training.refit import full_refit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        required=True,
        help="Parent dir of the sweep (contains sweep_meta.json and per-trial run dirs).",
    )
    epoch_group = parser.add_mutually_exclusive_group()
    epoch_group.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="Epochs for the full-data refit.",
    )
    epoch_group.add_argument(
        "--cross-validation",
        action="store_true",
        help="Run 5-fold CV and use the mean best-epoch for the full-data refit.",
    )
    args = parser.parse_args()

    meta = json.loads((args.sweep_dir / "sweep_meta.json").read_text())
    study = optuna.load_study(study_name=meta["study_name"], storage=meta["storage"])
    best = study.best_trial
    print(f"Study {meta['study_name']!r}: best trial #{best.number}, val MAE {best.value:.4f}")
    for k, v in best.params.items():
        print(f"  {k}: {v}")

    run_dir_str = best.user_attrs.get("run_dir")
    if run_dir_str is None:
        raise RuntimeError(
            f"Best trial #{best.number} has no 'run_dir' user attr — was it run "
            "before sweep.py started recording it?"
        )
    full_refit(
        Path(run_dir_str),
        num_epochs=args.num_epochs,
        cross_validation=args.cross_validation,
    )


if __name__ == "__main__":
    main()
