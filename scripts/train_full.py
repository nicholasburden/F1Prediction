"""Refit the config saved in any prior run dir on every event (no val/test).

Pass ``--run-dir`` pointing at any run directory that contains a ``config.json``.
"""

import argparse
from pathlib import Path

from f1prediction.training.refit import full_refit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run dir containing config.json to use as the base config.",
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
    full_refit(
        args.run_dir,
        num_epochs=args.num_epochs,
        cross_validation=args.cross_validation,
    )


if __name__ == "__main__":
    main()
