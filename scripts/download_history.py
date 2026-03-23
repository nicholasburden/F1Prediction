"""CLI entry point to download historical F1 data."""

import argparse
import logging
from pathlib import Path

from f1prediction.data.download import download_history


def main():
    parser = argparse.ArgumentParser(description="Download historical F1 session data")
    parser.add_argument("--start-year", type=int, default=2018)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--no-skip-existing", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    download_history(
        start_year=args.start_year,
        end_year=args.end_year,
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        skip_existing=not args.no_skip_existing,
    )


if __name__ == "__main__":
    main()
