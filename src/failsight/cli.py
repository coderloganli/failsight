"""Command-line entry points: download, build, train, score."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from failsight.config import load_config


def _cmd_download(cfg: dict, args: argparse.Namespace) -> None:
    from failsight.download import download

    if args.periods:
        cfg["download"]["periods"] = args.periods
    files = download(cfg)
    logging.info("Extracted %d daily CSVs into %s", len(files), cfg["paths"]["raw_dir"])


def _cmd_build(cfg: dict, args: argparse.Namespace) -> None:
    from failsight.data import build_dataset, connect

    raw_dir = Path(args.raw_dir or cfg["paths"]["raw_dir"])
    with connect(cfg["paths"]["database"]) as con:
        info = build_dataset(con, cfg, raw_dir)
    logging.info("Built dataset: %s", info)


def _cmd_train(cfg: dict, args: argparse.Namespace) -> None:
    from failsight.data import connect
    from failsight.train import train_all

    with connect(cfg["paths"]["database"]) as con:
        train_all(con, cfg, args.family)
    logging.info("Training finished; metrics are in MLflow (%s)", cfg["tracking"]["tracking_uri"])


def _cmd_score(cfg: dict, args: argparse.Namespace) -> None:
    from failsight.data import connect
    from failsight.scoring import score, write_scores

    with connect(cfg["paths"]["database"]) as con:
        scores = score(con, cfg, args.scope)
        path = write_scores(con, scores, args.output or cfg["paths"]["scores_path"])
    logging.info("Wrote %d scores to %s", len(scores), path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="failsight", description=__doc__)
    parser.add_argument("--config", default=None, help="YAML config layered over the default")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="Download and extract Backblaze archives")
    p.add_argument("--periods", nargs="+", help="e.g. Q1_2026 Q2_2026 (overrides config)")
    p.set_defaults(func=_cmd_download)

    p = sub.add_parser("build", help="Ingest, clean, build features and split in DuckDB")
    p.add_argument("--raw-dir", default=None)
    p.set_defaults(func=_cmd_build)

    p = sub.add_parser("train", help="Train and evaluate model families")
    p.add_argument("--family", default="all", choices=["all", "supervised", "lstm", "anomaly"])
    p.set_defaults(func=_cmd_train)

    p = sub.add_parser("score", help="Write the failure-prediction scores table")
    p.add_argument("--scope", default="test", choices=["test", "latest"])
    p.add_argument("--output", default=None, help="Parquet path (overrides config)")
    p.set_defaults(func=_cmd_score)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    args.func(cfg, args)


if __name__ == "__main__":
    main()
