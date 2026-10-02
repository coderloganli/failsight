"""Run the full data layer: ingest, clean, features, split."""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb

from failsight.data.clean import build_clean
from failsight.data.features import build_features
from failsight.data.ingest import ingest_snapshots
from failsight.data.split import build_split

log = logging.getLogger(__name__)


def connect(path: str | Path, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    path = Path(path)
    if not read_only:
        path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path), read_only=read_only)


def build_dataset(con: duckdb.DuckDBPyConnection, cfg: dict, raw_dir: str | Path) -> dict:
    n_raw = ingest_snapshots(con, raw_dir, cfg)
    n_clean = build_clean(con, cfg)
    n_feat = build_features(con, cfg)
    info = build_split(con, cfg)
    log.info("snapshots=%d clean=%d features=%d split=%s", n_raw, n_clean, n_feat, info["rows"])
    return {"snapshots": n_raw, "clean": n_clean, "features": n_feat, **info}
