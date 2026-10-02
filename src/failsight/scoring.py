"""Produce the failure-prediction scores table consumed by TestLens.

The schema is documented in ``docs/scores_schema.md`` and defined by
``SCORES_SCHEMA`` below; both must change together.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path

import duckdb
import joblib
import pandas as pd

from failsight.data.split import LABEL_COLUMN
from failsight.models import sequence
from failsight.models.supervised import FAMILIES
from failsight.train import RUN_IDS_FILE

log = logging.getLogger(__name__)

SCORES_TABLE = "failure_scores"

# (column, DuckDB type)
SCORES_SCHEMA: list[tuple[str, str]] = [
    ("serial_number", "VARCHAR"),
    ("model", "VARCHAR"),
    ("score_date", "DATE"),
    ("model_family", "VARCHAR"),
    ("score", "DOUBLE"),
    ("score_type", "VARCHAR"),
    ("horizon_days", "INTEGER"),
    ("run_id", "VARCHAR"),
    ("scored_at", "TIMESTAMP"),
]


def target_rows(con: duckdb.DuckDBPyConnection, scope: str) -> tuple[pd.DataFrame, str]:
    """Rows to score and an equivalent SQL query of (serial_number, date, label).

    ``test``: the held-out split. ``latest``: every drive on the last day of data.
    """
    if scope == "test":
        sql = f"SELECT *, {LABEL_COLUMN} AS label FROM dataset WHERE split = 'test'"
    elif scope == "latest":
        sql = (
            f"SELECT *, {LABEL_COLUMN} AS label FROM features "
            "WHERE date = (SELECT max(date) FROM features)"
        )
    else:
        raise ValueError(f"Unknown scope: {scope}")
    rows = con.execute(f"{sql} ORDER BY serial_number, date").df()
    return rows, sql


def _frame(rows: pd.DataFrame, family: str, scores, score_type: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "serial_number": rows["serial_number"].to_numpy(),
            "model": rows["model"].to_numpy(),
            "score_date": rows["date"].to_numpy(),
            "model_family": family,
            "score": scores,
            "score_type": score_type,
        }
    )


def score(con: duckdb.DuckDBPyConnection, cfg: dict, scope: str = "test") -> pd.DataFrame:
    art = Path(cfg["paths"]["artifacts_dir"])
    run_ids_path = art / RUN_IDS_FILE
    run_ids = json.loads(run_ids_path.read_text()) if run_ids_path.exists() else {}
    horizon = int(cfg["data"]["label_horizon_days"])
    rows, targets_sql = target_rows(con, scope)
    parts = []

    for family in FAMILIES:
        path = art / f"{family}.joblib"
        if path.exists():
            bundle = joblib.load(path)
            scores = bundle["pipeline"].predict_proba(rows[bundle["features"]])[:, 1]
            parts.append(_frame(rows, family, scores, "classifier_score"))

    lstm_path = art / "lstm.pt"
    if lstm_path.exists():
        model = sequence.SequenceModel.load(lstm_path)
        frame = sequence.load_sequence_frame(
            con, f"SELECT serial_number, date, label FROM ({targets_sql})", model.columns
        )
        targets = frame[frame["is_target"]]
        parts.append(_frame(targets, "lstm", model.predict(frame), "classifier_score"))

    anomaly_path = art / "anomaly.joblib"
    if anomaly_path.exists():
        for drive_model, model in joblib.load(anomaly_path).items():
            subset = rows[rows["model"] == drive_model]
            if subset.empty:
                continue
            for family, scores in model.score(subset).items():
                parts.append(_frame(subset, family, scores, "anomaly_percentile"))

    if not parts:
        raise FileNotFoundError(f"No trained models found in {art}; run `failsight train` first.")
    out = pd.concat(parts, ignore_index=True)
    out["horizon_days"] = horizon
    out["run_id"] = out["model_family"].map(run_ids)
    out["scored_at"] = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    return out


def write_scores(con: duckdb.DuckDBPyConnection, scores: pd.DataFrame, parquet_path: str | Path):
    """Write the scores to the ``failure_scores`` DuckDB table and a Parquet file."""
    casts = ", ".join(f"CAST({col} AS {typ}) AS {col}" for col, typ in SCORES_SCHEMA)
    con.register("scores_df", scores)
    con.execute(f"CREATE OR REPLACE TABLE {SCORES_TABLE} AS SELECT {casts} FROM scores_df")
    con.unregister("scores_df")
    parquet_path = Path(parquet_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    target = str(parquet_path).replace("\\", "/").replace("'", "''")
    con.execute(f"COPY {SCORES_TABLE} TO '{target}' (FORMAT parquet)")
    return parquet_path
