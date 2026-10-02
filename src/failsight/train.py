"""Training entry points for each model family.

Every family is fitted on the ``train`` split and evaluated on the same
held-out ``test`` split, with parameters and metrics logged to MLflow and the
fitted model written to the artifacts directory.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import duckdb
import joblib
import pandas as pd

from failsight.data.split import LABEL_COLUMN, feature_columns, load_split
from failsight.evaluation import evaluate
from failsight.models import anomaly, sequence, supervised
from failsight.models.common import usable_features
from failsight.models.tuning import select_features, time_folds, tune
from failsight.tracking import log_metrics, log_params, tracked_run

log = logging.getLogger(__name__)

RUN_IDS_FILE = "run_ids.json"


def artifacts_dir(cfg: dict) -> Path:
    path = Path(cfg["paths"]["artifacts_dir"])
    path.mkdir(parents=True, exist_ok=True)
    return path


def _record_run(cfg: dict, family: str, run_id: str) -> None:
    path = artifacts_dir(cfg) / RUN_IDS_FILE
    run_ids = json.loads(path.read_text()) if path.exists() else {}
    run_ids[family] = run_id
    path.write_text(json.dumps(run_ids, indent=2))


def _split_info(con: duckdb.DuckDBPyConnection) -> dict:
    test_start, horizon = con.execute("SELECT test_start, horizon_days FROM split_info").fetchone()
    return {"test_start": str(test_start), "horizon_days": int(horizon)}


def train_supervised(
    con: duckdb.DuckDBPyConnection, cfg: dict, families: list[str] | None = None
) -> dict[str, dict]:
    families = families or cfg["supervised"]["families"]
    seed = int(cfg["split"]["seed"])
    tcfg = cfg["supervised"]["tuning"]
    info = _split_info(con)
    train, test = load_split(con, "train"), load_split(con, "test")
    candidates = usable_features(train, feature_columns(con))
    y_train = train[LABEL_COLUMN].to_numpy()
    folds = time_folds(
        train["date"], train["serial_number"], int(tcfg["n_splits"]), info["horizon_days"], seed
    )

    results = {}
    for family in families:
        params = {"family": family, **info, "tuning": tcfg, "n_candidate_features": len(candidates)}
        with tracked_run(cfg, family, params) as run_id:
            study = tune(family, train[candidates], y_train, folds, int(tcfg["n_trials"]), seed)
            best = study.best_params
            selected, _ = select_features(
                family,
                best,
                train[candidates],
                y_train,
                folds[-1],
                float(cfg["supervised"]["selection"]["min_importance"]),
                seed,
            )
            model = supervised.make_pipeline(family, best, seed).fit(train[selected], y_train)
            scores = model.predict_proba(test[selected])[:, 1]
            metrics = evaluate(test["serial_number"], test[LABEL_COLUMN].to_numpy(), scores)

            log_params(best, prefix="best.")
            log_params({"selected_features": ",".join(selected)})
            log_metrics({"cv_average_precision": study.best_value, "n_selected": len(selected)})
            log_metrics(metrics, prefix="test_")
        joblib.dump(
            {"family": family, "pipeline": model, "features": selected},
            artifacts_dir(cfg) / f"{family}.joblib",
        )
        _record_run(cfg, family, run_id)
        results[family] = metrics
    return results


def _targets_sql(split: str) -> str:
    return (
        f"SELECT serial_number, date, {LABEL_COLUMN} AS label FROM dataset WHERE split = '{split}'"
    )


def train_lstm(con: duckdb.DuckDBPyConnection, cfg: dict) -> dict:
    seed = int(cfg["split"]["seed"])
    info = _split_info(con)
    all_cols = sequence.sequence_columns(cfg)
    train_frame = sequence.load_sequence_frame(con, _targets_sql("train"), all_cols)
    columns = sequence.sequence_columns(cfg, train_frame)
    with tracked_run(cfg, "lstm", {"family": "lstm", **info, "lstm": cfg["lstm"]}) as run_id:
        model = sequence.train_lstm(train_frame, columns, cfg, seed)
        test_frame = sequence.load_sequence_frame(con, _targets_sql("test"), all_cols)
        targets = test_frame[test_frame["is_target"]]
        scores = model.predict(test_frame)
        metrics = evaluate(targets["serial_number"], targets["label"].to_numpy(int), scores)
        log_params({"columns": ",".join(columns)})
        log_metrics(metrics, prefix="test_")
    model.save(artifacts_dir(cfg) / "lstm.pt")
    _record_run(cfg, "lstm", run_id)
    return metrics


def train_anomaly(con: duckdb.DuckDBPyConnection, cfg: dict) -> dict[str, dict]:
    seed = int(cfg["split"]["seed"])
    info = _split_info(con)
    train, test = load_split(con, "train"), load_split(con, "test")
    candidates = usable_features(train, feature_columns(con))
    drive_models = anomaly.low_failure_models(train, int(cfg["anomaly"]["min_failures"]))
    params = {
        "family": "anomaly",
        **info,
        "anomaly": cfg["anomaly"],
        "drive_models": ",".join(drive_models),
    }

    fitted: dict[str, anomaly.AnomalyModel] = {}
    results: dict[str, dict] = {}
    with tracked_run(cfg, "anomaly", params) as run_id:
        scored = []
        for drive_model in drive_models:
            fitted[drive_model] = anomaly.fit_anomaly_model(
                train, drive_model, candidates, cfg, seed
            )
            rows = test[test["model"] == drive_model]
            if len(rows):
                scored.append((rows, fitted[drive_model].score(rows)))
        for family in anomaly.ANOMALY_FAMILIES:
            if not scored:
                results[family] = {}
                continue
            serials = [r["serial_number"] for r, _ in scored]
            labels = [r[LABEL_COLUMN] for r, _ in scored]
            metrics = evaluate(
                pd.concat(serials),
                pd.concat(labels).to_numpy(),
                pd.concat([pd.Series(s[family]) for _, s in scored]).to_numpy(),
            )
            log_metrics(metrics, prefix=f"{family}.test_")
            results[family] = metrics
    joblib.dump(fitted, artifacts_dir(cfg) / "anomaly.joblib")
    for family in anomaly.ANOMALY_FAMILIES:
        _record_run(cfg, family, run_id)
    return results


def train_all(con: duckdb.DuckDBPyConnection, cfg: dict, family: str = "all") -> dict:
    results = {}
    if family in ("all", "supervised"):
        results.update(train_supervised(con, cfg))
    if family in ("all", "lstm"):
        results["lstm"] = train_lstm(con, cfg)
    if family in ("all", "anomaly"):
        results.update(train_anomaly(con, cfg))
    return results
