"""Smoke tests: each model family trains on CPU on the synthetic fixture.

These check mechanics only. Metrics on synthetic data mean nothing and are not asserted.
"""

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from failsight.data import load_split
from failsight.evaluation import pr_metrics
from failsight.models.anomaly import low_failure_models
from failsight.models.sequence import build_sequences
from failsight.scoring import SCORES_SCHEMA, score, write_scores

SUPERVISED = ["logistic_regression", "random_forest", "gradient_boosting"]


def test_every_family_trains_and_reports_pr_metrics(trained):
    for family in [*SUPERVISED, "lstm"]:
        assert "average_precision" in trained[family]
        assert "drive_average_precision" in trained[family]
    for family in ["isolation_forest", "autoencoder"]:
        assert family in trained


def test_artifacts_are_written(trained, session_cfg):
    art = Path(session_cfg["paths"]["artifacts_dir"])
    for name in [*(f"{f}.joblib" for f in SUPERVISED), "lstm.pt", "anomaly.joblib"]:
        assert (art / name).exists(), name


def test_runs_are_tracked_in_mlflow(trained, session_cfg):
    mlflow.set_tracking_uri(session_cfg["tracking"]["tracking_uri"])
    runs = mlflow.search_runs(experiment_names=[session_cfg["tracking"]["experiment"]])
    assert set(runs["tags.mlflow.runName"]) >= {*SUPERVISED, "lstm", "anomaly"}
    assert "params.horizon_days" in runs.columns
    assert any(c.startswith("metrics.") for c in runs.columns)


def test_unsupervised_path_targets_low_failure_models(con, session_cfg):
    train = load_split(con, "train")
    models = low_failure_models(train, session_cfg["anomaly"]["min_failures"])
    assert "SYN-MODEL-B" in models
    assert "SYN-MODEL-A" not in models


def test_scores_table_matches_schema(trained, con, session_cfg, tmp_path):
    scores = score(con, session_cfg, "test")
    path = write_scores(con, scores, tmp_path / "scores.parquet")
    table = pd.read_parquet(path)
    assert list(table.columns) == [c for c, _ in SCORES_SCHEMA]
    assert set(table["model_family"]) == {*SUPERVISED, "lstm", "isolation_forest", "autoencoder"}
    assert table["score"].between(0, 1).all()
    assert table["run_id"].notna().all()
    assert set(table["score_type"]) == {"classifier_score", "anomaly_percentile"}
    n_test = len(load_split(con, "test"))
    assert (table.groupby("model_family").size()[SUPERVISED + ["lstm"]] == n_test).all()


def test_latest_scope_scores_every_drive_on_last_day(trained, con, session_cfg):
    scores = score(con, session_cfg, "latest")
    last = con.execute("SELECT max(date) FROM features").fetchone()[0]
    assert (pd.to_datetime(scores["score_date"]) == pd.Timestamp(last)).all()


def test_build_sequences_pads_and_never_crosses_drives():
    values = np.arange(5, dtype=float).reshape(-1, 1) + 1  # 1..5
    serials = np.array(["A", "A", "A", "B", "B"])
    target = np.array([False, True, True, True, True])
    seqs = build_sequences(values, serials, target, seq_len=3)[..., 0]
    assert seqs.tolist() == [[0, 1, 2], [1, 2, 3], [0, 0, 4], [0, 4, 5]]


def test_pr_metrics():
    perfect = pr_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert perfect["average_precision"] == 1.0
    assert np.isnan(pr_metrics(np.zeros(3), np.ones(3))["average_precision"])
