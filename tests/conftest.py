from __future__ import annotations

import copy
import os

import pytest

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from failsight.config import load_config  # noqa: E402
from failsight.data import build_dataset, connect  # noqa: E402

from .synthetic import generate_frame, write_daily_csvs  # noqa: E402

TEST_OVERRIDES = {
    "data": {"label_horizon_days": 10},
    "split": {
        "test_start": "2026-03-21",
        "test_drive_fraction": 0.3,
        "train_negative_fraction": 1.0,
    },
    "features": {"windows_days": [3, 7]},
    "supervised": {"tuning": {"n_trials": 2, "n_splits": 2}},
    "lstm": {"seq_len": 7, "hidden_size": 8, "epochs": 1, "batch_size": 256},
    "anomaly": {
        "min_failures": 5,
        "pca_components": 3,
        "isolation_forest": {"n_estimators": 20},
        "autoencoder": {"hidden_size": 2, "epochs": 2, "batch_size": 256},
    },
}


@pytest.fixture(scope="session")
def frame():
    return generate_frame(seed=0)


@pytest.fixture(scope="session")
def raw_dir(tmp_path_factory, frame):
    path = tmp_path_factory.mktemp("raw")
    write_daily_csvs(frame, path)
    return path


@pytest.fixture(scope="session")
def session_cfg(tmp_path_factory):
    root = tmp_path_factory.mktemp("work")
    overrides = copy.deepcopy(TEST_OVERRIDES)
    overrides["paths"] = {
        "database": str(root / "failsight.duckdb"),
        "artifacts_dir": str(root / "artifacts"),
        "scores_path": str(root / "scores" / "failure_scores.parquet"),
    }
    overrides["tracking"] = {
        "tracking_uri": "sqlite:///" + (root / "mlflow.db").as_posix(),
        "experiment": "failsight-test",
    }
    return load_config(overrides=overrides)


@pytest.fixture
def cfg(session_cfg):
    return copy.deepcopy(session_cfg)


@pytest.fixture(scope="session")
def con(session_cfg, raw_dir):
    connection = connect(session_cfg["paths"]["database"])
    build_dataset(connection, session_cfg, raw_dir)
    yield connection
    connection.close()


@pytest.fixture(scope="session")
def trained(con, session_cfg):
    from failsight.train import train_all

    return train_all(con, session_cfg, "all")
