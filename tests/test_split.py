import duckdb
import numpy as np
import pandas as pd

from failsight.data.split import LABEL_COLUMN, build_split, load_split
from failsight.models.tuning import time_folds


def _split_frames(con):
    return load_split(con, "train"), load_split(con, "test")


def test_no_drive_appears_in_both_splits(con):
    train, test = _split_frames(con)
    assert len(train) and len(test)
    assert not set(train["serial_number"]) & set(test["serial_number"])


def test_training_labels_end_before_test_period(con, cfg):
    train, test = _split_frames(con)
    horizon = cfg["data"]["label_horizon_days"]
    test_start = pd.Timestamp(cfg["split"]["test_start"])
    last_label_day = train["date"].max() + pd.Timedelta(days=horizon - 1)
    assert last_label_day < test_start <= test["date"].min()


def test_test_labels_are_fully_observed(con, cfg):
    _, test = _split_frames(con)
    horizon = cfg["data"]["label_horizon_days"]
    last_day = con.execute("SELECT max(date) FROM features").fetchone()[0]
    assert (test["date"] + pd.Timedelta(days=horizon - 1)).max() <= pd.Timestamp(last_day)


def _copy_features(con):
    other = duckdb.connect()
    other.register("features_df", con.execute("SELECT * FROM features").fetch_arrow_table())
    other.execute("CREATE TABLE features AS SELECT * FROM features_df")
    return other


def test_negative_subsampling_keeps_every_positive(con, cfg):
    other = _copy_features(con)
    cfg["split"]["train_negative_fraction"] = 0.2
    build_split(other, cfg)
    sub, full = load_split(other, "train"), load_split(con, "train")
    assert sub[LABEL_COLUMN].sum() == full[LABEL_COLUMN].sum()
    assert len(sub) < len(full)


def test_time_folds_respect_time_and_drive_identity(con, cfg):
    train, _ = _split_frames(con)
    horizon = cfg["data"]["label_horizon_days"]
    folds = time_folds(train["date"], train["serial_number"], 2, horizon)
    assert len(folds) == 2
    for train_idx, val_idx in folds:
        assert len(train_idx) and len(val_idx)
        tr, va = train.iloc[train_idx], train.iloc[val_idx]
        assert not set(tr["serial_number"]) & set(va["serial_number"])
        gap = pd.Timedelta(days=horizon - 1)
        assert tr["date"].max() + gap < va["date"].min()


def test_split_is_deterministic(con, cfg):
    results = []
    for _ in range(2):
        other = _copy_features(con)
        build_split(other, cfg)
        results.append(load_split(other, "test")["serial_number"].unique())
    assert np.array_equal(results[0], results[1])
