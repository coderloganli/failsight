"""Optuna tuning under time-based cross-validation, and permutation-importance selection."""

from __future__ import annotations

import numpy as np
import optuna
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score

from failsight.models.common import drive_bucket
from failsight.models.supervised import make_pipeline, suggest_params


def time_folds(
    dates: pd.Series, serials: pd.Series, n_splits: int, horizon_days: int, seed: int = 0
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window folds that respect both time and drive identity.

    The date range is cut into ``n_splits + 1`` contiguous blocks. Fold ``k``
    validates on block ``k + 1`` for the drives in hash bucket ``k`` and trains on
    rows of the other drives whose label window ends before the validation block.
    """
    dates = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    serials = pd.Series(serials).reset_index(drop=True)
    buckets = drive_bucket(serials, n_splits, seed)
    edges = pd.date_range(dates.min(), dates.max() + pd.Timedelta(days=1), periods=n_splits + 2)
    gap = pd.Timedelta(days=horizon_days - 1)
    folds = []
    for k in range(n_splits):
        val_start, val_end = edges[k + 1], edges[k + 2]
        val = (dates >= val_start) & (dates < val_end) & (buckets == k)
        train = (dates + gap < val_start) & (buckets != k)
        folds.append((np.flatnonzero(train.to_numpy()), np.flatnonzero(val.to_numpy())))
    return folds


def cv_average_precision(
    family: str,
    params: dict,
    X: pd.DataFrame,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> float:
    """Mean validation average precision over folds where it is defined."""
    scores = []
    for train_idx, val_idx in folds:
        y_tr, y_val = y[train_idx], y[val_idx]
        if len(np.unique(y_tr)) < 2 or y_val.sum() == 0:
            continue
        model = make_pipeline(family, params, seed).fit(X.iloc[train_idx], y_tr)
        scores.append(average_precision_score(y_val, model.predict_proba(X.iloc[val_idx])[:, 1]))
    return float(np.mean(scores)) if scores else 0.0


def tune(
    family: str,
    X: pd.DataFrame,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    n_trials: int,
    seed: int,
) -> optuna.Study:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(
        lambda trial: cv_average_precision(
            family, suggest_params(trial, family), X, y, folds, seed
        ),
        n_trials=n_trials,
    )
    return study


def select_features(
    family: str,
    params: dict,
    X: pd.DataFrame,
    y: np.ndarray,
    fold: tuple[np.ndarray, np.ndarray],
    min_importance: float,
    seed: int,
) -> tuple[list[str], pd.Series]:
    """Permutation importance (drop in average precision) on a time-based holdout fold.

    Keeps features whose mean importance exceeds ``min_importance``; if none does,
    or the fold cannot be scored, every feature is kept.
    """
    train_idx, val_idx = fold
    columns = list(X.columns)
    if len(np.unique(y[train_idx])) < 2 or y[val_idx].sum() == 0:
        return columns, pd.Series(np.nan, index=columns)
    model = make_pipeline(family, params, seed).fit(X.iloc[train_idx], y[train_idx])
    result = permutation_importance(
        model,
        X.iloc[val_idx],
        y[val_idx],
        scoring="average_precision",
        n_repeats=5,
        random_state=seed,
        n_jobs=1,
    )
    importance = pd.Series(result.importances_mean, index=columns).sort_values(ascending=False)
    selected = importance[importance > min_importance].index.tolist()
    return (selected or columns), importance
