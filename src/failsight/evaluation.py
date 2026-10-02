"""Precision-recall evaluation for heavily imbalanced failure labels.

Accuracy is not reported: with failures a tiny fraction of drive-days, a model
that never predicts failure is almost perfectly accurate and useless.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve


def pr_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Row-level precision-recall summary. Metrics are NaN when there are no positives."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    out = {"n_rows": float(len(y_true)), "n_positive": float(y_true.sum())}
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        out.update(average_precision=np.nan, best_f1=np.nan, best_f1_threshold=np.nan)
        return out
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    best = int(np.argmax(f1[:-1])) if len(thresholds) else 0
    out.update(
        average_precision=float(average_precision_score(y_true, scores)),
        best_f1=float(f1[best]),
        best_f1_threshold=float(thresholds[best]) if len(thresholds) else np.nan,
    )
    return out


def drive_level_metrics(
    serials: pd.Series, y_true: np.ndarray, scores: np.ndarray
) -> dict[str, float]:
    """Drive-level view: a drive's score is its maximum daily score in the period,
    and it is positive if any of its rows is positive."""
    frame = pd.DataFrame({"serial": np.asarray(serials), "y": y_true, "s": scores})
    per_drive = frame.groupby("serial").agg(y=("y", "max"), s=("s", "max"))
    metrics = pr_metrics(per_drive["y"].to_numpy(), per_drive["s"].to_numpy())
    return {f"drive_{k}": v for k, v in metrics.items()}


def evaluate(serials: pd.Series, y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {**pr_metrics(y_true, scores), **drive_level_metrics(serials, y_true, scores)}
