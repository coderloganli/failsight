"""MLflow tracking: every training run logs its parameters and metrics."""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import mlflow


def _flatten(params: dict[str, Any], prefix: str = "") -> dict[str, str]:
    flat: dict[str, str] = {}
    for key, value in params.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{name}."))
        else:
            flat[name] = str(value)[:500]
    return flat


def setup(cfg: dict) -> None:
    mlflow.set_tracking_uri(cfg["tracking"]["tracking_uri"])
    mlflow.set_experiment(cfg["tracking"]["experiment"])


@contextmanager
def tracked_run(cfg: dict, run_name: str, params: dict[str, Any]) -> Iterator[str]:
    """Start an MLflow run, log ``params`` and yield the run id."""
    setup(cfg)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(_flatten(params))
        yield run.info.run_id


def log_params(params: dict[str, Any], prefix: str = "") -> None:
    if params:
        mlflow.log_params(_flatten(params, prefix))


def log_metrics(metrics: dict[str, float], prefix: str = "") -> None:
    clean = {
        f"{prefix}{k}": float(v)
        for k, v in metrics.items()
        if v is not None and not math.isnan(float(v))
    }
    if clean:
        mlflow.log_metrics(clean)
