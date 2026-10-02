"""Feature-engineered supervised models in scikit-learn.

Class imbalance is handled with class weights on every family (on top of the
negative subsampling done in the data split).
"""

from __future__ import annotations

from typing import Any

import optuna
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FAMILIES = ("logistic_regression", "random_forest", "gradient_boosting")


def make_pipeline(family: str, params: dict[str, Any] | None = None, seed: int = 0) -> Pipeline:
    params = dict(params or {})
    imputer = ("impute", SimpleImputer(strategy="median", keep_empty_features=True))
    if family == "logistic_regression":
        model = LogisticRegression(class_weight="balanced", max_iter=2000, **params)
        return Pipeline([imputer, ("scale", StandardScaler()), ("model", model)])
    if family == "random_forest":
        model = RandomForestClassifier(
            class_weight="balanced_subsample", n_jobs=-1, random_state=seed, **params
        )
        return Pipeline([imputer, ("model", model)])
    if family == "gradient_boosting":
        model = HistGradientBoostingClassifier(class_weight="balanced", random_state=seed, **params)
        return Pipeline([imputer, ("model", model)])
    raise ValueError(f"Unknown supervised family: {family}")


def suggest_params(trial: optuna.Trial, family: str) -> dict[str, Any]:
    """Optuna search space per family."""
    if family == "logistic_regression":
        return {"C": trial.suggest_float("C", 1e-3, 1e2, log=True)}
    if family == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50),
            "max_features": trial.suggest_float("max_features", 0.1, 1.0),
        }
    if family == "gradient_boosting":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 8, 64),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 200),
            "l2_regularization": trial.suggest_float("l2_regularization", 0.0, 1.0),
            "max_iter": trial.suggest_int("max_iter", 50, 300),
        }
    raise ValueError(f"Unknown supervised family: {family}")
