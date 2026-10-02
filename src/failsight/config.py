"""Configuration loading."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(
    path: str | Path | None = None, overrides: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Load a YAML config, layered over the packaged default, then apply overrides."""
    with open(DEFAULT_CONFIG, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if path is not None and Path(path).resolve() != DEFAULT_CONFIG:
        with open(path, encoding="utf-8") as fh:
            cfg = _merge(cfg, yaml.safe_load(fh) or {})
    if overrides:
        cfg = _merge(cfg, overrides)
    return cfg


def smart_columns(cfg: dict[str, Any]) -> list[str]:
    """Names of the SMART columns used downstream, in a stable order."""
    data = cfg["data"]
    cols = [f"smart_{n}_raw" for n in data["raw_attributes"]]
    cols += [f"smart_{n}_normalized" for n in data["normalized_attributes"]]
    return cols
