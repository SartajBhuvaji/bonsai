"""Experiment configuration.

Everything a step needs to know (which model, which data, which compute
target) is loaded from a TOML file rather than hardcoded, so a notebook and a
training script for the same step read the exact same settings. `tomllib` is
stdlib as of the Python version this project requires, so this needs no extra
dependency.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Which pretrained model a step operates on."""

    repo_id: str
    revision: str = "main"


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Where the calibration and retraining data come from.

    `calibration_shards` and `retraining_shards` must be disjoint -- nothing
    here enforces that at load time, but the data pipeline that consumes this
    config asserts it before use.
    """

    dataset_id: str
    data_dir: str
    calibration_shards: tuple[str, ...] = ()
    retraining_shards: tuple[str, ...] = ()
    calibration_samples: int = 1024


@dataclass(frozen=True, slots=True)
class ComputeConfig:
    """Where a step actually runs."""

    provider: str = "local"
    instance_type: str | None = None


@dataclass(frozen=True, slots=True)
class Config:
    """The full experiment configuration for one step."""

    name: str
    model: ModelConfig
    data: DataConfig
    compute: ComputeConfig
    extra: dict[str, Any] = field(default_factory=dict)


def _model_config(raw: dict[str, Any]) -> ModelConfig:
    return ModelConfig(repo_id=raw["repo_id"], revision=raw.get("revision", "main"))


def _data_config(raw: dict[str, Any]) -> DataConfig:
    return DataConfig(
        dataset_id=raw["dataset_id"],
        data_dir=raw["data_dir"],
        calibration_shards=tuple(raw.get("calibration_shards", ())),
        retraining_shards=tuple(raw.get("retraining_shards", ())),
        calibration_samples=raw.get("calibration_samples", 1024),
    )


def _compute_config(raw: dict[str, Any]) -> ComputeConfig:
    return ComputeConfig(
        provider=raw.get("provider", "local"),
        instance_type=raw.get("instance_type"),
    )


def load_config(path: str | Path) -> Config:
    """Load a step's configuration from a TOML file.

    Expects top-level `[model]`, `[data]` and (optionally) `[compute]` tables.
    Anything else in the file is kept as-is under `Config.extra` rather than
    rejected, so a step can carry settings this module doesn't know about yet
    without a change here.
    """
    path = Path(path)
    raw = tomllib.loads(path.read_text())

    known = {"name", "model", "data", "compute"}
    return Config(
        name=raw.get("name", path.stem),
        model=_model_config(raw["model"]),
        data=_data_config(raw["data"]),
        compute=_compute_config(raw.get("compute", {})),
        extra={k: v for k, v in raw.items() if k not in known},
    )
