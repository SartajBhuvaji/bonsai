"""Checkpoint paths for resumable bigcode-eval generation runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def intermediate_generations_path(save_generations_path: str | Path, task_name: str) -> Path:
    """Path bigcode-eval uses for periodic generation checkpoints.

    Matches ``bigcode_eval.evaluator.Evaluator.generate_text``:
    ``{stem(save_generations_path)}_{task_name}_intermediate.json``.
    """
    path = Path(save_generations_path)
    return path.parent / f"{path.stem}_{task_name}_intermediate.json"


def load_intermediate_generations(path: Path) -> list[Any] | None:
    """Load a checkpoint file if it exists; return ``None`` if missing."""
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        msg = f"expected a list in {path}, got {type(data).__name__}"
        raise TypeError(msg)
    return data


def count_completed_tasks(intermediate: list[Any]) -> int:
    """Number of dataset rows with at least one non-empty generation saved."""
    count = 0
    for entry in intermediate:
        if not entry:
            continue
        if (isinstance(entry, list) and any(entry)) or entry:
            count += 1
    return count
