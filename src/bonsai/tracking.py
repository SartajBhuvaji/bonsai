"""Local experiment tracking.

No external service: every step writes to a plain directory under `runs/`,
one JSON-lines file for metrics and a copy of the config that produced them.
This is deliberately the simplest thing that works -- there are not yet
enough concurrent experiments to justify a dashboard, and a flat file is
trivial to `grep` or load with `pandas.read_json(lines=True)` from a
notebook. Revisit if the number of candidates in Phase 2/4 makes that
painful.
"""

from __future__ import annotations

import json
import secrets
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RUNS_DIR = Path("runs")


def new_run(name: str, *, root: Path = RUNS_DIR) -> Path:
    """Create a fresh run directory: `runs/<timestamp>-<suffix>-<name>/`.

    The timestamp keeps repeated runs of the same step sortable; the random
    suffix is what actually guarantees no collision. Two candidates from the
    same sweep starting in the same wall-clock second is the normal case once
    a script loops over several of them, not an edge case worth ignoring.
    """
    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    suffix = secrets.token_hex(3)
    run_dir = root / f"{stamp}-{suffix}-{name}"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=False)
    return run_dir


def save_config_snapshot(run_dir: Path, config: Any) -> None:  # noqa: ANN401
    """Copy the config that produced this run into the run directory.

    Takes either a dataclass (from `bonsai.config`) or a plain dict, so a
    notebook doing something ad hoc doesn't need to construct a `Config` just
    to record what it did -- genuinely either shape is accepted, not a case
    of an unannotated caller.
    """
    data = asdict(config) if is_dataclass(config) and not isinstance(config, type) else config
    (run_dir / "config.json").write_text(json.dumps(data, indent=2, default=str))


def log_metric(run_dir: Path, **fields: Any) -> None:  # noqa: ANN401
    """Append one line to `run_dir/metrics.jsonl`.

    Fields are genuinely arbitrary -- whatever the caller wants recorded for
    this step, as long as `json.dumps` can serialize it. A `time` field is
    added automatically if the caller doesn't supply one. Each call is a
    single JSON object; reading the file back is
    `[json.loads(line) for line in path.open()]` or
    `pandas.read_json(path, lines=True)`.
    """
    fields.setdefault("time", datetime.now(tz=UTC).isoformat())
    with (run_dir / "metrics.jsonl").open("a") as f:
        f.write(json.dumps(fields, default=str) + "\n")


def load_metrics(run_dir: Path) -> list[dict[str, Any]]:
    """Read back everything `log_metric` wrote for this run."""
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def load_config_snapshot(run_dir: Path) -> dict[str, Any]:
    """Read back what `save_config_snapshot` wrote for this run."""
    data: dict[str, Any] = json.loads((run_dir / "config.json").read_text())
    return data


__all__ = [
    "RUNS_DIR",
    "load_config_snapshot",
    "load_metrics",
    "log_metric",
    "new_run",
    "save_config_snapshot",
]
