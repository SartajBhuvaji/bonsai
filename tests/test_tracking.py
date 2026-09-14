from pathlib import Path

from bonsai.config import ComputeConfig, Config, DataConfig, ModelConfig
from bonsai.tracking import (
    load_config_snapshot,
    load_metrics,
    log_metric,
    new_run,
    save_config_snapshot,
)


def test_new_run_creates_checkpoints_dir(tmp_path: Path) -> None:
    run_dir = new_run("test-step", root=tmp_path)

    assert run_dir.is_dir()
    assert (run_dir / "checkpoints").is_dir()
    assert run_dir.name.endswith("-test-step")


def test_new_run_names_dont_collide(tmp_path: Path) -> None:
    first = new_run("test-step", root=tmp_path)
    second = new_run("test-step", root=tmp_path)

    assert first != second


def test_log_metric_appends_jsonl(tmp_path: Path) -> None:
    run_dir = new_run("test-step", root=tmp_path)

    log_metric(run_dir, step=0, loss=2.5)
    log_metric(run_dir, step=1, loss=2.1)

    metrics = load_metrics(run_dir)
    assert [m["step"] for m in metrics] == [0, 1]
    assert [m["loss"] for m in metrics] == [2.5, 2.1]
    assert all("time" in m for m in metrics)


def test_load_metrics_empty_before_any_logged(tmp_path: Path) -> None:
    run_dir = new_run("test-step", root=tmp_path)

    assert load_metrics(run_dir) == []


def test_save_config_snapshot_from_dataclass(tmp_path: Path) -> None:
    run_dir = new_run("test-step", root=tmp_path)
    config = Config(
        name="test-step",
        model=ModelConfig(repo_id="Qwen/Qwen3-30B-A3B-Base"),
        data=DataConfig(dataset_id="bigcode/the-stack-dedup", data_dir="data/typescript"),
        compute=ComputeConfig(),
    )

    save_config_snapshot(run_dir, config)

    snapshot = load_config_snapshot(run_dir)
    assert snapshot["name"] == "test-step"
    assert snapshot["model"]["repo_id"] == "Qwen/Qwen3-30B-A3B-Base"


def test_save_config_snapshot_from_plain_dict(tmp_path: Path) -> None:
    run_dir = new_run("test-step", root=tmp_path)

    save_config_snapshot(run_dir, {"ad_hoc": True})

    assert load_config_snapshot(run_dir) == {"ad_hoc": True}
