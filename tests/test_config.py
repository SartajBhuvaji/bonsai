from pathlib import Path

import pytest

from bonsai.config import load_config


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(text)
    return path


def test_loads_model_and_data(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        """
        [model]
        repo_id = "Qwen/Qwen3-30B-A3B-Base"

        [data]
        dataset_id = "bigcode/the-stack-dedup"
        data_dir = "data/typescript"
        """,
    )

    config = load_config(path)

    assert config.name == "config"
    assert config.model.repo_id == "Qwen/Qwen3-30B-A3B-Base"
    assert config.model.revision == "main"
    assert config.data.dataset_id == "bigcode/the-stack-dedup"
    assert config.data.calibration_samples == 1024
    assert config.compute.provider == "local"


def test_explicit_name_and_compute(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        """
        name = "qwen3-30b-a3b"

        [model]
        repo_id = "Qwen/Qwen3-30B-A3B-Base"
        revision = "abc123"

        [data]
        dataset_id = "bigcode/the-stack-dedup"
        data_dir = "data/typescript"
        calibration_samples = 512
        calibration_shards = ["data-00000-of-00071.parquet"]

        [compute]
        provider = "lambda"
        instance_type = "gpu_1x_a100"
        """,
    )

    config = load_config(path)

    assert config.name == "qwen3-30b-a3b"
    assert config.model.revision == "abc123"
    assert config.data.calibration_samples == 512
    assert config.data.calibration_shards == ("data-00000-of-00071.parquet",)
    assert config.compute.provider == "lambda"
    assert config.compute.instance_type == "gpu_1x_a100"


def test_unknown_top_level_keys_kept_in_extra(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        """
        [model]
        repo_id = "Qwen/Qwen3-30B-A3B-Base"

        [data]
        dataset_id = "bigcode/the-stack-dedup"
        data_dir = "data/typescript"

        [distillation]
        temperature = 1.0
        """,
    )

    config = load_config(path)

    assert config.extra == {"distillation": {"temperature": 1.0}}


def test_missing_model_table_raises(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        """
        [data]
        dataset_id = "bigcode/the-stack-dedup"
        data_dir = "data/typescript"
        """,
    )

    with pytest.raises(KeyError):
        load_config(path)
