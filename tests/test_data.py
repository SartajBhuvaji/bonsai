"""Tests against tiny local parquet shards.

Not `bigcode/the-stack-dedup` itself -- that's a 15GB gated Hugging Face
dataset, a bad fit for a CI run. Writing a couple of rows to a local parquet
file and pointing `datasets.load_dataset` at it exercises the exact same
streaming/shuffle/tokenize code path, offline and in milliseconds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import datasets
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from bonsai.config import DataConfig
from bonsai.data import (
    _filter_shards,
    _hub_urls,
    _load_shards,
    load_samples,
    make_split,
    tokenize_samples,
    verify_disjoint,
)


def write_shard(tmp_path: Path, rows: list[dict[str, Any]], name: str = "shard.parquet") -> str:
    path = tmp_path / name
    datasets.Dataset.from_list(rows).to_parquet(str(path))
    return str(path)


def row(hexsha: str, content: str) -> dict[str, Any]:
    return {"hexsha": hexsha, "content": content}


def base_config() -> DataConfig:
    return DataConfig(dataset_id="bigcode/the-stack-dedup", data_dir="data/typescript")


def tiny_tokenizer() -> PreTrainedTokenizerFast:
    vocab = {word: i for i, word in enumerate(["[UNK]", "const", "x", "=", "1", ";", "function", "(", ")"])}
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]")


# --- shard filtering ---------------------------------------------------


def test_filter_shards_keeps_only_matching_prefix_and_suffix() -> None:
    files = [
        "data/typescript/data-00000-of-00002.parquet",
        "data/typescript/data-00001-of-00002.parquet",
        "data/typescript/README.md",
        "data/python/data-00000-of-00001.parquet",
    ]

    shards = _filter_shards(files, "data/typescript")

    assert shards == (
        "data/typescript/data-00000-of-00002.parquet",
        "data/typescript/data-00001-of-00002.parquet",
    )


def test_hub_urls_build_hf_scheme_paths() -> None:
    urls = _hub_urls("bigcode/the-stack-dedup", ("data/typescript/data-00000-of-00071.parquet",))

    assert urls == ["hf://datasets/bigcode/the-stack-dedup/data/typescript/data-00000-of-00071.parquet"]


# --- split ---------------------------------------------------------------


def test_make_split_reserves_first_n_shards_for_calibration() -> None:
    shards = tuple(f"data-{i:05d}-of-00005.parquet" for i in range(5))

    split = make_split(base_config(), shards, calibration_shard_count=2)

    assert split.calibration_shards == shards[:2]
    assert split.retraining_shards == shards[2:]
    assert not set(split.calibration_shards) & set(split.retraining_shards)


def test_make_split_rejects_taking_every_shard() -> None:
    shards = ("a.parquet", "b.parquet")

    with pytest.raises(ValueError, match="retraining shard"):
        make_split(base_config(), shards, calibration_shard_count=2)


def test_make_split_rejects_zero_calibration_shards() -> None:
    shards = ("a.parquet", "b.parquet")

    with pytest.raises(ValueError, match="retraining shard"):
        make_split(base_config(), shards, calibration_shard_count=0)


def test_make_split_preserves_other_fields() -> None:
    split = make_split(base_config(), ("a.parquet", "b.parquet"))

    assert split.dataset_id == "bigcode/the-stack-dedup"
    assert split.calibration_samples == 1024  # DataConfig's default, untouched


# --- loading ---------------------------------------------------------------


def test_load_shards_reads_local_files(tmp_path: Path) -> None:
    rows = [row(f"h{i}", f"const x{i} = {i};") for i in range(10)]
    path = write_shard(tmp_path, rows)

    loaded = _load_shards([path], n=5, seed=0)

    assert len(loaded) == 5
    assert {r["hexsha"] for r in loaded} <= {r["hexsha"] for r in rows}


def test_load_samples_raises_for_empty_pool() -> None:
    with pytest.raises(ValueError, match="no shards"):
        load_samples(base_config(), "calibration", n=1)


def test_load_samples_shard_limit_caps_which_shards_open(monkeypatch: pytest.MonkeyPatch) -> None:
    # `load_samples` always resolves shards to `hf://` URLs, so this seam --
    # not a local-file fixture -- is what lets the slicing itself be checked
    # without a real Hub dataset behind it.
    captured: list[str] = []

    def fake_load_shards(
        data_files: list[str], n: int, *, seed: int = 0, token: str | None = None
    ) -> list[dict[str, Any]]:
        captured.extend(data_files)
        return []

    monkeypatch.setattr("bonsai.data._load_shards", fake_load_shards)
    data = DataConfig(
        dataset_id="bigcode/the-stack-dedup",
        data_dir="data/typescript",
        retraining_shards=(
            "data/typescript/data-00001-of-00071.parquet",
            "data/typescript/data-00002-of-00071.parquet",
        ),
    )

    load_samples(data, "retraining", n=5, shard_limit=1)

    assert captured == ["hf://datasets/bigcode/the-stack-dedup/data/typescript/data-00001-of-00071.parquet"]


# --- tokenization ------------------------------------------------------


def test_tokenize_samples_reports_lengths_and_mean() -> None:
    samples = [row("h0", "const x = 1;"), row("h1", "function ( ) x")]

    tokenized = tokenize_samples(samples, tiny_tokenizer())

    assert len(tokenized.input_ids) == 2
    assert tokenized.lengths == tuple(len(ids) for ids in tokenized.input_ids)
    assert tokenized.mean_length == sum(tokenized.lengths) / 2


def test_tokenize_samples_truncates_to_max_length() -> None:
    long_sample = [row("h0", " ".join(["const"] * 20))]

    tokenized = tokenize_samples(long_sample, tiny_tokenizer(), max_length=5)

    assert tokenized.lengths[0] <= 5


# --- disjointness --------------------------------------------------------


def test_verify_disjoint_passes_for_non_overlapping_keys() -> None:
    calibration = [row("cal-0", "a"), row("cal-1", "b")]
    retraining = [row("re-0", "c")]

    verify_disjoint(calibration, retraining)  # does not raise


def test_verify_disjoint_raises_on_shared_key() -> None:
    calibration = [row("dup", "a")]
    retraining = [row("dup", "a")]

    with pytest.raises(ValueError, match="both calibration and retraining"):
        verify_disjoint(calibration, retraining)
