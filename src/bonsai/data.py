"""Pull, split, and tokenize the TypeScript calibration/retraining data.

Both Minitron papers draw their calibration set from the same blend used for
retraining (Section 2.2 / "Dataset" in 2408.11796); this project draws both
from `bigcode/the-stack-dedup/data/typescript`. Calibration and retraining
shards are disjoint by construction -- whole parquet shards are reserved for
calibration, so nothing downstream needs row-level bookkeeping to avoid
leaking a retraining example into the calibration set.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

import datasets
from huggingface_hub import HfApi

from bonsai.config import DataConfig

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


def _filter_shards(files: list[str], data_dir: str) -> tuple[str, ...]:
    prefix = data_dir.rstrip("/") + "/"
    return tuple(sorted(f for f in files if f.startswith(prefix) and f.endswith(".parquet")))


def list_shards(dataset_id: str, data_dir: str, *, token: str | None = None) -> tuple[str, ...]:
    """List parquet shard paths under `data_dir` in a Hub dataset repo, sorted."""
    files = HfApi(token=token).list_repo_files(dataset_id, repo_type="dataset")
    return _filter_shards(files, data_dir)


def make_split(base: DataConfig, shards: tuple[str, ...], *, calibration_shard_count: int = 1) -> DataConfig:
    """Reserve the first `calibration_shard_count` shards for calibration, the rest for retraining.

    Shard-level partitioning, not a row-level random split: two shards never
    share a row, so calibration and retraining are disjoint without tracking
    individual examples. Returns a new `DataConfig` with `base`'s
    `calibration_shards`/`retraining_shards` filled in.
    """
    if not 0 < calibration_shard_count < len(shards):
        raise ValueError(
            f"calibration_shard_count must leave at least one retraining shard "
            f"(got {calibration_shard_count} of {len(shards)} shards)"
        )
    return replace(
        base,
        calibration_shards=shards[:calibration_shard_count],
        retraining_shards=shards[calibration_shard_count:],
    )


def _hub_urls(dataset_id: str, shards: tuple[str, ...]) -> list[str]:
    return [f"hf://datasets/{dataset_id}/{shard}" for shard in shards]


def _load_shards(
    data_files: list[str], n: int, *, seed: int = 0, token: str | None = None
) -> list[dict[str, Any]]:
    """Stream `n` shuffled rows out of `data_files`.

    Streams rather than downloading whole files: `n` is always far smaller
    than a shard (the papers' own calibration set is 1024 rows out of tens of
    thousands per shard), so only as many rows as the shuffle buffer and `n`
    actually need get fetched. Local paths work the same way as `hf://` URLs,
    which is what lets this run offline in tests.
    """
    ds = datasets.load_dataset(
        "parquet", data_files={"train": data_files}, split="train", streaming=True, token=token
    )
    ds = ds.shuffle(seed=seed, buffer_size=max(n * 4, 1000))
    return [row for row, _ in zip(ds, range(n), strict=False)]


def load_samples(
    data: DataConfig,
    pool: Literal["calibration", "retraining"],
    n: int,
    *,
    shard_limit: int | None = None,
    seed: int = 0,
    token: str | None = None,
) -> list[dict[str, Any]]:
    """Stream `n` shuffled rows from `data`'s calibration or retraining shards.

    `shard_limit` caps how many of the pool's shards get opened, taken from
    the front. Opening a shard has a fixed per-file resolution cost before
    any row comes back, so a 70-shard retraining pool is expensive to even
    peek at -- a real retraining run wants every shard and omits this, but a
    quick check (is this pool sane, does it look disjoint from calibration)
    doesn't need all 70 to answer that.
    """
    shards = data.calibration_shards if pool == "calibration" else data.retraining_shards
    if not shards:
        raise ValueError(f"{pool} has no shards -- call make_split first")
    if shard_limit is not None:
        shards = shards[:shard_limit]
    return _load_shards(_hub_urls(data.dataset_id, shards), n, seed=seed, token=token)


@dataclass(frozen=True, slots=True)
class TokenizedSamples:
    """Token-length stats for a batch of samples, plus the encoded batch itself."""

    input_ids: tuple[tuple[int, ...], ...]

    @property
    def lengths(self) -> tuple[int, ...]:
        return tuple(len(ids) for ids in self.input_ids)

    @property
    def mean_length(self) -> float:
        lengths = self.lengths
        return sum(lengths) / len(lengths)


def tokenize_samples(
    samples: list[dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_length: int = 2048,
    text_field: str = "content",
) -> TokenizedSamples:
    """Tokenize `samples[text_field]`, truncated to `max_length`.

    `max_length` is a pipeline default, not a claim about the right
    calibration sequence length -- that's Phase 2's call once importance
    estimation is actually running against these tokens.
    """
    texts = [s[text_field] for s in samples]
    encoded = tokenizer(texts, truncation=True, max_length=max_length)
    input_ids = tuple(tuple(ids) for ids in encoded["input_ids"])
    return TokenizedSamples(input_ids=input_ids)


def verify_disjoint(
    calibration: list[dict[str, Any]], retraining: list[dict[str, Any]], *, key: str = "hexsha"
) -> None:
    """Raise if any sample appears in both pools, keyed by `key`.

    `hexsha` is the blob hash the-stack-dedup assigns each file -- an exact
    duplicate would share it, which is exactly the leak this checks for. The
    shard-level split already makes this structurally impossible; this is
    the empirical confirmation, not a substitute for it.
    """
    calibration_keys = {s[key] for s in calibration}
    retraining_keys = {s[key] for s in retraining}
    overlap = calibration_keys & retraining_keys
    if overlap:
        raise ValueError(
            f"{len(overlap)} sample(s) present in both calibration and retraining: {overlap}"
        )


__all__ = [
    "TokenizedSamples",
    "list_shards",
    "load_samples",
    "make_split",
    "tokenize_samples",
    "verify_disjoint",
]
