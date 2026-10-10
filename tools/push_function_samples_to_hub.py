#!/usr/bin/env python3
"""Upload function-level KD training samples to a Hugging Face dataset repo.

Matches `cache_teacher_logits.py --sample-mode function` (same Stack split,
seed, and `load_function_samples` budget). Use `--from-cache-pt` after a
cache run to export the exact tokenized rows (decoded with the teacher
tokenizer) without re-streaming raw files.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from datasets import Dataset
from huggingface_hub import HfApi
from transformers import AutoTokenizer

from bonsai.config import DataConfig
from bonsai.data import list_shards, make_split
from bonsai.data.functions import load_function_samples

DATASET_ID = "bigcode/the-stack-dedup"
DATA_DIR = "data/typescript"
SEED = 11
BASELINE_TEACHER = "Qwen/Qwen3-30B-A3B-Base"


def _rows_from_cache(cache_path: Path, *, teacher_repo_id: str, hf_token: str | None) -> list[dict]:
    blob = torch.load(cache_path, map_location="cpu", weights_only=False)
    teacher = blob.get("teacher_repo_id", teacher_repo_id)
    tokenizer = AutoTokenizer.from_pretrained(teacher, token=hf_token)
    rows: list[dict] = []
    for i, sample in enumerate(blob["samples"]):
        ids = sample["input_ids"]
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        text = tokenizer.decode(ids, skip_special_tokens=False)
        rows.append(
            {
                "index": i,
                "content": text,
                "char_length": len(text),
                "token_length": len(ids),
            }
        )
    return rows


def _rows_from_stream(
    *,
    num_samples: int,
    seed: int,
    dataset_id: str,
    data_dir: str,
    hf_token: str | None,
) -> list[dict]:
    shards = list_shards(dataset_id, data_dir, token=hf_token)
    split = make_split(DataConfig(dataset_id=dataset_id, data_dir=data_dir), shards)
    raw = load_function_samples(split, "retraining", n=num_samples, seed=seed, token=hf_token)
    tokenizer = AutoTokenizer.from_pretrained(BASELINE_TEACHER, token=hf_token)
    rows: list[dict] = []
    for i, row in enumerate(raw):
        content = row["content"]
        ids = tokenizer.encode(content, add_special_tokens=False)
        rows.append(
            {
                "index": i,
                "content": content,
                "char_length": len(content),
                "token_length": len(ids),
            }
        )
    return rows


def _length_summary(rows: list[dict]) -> dict[str, float]:
    chars = [r["char_length"] for r in rows]
    toks = [r["token_length"] for r in rows]
    return {
        "n": float(len(rows)),
        "char_mean": statistics.mean(chars),
        "char_median": statistics.median(chars),
        "token_mean": statistics.mean(toks),
        "token_median": statistics.median(toks),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-samples", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument(
        "--from-cache-pt",
        type=Path,
        default=None,
        help="decode input_ids from a teacher top-k logits .pt (exact cache rows)",
    )
    parser.add_argument("--teacher-model", default=BASELINE_TEACHER)
    parser.add_argument("--push-to-hub", required=True, help="HF dataset repo id")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument(
        "--hub-public",
        action="store_true",
        help="create/update dataset repo as public",
    )
    parser.add_argument(
        "--write-jsonl",
        type=Path,
        default=None,
        help="optional local path to write samples.jsonl before upload",
    )
    args = parser.parse_args()

    t0 = time.monotonic()
    if args.from_cache_pt is not None:
        rows = _rows_from_cache(
            args.from_cache_pt, teacher_repo_id=args.teacher_model, hf_token=args.hf_token
        )
        source = f"cache:{args.from_cache_pt}"
    else:
        rows = _rows_from_stream(
            num_samples=args.num_samples,
            seed=args.seed,
            dataset_id=args.dataset_id,
            data_dir=args.data_dir,
            hf_token=args.hf_token,
        )
        source = "load_function_samples"

    summary = _length_summary(rows)
    elapsed = time.monotonic() - t0
    meta = {
        "source": source,
        "num_samples": len(rows),
        "seed": args.seed,
        "dataset_id": args.dataset_id,
        "data_dir": args.data_dir,
        "sample_mode": "function",
        "wall_clock_sec": round(elapsed, 2),
        **{f"{k}": v for k, v in summary.items()},
    }
    print(json.dumps(meta, indent=2))

    if args.write_jsonl is not None:
        args.write_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.write_jsonl.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        meta_path = args.write_jsonl.with_name("dataset_meta.json")
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.write_jsonl} ({len(rows)} rows)")

    private = not args.hub_public
    token = args.hf_token
    api = HfApi(token=token)
    api.create_repo(args.push_to_hub, repo_type="dataset", private=private, exist_ok=True)

    ds = Dataset.from_list(rows)
    ds.push_to_hub(args.push_to_hub, token=token, private=private)
    readme = (
        f"# {args.push_to_hub}\n\n"
        "Function-level TypeScript samples for Phase 4 KD (`bonsai.data.functions`), "
        f"seed={args.seed}, n={len(rows)}.\n\n"
        f"```json\n{json.dumps(meta, indent=2)}\n```\n"
    )
    api.upload_file(
        path_or_fileobj=readme.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.push_to_hub,
        repo_type="dataset",
    )
    if args.hub_public:
        api.update_repo_settings(args.push_to_hub, private=False, token=token, repo_type="dataset")
    vis = "public" if args.hub_public else "private"
    print(f"pushed dataset to https://huggingface.co/datasets/{args.push_to_hub} ({vis})")


if __name__ == "__main__":
    main()
