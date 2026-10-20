#!/usr/bin/env python3
"""Export HumanEval-TS prompts for vLLM / SGLang serving benchmarks."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from bonsai.eval import TsMultiPLE, register

OUTPUT_TOKENS = 512


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs/debug"),
        help="Directory for JSONL exports (default: docs/debug)",
    )
    parser.add_argument(
        "--output-tokens",
        type=int,
        default=OUTPUT_TOKENS,
        help="Fixed generation cap per request (default: 512)",
    )
    args = parser.parse_args()

    register()
    task = TsMultiPLE()
    dataset = task.get_dataset()

    vllm_path = args.out_dir / "phase4_inference_benchmark_humaneval_ts_vllm.jsonl"
    sglang_path = args.out_dir / "phase4_inference_benchmark_humaneval_ts_sglang.jsonl"
    stats_path = args.out_dir / "phase4_inference_benchmark_humaneval_ts_prompt_stats.json"

    args.out_dir.mkdir(parents=True, exist_ok=True)

    char_lens: list[int] = []
    with vllm_path.open("w", encoding="utf-8") as vf, sglang_path.open("w", encoding="utf-8") as sf:
        for idx in range(len(dataset)):
            prompt = task.get_prompt(dataset[idx])
            char_lens.append(len(prompt))
            vf.write(
                json.dumps(
                    {"prompt": prompt, "output_tokens": args.output_tokens},
                    ensure_ascii=False,
                )
                + "\n"
            )
            sf.write(
                json.dumps(
                    {
                        "conversations": [
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": "// placeholder"},
                        ]
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    stats = {
        "n_prompts": len(char_lens),
        "output_tokens_per_request": args.output_tokens,
        "prompt_char_len_mean": statistics.mean(char_lens),
        "prompt_char_len_median": statistics.median(char_lens),
        "prompt_char_len_min": min(char_lens),
        "prompt_char_len_max": max(char_lens),
        "vllm_dataset_path": str(vllm_path),
        "sglang_dataset_path": str(sglang_path),
        "note": (
            "SGLang custom loader requires two-turn conversations; "
            "output length is set via --sharegpt-output-len at bench time."
        ),
    }
    stats_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
