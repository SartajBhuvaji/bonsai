#!/usr/bin/env python3
"""Merge per-run benchmark JSON into phase4_inference_benchmark_summary.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

CANDIDATE_PARAMS_B: dict[str, float] = {
    "baseline": 30.5e9,
    "depth0_experts64": 16.03e9,
    "depth16_experts0": 20.56e9,
    "depth0_experts8": 28.72e9,
}


def _mean_output_tokens(data: dict[str, Any]) -> float | None:
    completed = data.get("completed", data.get("successful_requests"))
    total_out = data.get("total_output_tokens")
    if completed and total_out:
        return float(total_out) / float(completed)
    return None


def load_result_document(path: Path) -> dict[str, Any]:
    """vLLM bench writes one JSON object; SGLang --output-details appends JSONL."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"empty result file: {path}")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        for line in text.splitlines():
            line = line.strip()
            if line:
                return json.loads(line)
        raise


def _get(d: dict[str, Any], *keys: str, default: object | None = None) -> object | None:
    for key in keys:
        if key in d:
            return d[key]
    return default


def parse_vllm_result(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "throughput_tok_s": _get(data, "output_throughput", "total_token_throughput"),
        "throughput_req_s": _get(data, "request_throughput"),
        "ttft_ms_mean": _get(data, "mean_ttft_ms", "ttft_mean"),
        "ttft_ms_p50": _get(data, "median_ttft_ms", "p50_ttft_ms"),
        "ttft_ms_p90": _get(data, "p90_ttft_ms"),
        "ttft_ms_p99": _get(data, "p99_ttft_ms"),
        "e2e_latency_ms_p50": _get(
            data, "median_e2el_ms", "median_e2e_latency_ms", "p50_e2e_latency_ms"
        ),
        "e2e_latency_ms_p90": _get(data, "p90_e2el_ms", "p90_e2e_latency_ms"),
        "e2e_latency_ms_p99": _get(data, "p99_e2el_ms", "p99_e2e_latency_ms"),
        "itl_ms_mean": _get(data, "mean_itl_ms", "mean_tpot_ms"),
        "tpot_ms_mean": _get(data, "mean_tpot_ms", "mean_itl_ms"),
        "completed_requests": _get(data, "completed", "successful_requests"),
        "mean_output_tokens_per_request": _mean_output_tokens(data),
    }


def parse_sglang_result(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "throughput_tok_s": _get(data, "output_throughput", "token_throughput"),
        "throughput_req_s": _get(data, "request_throughput"),
        "ttft_ms_mean": _get(data, "mean_ttft_ms"),
        "ttft_ms_p50": _get(data, "median_ttft_ms"),
        "ttft_ms_p90": _get(data, "p90_ttft_ms"),
        "ttft_ms_p99": _get(data, "p99_ttft_ms"),
        "e2e_latency_ms_p50": _get(data, "median_e2e_latency_ms"),
        "e2e_latency_ms_p90": _get(data, "p90_e2e_latency_ms"),
        "e2e_latency_ms_p99": _get(data, "p99_e2e_latency_ms"),
        "itl_ms_mean": _get(data, "mean_itl_ms"),
        "tpot_ms_mean": _get(data, "mean_tpot_ms", "mean_itl_ms"),
        "completed_requests": _get(data, "completed"),
        "mean_output_tokens_per_request": _mean_output_tokens(data),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="JSON list of run metadata (candidate, engine, concurrency, paths)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/debug/phase4_inference_benchmark_summary.json"),
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    rows: list[dict[str, Any]] = []

    manifest_dir = args.manifest.resolve().parent

    def resolve_result_path(raw: str) -> Path:
        p = Path(raw)
        if p.is_file():
            return p
        by_name = manifest_dir / p.name
        if by_name.is_file():
            return by_name
        return p

    for entry in manifest:
        result_path = resolve_result_path(entry["result_json"])
        if not result_path.is_file():
            print(f"skip missing result: {result_path}")
            continue
        raw = load_result_document(result_path)
        completed = raw.get("completed", raw.get("successful_requests"))
        if completed == 0:
            print(f"skip zero completions: {result_path}")
            continue
        engine = entry["engine"]
        metrics = parse_vllm_result(raw) if engine == "vllm" else parse_sglang_result(raw)
        slug = entry["candidate"]
        rows.append(
            {
                "candidate": slug,
                "model_id": entry["model_id"],
                "params_b": CANDIDATE_PARAMS_B.get(slug),
                "engine": engine,
                "concurrency": entry["concurrency"],
                "gpu_memory_utilization": entry.get("gpu_memory_utilization"),
                "num_prompts": entry.get("num_prompts", 159),
                "output_tokens": entry.get("output_tokens", 512),
                "raw_result_path": str(result_path),
                **metrics,
            }
        )

    meta = {
        "vllm_version": manifest[0].get("vllm_version") if manifest else None,
        "sglang_version": manifest[0].get("sglang_version") if manifest else None,
        "gpu": manifest[0].get("gpu") if manifest else None,
        "prompt_stats_path": manifest[0].get("prompt_stats_path") if manifest else None,
        "fixed_output_tokens": 512,
        "ignore_eos": True,
        "methodology_note": (
            "2026-10-20 B300 HumanEval-TS serving (159 prompts, 512 output tokens, "
            "concurrency 1/4/16/32). SGLang: --sharegpt-output-len 512 (ignore EOS by "
            "default). vLLM: --custom-output-len 512 --ignore-eos after a VLLM_ONLY=1 "
            "re-sweep; earlier committed vLLM raw JSONs were EOS-shortened while summary "
            "incorrectly reported 512 until that re-sweep (raw total_output_tokens/159=512)."
        ),
    }
    out_doc = {"meta": meta, "rows": rows}
    args.out.write_text(json.dumps(out_doc, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
