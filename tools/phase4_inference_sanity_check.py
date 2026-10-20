#!/usr/bin/env python3
"""Single-prompt load/generate sanity check for pruned MoE checkpoints."""

from __future__ import annotations

import argparse
import time


def check_vllm(model: str, gpu_memory_utilization: float) -> None:
    from vllm import LLM, SamplingParams

    print(f"[vLLM] loading {model} ...", flush=True)
    t0 = time.perf_counter()
    import os

    max_model_len = int(os.environ.get("VLLM_MAX_MODEL_LEN", "8192"))
    llm = LLM(
        model=model,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    print(f"[vLLM] loaded in {time.perf_counter() - t0:.1f}s", flush=True)
    prompt = "function add(a: number, b: number): number {\n"
    out = llm.generate([prompt], SamplingParams(max_tokens=32, temperature=0.0))[0]
    text = out.outputs[0].text
    print(f"[vLLM] ok, sample output ({len(text)} chars): {text[:120]!r}", flush=True)


def check_sglang(model: str, mem_fraction_static: float) -> None:
    from sglang.srt.entrypoints.engine import Engine

    print(f"[SGLang] loading {model} ...", flush=True)
    t0 = time.perf_counter()
    import os

    context_length = int(os.environ.get("VLLM_MAX_MODEL_LEN", "8192"))
    engine = Engine(
        model_path=model,
        trust_remote_code=True,
        context_length=context_length,
        mem_fraction_static=mem_fraction_static,
    )
    print(f"[SGLang] loaded in {time.perf_counter() - t0:.1f}s", flush=True)
    prompt = "function add(a: number, b: number): number {\n"
    out = engine.generate(
        prompt,
        sampling_params={"max_new_tokens": 32, "temperature": 0.0},
    )
    if isinstance(out, dict):
        text = out.get("text", "")
    elif isinstance(out, list) and out and isinstance(out[0], dict):
        text = out[0].get("text", "")
    else:
        text = str(out)
    print(f"[SGLang] ok, sample output ({len(text)} chars): {text[:120]!r}", flush=True)
    engine.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--engine",
        choices=("vllm", "sglang", "both"),
        default="both",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="vLLM gpu_memory_utilization / SGLang mem_fraction_static",
    )
    args = parser.parse_args()

    try:
        if args.engine in ("vllm", "both"):
            check_vllm(args.model, args.gpu_memory_utilization)
        if args.engine in ("sglang", "both"):
            check_sglang(args.model, args.gpu_memory_utilization)
    except Exception as exc:
        print(f"FAILED: {exc!r}", flush=True)
        raise


if __name__ == "__main__":
    main()
