#!/usr/bin/env python3
"""One-request check: both engines emit exactly OUTPUT_TOKENS completion tokens."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

OUTPUT_TOKENS = 512
VLLM_PORT = 8000
SGLANG_PORT = 30000

MODELS: dict[str, str] = {
    "baseline": "Qwen/Qwen3-30B-A3B-Base",
    "depth0_experts64": (
        "sartajbhuvaji/bonsai-distilled-depth0-experts64-500-warmup75-lr1e4-fixedloss-best"
    ),
    "depth16_experts0": (
        "sartajbhuvaji/bonsai-distilled-depth16-experts0-500-warmup75-lr1e4-fixedloss-best"
    ),
    "depth0_experts8": (
        "sartajbhuvaji/bonsai-distilled-depth0-experts8-500-warmup75-lr1e4-fixedloss-best"
    ),
}


def _http_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read())


def _wait_health(url: str, timeout_sec: int = 900) -> None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=5)
            return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(3)
    raise RuntimeError(f"server not healthy: {url}")


def _load_prompt(debug_dir: Path) -> str:
    path = debug_dir / "phase4_inference_benchmark_humaneval_ts_vllm.jsonl"
    line = path.read_text(encoding="utf-8").splitlines()[0]
    return json.loads(line)["prompt"]


def check_vllm(model: str, prompt: str, gpu_mem: float, max_model_len: int) -> int:
    proc = subprocess.Popen(
        [
            "vllm",
            "serve",
            model,
            "--trust-remote-code",
            "--dtype",
            "bfloat16",
            "--max-model-len",
            str(max_model_len),
            "--gpu-memory-utilization",
            str(gpu_mem),
            "--port",
            str(VLLM_PORT),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_health(f"http://127.0.0.1:{VLLM_PORT}/health")
        body = _http_json(
            f"http://127.0.0.1:{VLLM_PORT}/v1/completions",
            {
                "model": model,
                "prompt": prompt,
                "max_tokens": OUTPUT_TOKENS,
                "temperature": 0,
                "ignore_eos": True,
            },
        )
        return int(body["usage"]["completion_tokens"])
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)


def check_sglang(model: str, prompt: str, gpu_mem: float, max_model_len: int) -> int:
    proc = subprocess.Popen(
        [
            "python",
            "-m",
            "sglang.launch_server",
            "--model-path",
            model,
            "--trust-remote-code",
            "--context-length",
            str(max_model_len),
            "--mem-fraction-static",
            str(gpu_mem),
            "--port",
            str(SGLANG_PORT),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_health(f"http://127.0.0.1:{SGLANG_PORT}/health")
        body = _http_json(
            f"http://127.0.0.1:{SGLANG_PORT}/generate",
            {
                "text": prompt,
                "sampling_params": {
                    "max_new_tokens": OUTPUT_TOKENS,
                    "temperature": 0,
                    "ignore_eos": True,
                },
            },
        )
        meta = body.get("meta_info") or {}
        if "completion_tokens" in meta:
            return int(meta["completion_tokens"])
        if "output_token_logprobs" in meta:
            return len(meta["output_token_logprobs"])
        text = body.get("text", "")
        raise RuntimeError(f"could not read completion length from SGLang response: {body!r}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug-dir", type=Path, default=Path("docs/debug"))
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--engine", choices=("vllm", "sglang", "both"), default="both")
    args = parser.parse_args()

    prompt = _load_prompt(args.debug_dir)
    failures: list[str] = []

    for slug, model in MODELS.items():
        if args.engine in ("vllm", "both"):
            n = check_vllm(model, prompt, args.gpu_memory_utilization, args.max_model_len)
            ok = n == OUTPUT_TOKENS
            print(f"[vLLM] {slug}: completion_tokens={n} (expected {OUTPUT_TOKENS})")
            if not ok:
                failures.append(f"vLLM {slug}: got {n}")
        if args.engine in ("sglang", "both"):
            n = check_sglang(model, prompt, args.gpu_memory_utilization, args.max_model_len)
            ok = n == OUTPUT_TOKENS
            print(f"[SGLang] {slug}: completion_tokens={n} (expected {OUTPUT_TOKENS})")
            if not ok:
                failures.append(f"SGLang {slug}: got {n}")

    if failures:
        raise SystemExit("fixed output sanity FAILED: " + "; ".join(failures))
    print("fixed output sanity: all pairs emitted exactly 512 completion tokens")


if __name__ == "__main__":
    main()
