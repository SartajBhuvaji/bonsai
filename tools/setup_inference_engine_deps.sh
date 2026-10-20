#!/usr/bin/env bash
# Box-local pip pins: vLLM and SGLang conflict on torch/kernels; swap before each engine block.
set -euo pipefail

VENV="${VENV:-/root/venv}"
source "${VENV}/bin/activate"

engine="${1:?usage: $0 vllm|sglang}"

if [[ "$engine" == "vllm" ]]; then
  pip install -q \
    "openai>=2.25.0" \
    "humming-kernels[cu13]==0.1.17" \
    "flashinfer-python==0.7.0.post1" \
    "compressed-tensors==0.17.0" \
    "xgrammar==0.2.8" \
    "tokenspeed-mla==0.2.15"
  python -c "import humming.tune.process_input; import vllm; print('vllm deps ok', vllm.__version__)"
elif [[ "$engine" == "sglang" ]]; then
  pip install -q \
    "openai==2.6.1" \
    "humming-kernels[cu13]==0.1.12" \
    "flashinfer-python==0.7.1" \
    "compressed-tensors==0.18.0" \
    "xgrammar==0.2.7" \
    "tokenspeed-mla==0.1.8"
  python -c "import sglang; print('sglang deps ok', sglang.__version__)"
else
  echo "unknown engine: $engine" >&2
  exit 1
fi
