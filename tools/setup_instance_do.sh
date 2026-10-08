#!/usr/bin/env bash
# Prepare a DigitalOcean GPU Droplet (Ubuntu + NVIDIA drivers).
# Run ON the instance as root, from /root/bonsai (clone or rsync this repo first).
#
# Unlike Lambda Stack, DO images do not ship torch via apt. This script creates
# /root/venv and installs CUDA-enabled torch from PyPI.
#
# WITH_EVAL=1 also installs the eval extra (lm-eval, bigcode-eval) plus Node.js
# and the TypeScript compiler for MultiPL-E TS tasks.
#
# Optional: BONSAI_TORCH_INDEX (e.g. https://download.pytorch.org/whl/cu124) if
# the default torch wheel does not see the GPU.

set -euo pipefail

ROOT="${BONSAI_ROOT:-/root/bonsai}"
VENV="${BONSAI_VENV:-/root/venv}"

cd "$ROOT"

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq software-properties-common git build-essential
if ! command -v python3.12 >/dev/null; then
  add-apt-repository -y ppa:deadsnakes/ppa
  apt-get update -qq
  apt-get install -y -qq python3.12 python3.12-venv python3.12-dev
fi
PY=python3.12

if [ ! -d "$VENV" ]; then
  "$PY" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -q --upgrade pip wheel

if ! command -v nvidia-smi >/dev/null; then
  echo "FATAL: nvidia-smi not found — use a GPU Droplet image with NVIDIA drivers."
  exit 1
fi
nvidia-smi -L

if [ -n "${BONSAI_TORCH_INDEX:-}" ]; then
  pip install -q torch --index-url "$BONSAI_TORCH_INDEX"
else
  pip install -q "torch>=2.4"
fi

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

pip install -q bitsandbytes
pip install -q -e ".[notebook]"

if [ "${WITH_EVAL:-0}" = "1" ]; then
  pip install -q -e ".[eval]"
  if ! command -v node >/dev/null; then
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
    apt-get install -y -qq nodejs
  fi
  npm install -g typescript
  node --version
  tsc --version
fi

python -m ipykernel install --user --name bonsai --display-name "bonsai (venv)"

python - <<'PY'
import torch, transformers
print("transformers", transformers.__version__, "at", transformers.__file__)
p = torch.cuda.get_device_properties(0)
print(f"gpu {p.name} sm_{p.major}{p.minor}")
print("  bf16 (needs sm_80):", p.major >= 8)
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM
print("Qwen3MoeForCausalLM import OK")
PY

echo "SETUP_DONE venv=$VENV root=$ROOT"
