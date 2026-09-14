#!/usr/bin/env bash
# Prepare a fresh Lambda Stack instance. Run ON the instance, from inside a
# clone of this repo (git clone git@github.com:SartajBhuvaji/bonsai.git first).
#
# Lambda Stack ships torch + CUDA via apt but NOT transformers. The venv uses
# --system-site-packages so it inherits the apt torch instead of pulling a second
# wheel that may not match the installed CUDA.

set -euo pipefail

python3 -m venv --system-site-packages ~/venv
source ~/venv/bin/activate
pip install -q --upgrade pip

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

pip install -q -e ".[notebook]"

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

echo "SETUP_DONE"
