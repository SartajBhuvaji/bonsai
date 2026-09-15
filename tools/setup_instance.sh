#!/usr/bin/env bash
# Prepare a fresh Lambda Stack instance. Run ON the instance, from inside a
# clone of this repo (git clone git@github.com:SartajBhuvaji/bonsai.git first).
#
# Lambda Stack ships torch + CUDA via apt but NOT transformers. The venv uses
# --system-site-packages so it inherits the apt torch instead of pulling a second
# wheel that may not match the installed CUDA.
#
# WITH_EVAL=1 ./tools/setup_instance.sh also installs the eval extra (lm-eval,
# bigcode-eval) plus Node.js + the TypeScript compiler -- MultiPL-E's TS task
# shells out to `tsc` and `node` to actually compile and run generated code,
# and neither is part of any Python package. Off by default: a plain
# calibration-data or architecture-notebook run doesn't need either.

set -euo pipefail

python3 -m venv --system-site-packages ~/venv
source ~/venv/bin/activate
pip install -q --upgrade pip

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

pip install -q -e ".[notebook]"

if [ "${WITH_EVAL:-0}" = "1" ]; then
  pip install -q -e ".[eval]"
  curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
  sudo apt-get install -y nodejs
  sudo npm install -g typescript
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

echo "SETUP_DONE"
