---
license: apache-2.0
language:
- code
tags:
- moe
- model-compression
- importance-scoring
- typescript
- qwen3
size_categories:
- n<1K
pretty_name: Bonsai importance scores (Qwen3-30B-A3B-Base)
---

# Bonsai importance scores

Activation-based importance tensors for **`Qwen/Qwen3-30B-A3B-Base`**, computed once on a fixed TypeScript calibration set and reused for every structured-pruning sweep in the [bonsai](https://github.com/SartajBhuvaji/bonsai) project (Minitron-style depth / expert / head selection).

This is a **private** artifact repo: request access from the owner if you need to reproduce Phase 4 candidate builds without re-running the forward-hook job.

## Files

| File | Description |
|------|-------------|
| `importance-scores.pt` | PyTorch dict of score tensors + metadata (see below) |

## Provenance

- **Source run (local):** `runs/20260919T150941Z-ff00be-importance-scores/`
- **Uploaded:** 2026-09-27 (byte-identical to the local run file; verified after download in `tools/build_candidate.py`)
- **Base model:** [Qwen/Qwen3-30B-A3B-Base](https://huggingface.co/Qwen/Qwen3-30B-A3B-Base) (bf16 forward passes)
- **Calibration data:** shard 0 of `bigcode/the-stack-dedup`, path `data/typescript` (see `configs/qwen3-30b-a3b.toml`)
- **Samples:** 48 sequences, mean length ~177 tokens

## Contents of `importance-scores.pt`

Keys loaded by notebooks and pruning code:

| Key | Shape | Meaning |
|-----|-------|---------|
| `head_scores` | `[48, 32]` | Per-layer attention head importance (`bonsai.importance.head_importance`, L2 norm aggregation) |
| `expert_scores` | `[48, 128]` | Per-layer MoE expert importance (gate-weighted output norm; REAP / Minitron-MoE style) |
| `depth_scores` | `[48]` | Block Importance per layer (`bonsai.importance.depth.block_importance`) |
| `num_layers`, `num_heads`, `num_experts`, `num_samples`, `mean_length` | scalars | Architecture + calibration summary |

Visualization and methodology: project notebook `notebooks/03_importance_scores.ipynb`.

## Usage

```python
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

path = Path(
    hf_hub_download(
        "sartajbhuvaji/bonsai-importance-scores",
        "importance-scores.pt",
        repo_type="dataset",
    )
)
data = torch.load(path, map_location="cpu", weights_only=False)
depth_scores = data["depth_scores"]
expert_scores = data["expert_scores"]
```

Or let the project loader pick local vs Hub automatically:

```bash
python tools/build_candidate.py --push-to-hub <your-candidate-repo>
```

(`tools/build_candidate.py` tries `runs/20260919T150941Z-ff00be-importance-scores/importance-scores.pt` first, then this dataset.)

## Downstream artifacts

These scores lock the **aggressive tier, depth+experts** candidate used in Phase 4:

- 16 layers pruned (lowest Block Importance), 80 experts pruned globally, with three [Super Expert](https://arxiv.org/abs/2507.23279) routes protected (layers 1/2/3, experts 68/92/82 on original layer indices).
- Built candidate: [`sartajbhuvaji/bonsai-candidate-aggressive-depth-experts`](https://huggingface.co/sartajbhuvaji/bonsai-candidate-aggressive-depth-experts) (private).

## License

Scores derived from forward passes on Qwen3-30B-A3B-Base; follow the [Qwen license](https://huggingface.co/Qwen/Qwen3-30B-A3B-Base) for model-derived artifacts. Calibration text comes from The Stack (see dataset card for `bigcode/the-stack-dedup`).

## Contact

Sartaj Bhuvaji — [sbhuvaji@seattleu.edu](mailto:sbhuvaji@seattleu.edu)
