"""Inspect a Qwen3-MoE model's architecture without downloading its weights.

`AutoConfig.from_pretrained` only fetches `config.json`, and building the
model on `meta` device allocates no real memory and needs no checkpoint --
every tensor exists as a shape and a dtype only. That's enough to answer
"which weight tensors correspond to which pruning axis" exactly, against the
real module structure rather than a guess from the config file alone.

Confirmed by reading `transformers.models.qwen3_moe.modeling_qwen3_moe`
directly: there is no shared expert, and the experts in a layer are not
separate submodules -- `Qwen3MoeExperts` holds them as two batched parameter
tensors, `gate_up_proj` and `down_proj`, with the expert index as dimension 0.
Pruning an expert means slicing that dimension, not deleting a submodule from
a list.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeDecoderLayer,
    Qwen3MoeForCausalLM,
    Qwen3MoeModel,
    Qwen3MoeSparseMoeBlock,
)


@dataclass(frozen=True, slots=True)
class LayerParams:
    """Parameter count for one transformer block, split by pruning axis."""

    layer_idx: int
    is_sparse: bool
    attention: int
    router: int
    experts: int
    norms: int

    @property
    def total(self) -> int:
        return self.attention + self.router + self.experts + self.norms


@dataclass(frozen=True, slots=True)
class ArchitectureSummary:
    """Whole-model parameter breakdown by pruning axis."""

    repo_id: str
    num_layers: int
    num_experts: int
    experts_per_token: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    moe_intermediate_size: int
    embedding: int
    lm_head: int
    layers: tuple[LayerParams, ...]

    @property
    def attention_total(self) -> int:
        return sum(layer.attention for layer in self.layers)

    @property
    def router_total(self) -> int:
        return sum(layer.router for layer in self.layers)

    @property
    def experts_total(self) -> int:
        return sum(layer.experts for layer in self.layers)

    @property
    def total(self) -> int:
        return self.embedding + self.lm_head + sum(layer.total for layer in self.layers)


def _count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _layer_params(layer: Qwen3MoeDecoderLayer, layer_idx: int) -> LayerParams:
    attention = _count_params(layer.self_attn)
    norms = _count_params(layer.input_layernorm) + _count_params(layer.post_attention_layernorm)

    if isinstance(layer.mlp, Qwen3MoeSparseMoeBlock):
        router = _count_params(layer.mlp.gate)
        experts = _count_params(layer.mlp.experts)
        is_sparse = True
    else:
        # mypy infers `layer.mlp` as exactly Qwen3MoeSparseMoeBlock and calls
        # this branch unreachable. It isn't: Qwen3MoeDecoderLayer.__init__
        # assigns a dense Qwen3MoeMLP here instead whenever the layer is in
        # config.mlp_only_layers, and test_architecture.py's dense-layer test
        # exercises exactly this path against a real (tiny) config.
        router = 0  # type: ignore[unreachable]
        experts = _count_params(layer.mlp)
        is_sparse = False

    return LayerParams(
        layer_idx=layer_idx,
        is_sparse=is_sparse,
        attention=attention,
        router=router,
        experts=experts,
        norms=norms,
    )


def load_meta_model(repo_id: str, revision: str = "main") -> Qwen3MoeForCausalLM:
    """Build the model on `meta` device: real module structure, no weights.

    `from_config`, not `from_pretrained` -- this never touches the checkpoint,
    only `config.json`. The cast is verified, not assumed: `AutoConfig` picks
    the model class from `config.json`'s `architectures` field, and
    `test_architecture.py` exercises this exact path against a real (tiny)
    Qwen3MoeConfig. `transformers`' own stubs type `AutoModelForCausalLM`
    generically across every architecture it supports, so mypy can't narrow
    this on its own.
    """
    config = AutoConfig.from_pretrained(repo_id, revision=revision)
    with torch.device("meta"):
        # from_config's own return type is untyped in transformers' stubs.
        model = AutoModelForCausalLM.from_config(config)  # type: ignore[no-untyped-call]
    return cast(Qwen3MoeForCausalLM, model)


def summarize(repo_id: str, revision: str = "main") -> ArchitectureSummary:
    """Build a per-axis (attention / router / experts / embedding) parameter breakdown."""
    model = load_meta_model(repo_id, revision=revision)
    config = model.config
    base = cast(Qwen3MoeModel, model.model)

    layers = tuple(
        _layer_params(cast(Qwen3MoeDecoderLayer, layer), idx)
        for idx, layer in enumerate(base.layers)
    )

    # Qwen3MoeAttention.__init__ falls back to this same computation when
    # config.head_dim isn't set; mirrored here rather than assumed present,
    # since not every Qwen3-MoE config sets it explicitly (this one does).
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

    return ArchitectureSummary(
        repo_id=repo_id,
        num_layers=len(layers),
        num_experts=config.num_experts,
        experts_per_token=config.num_experts_per_tok,
        hidden_size=config.hidden_size,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=head_dim,
        moe_intermediate_size=config.moe_intermediate_size,
        embedding=_count_params(base.embed_tokens),
        lm_head=0 if config.tie_word_embeddings else _count_params(model.lm_head),
        layers=layers,
    )


__all__ = ["ArchitectureSummary", "LayerParams", "load_meta_model", "summarize"]
