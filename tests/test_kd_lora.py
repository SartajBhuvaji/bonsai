"""Real tests for the PEFT/LoRA mechanics `bonsai.training.kd_run.distill_student`'s
`use_lora` path depends on.

`distill_student` itself can't be unit-tested here -- it loads real models by HF
repo id and needs real GPU/network access, same constraint as the rest of
`bonsai.training.kd_run`. What *is* testable locally, on a tiny real model, is
the PEFT API surface it relies on: that merging a LoRA adapter into its base
model reproduces the adapter's own forward output exactly, that the merged
result is a plain model loadable with `AutoModelForCausalLM.from_pretrained`
alone, and that an adapter saved mid-training can be reconstructed onto a
fresh base checkpoint and merged identically. A bug in any of these would
silently corrupt every LoRA-trained checkpoint this project produces.
"""

from __future__ import annotations

from copy import deepcopy
from typing import cast

import pytest
import torch
from transformers import AutoModelForCausalLM
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM

peft = pytest.importorskip("peft")

TINY_CONFIG = Qwen3MoeConfig(
    vocab_size=64,
    hidden_size=16,
    intermediate_size=32,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    moe_intermediate_size=8,
    num_experts=4,
    num_experts_per_tok=2,
)


def tiny_model() -> Qwen3MoeForCausalLM:
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(deepcopy(TINY_CONFIG))  # type: ignore[no-untyped-call]
    return cast(Qwen3MoeForCausalLM, model)


def lora_config() -> peft.LoraConfig:  # type: ignore[name-defined]
    return peft.LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )


def perturb_lora_b(model: object) -> None:
    """Move the adapter away from its (zero-initialized) identity start, so a
    merge bug that silently drops the adapter would actually change outputs."""
    with torch.no_grad():
        for name, param in model.named_parameters():  # type: ignore[attr-defined]
            if "lora_B" in name:
                param.add_(0.1)


def test_merging_a_lora_adapter_reproduces_its_own_forward_output() -> None:
    peft_model = peft.get_peft_model(tiny_model(), lora_config())
    perturb_lora_b(peft_model)
    peft_model.eval()

    input_ids = torch.randint(0, TINY_CONFIG.vocab_size, (1, 8))
    with torch.no_grad():
        before = peft_model(input_ids=input_ids).logits

    merged = peft_model.to("cpu").merge_and_unload()
    assert isinstance(merged, Qwen3MoeForCausalLM)  # plain model, no PEFT wrapper left
    merged.eval()  # type: ignore[no-untyped-call]
    with torch.no_grad():
        after = merged(input_ids=input_ids).logits

    assert torch.allclose(before, after, atol=1e-5)


def test_merged_model_saves_and_reloads_as_a_plain_checkpoint(tmp_path: object) -> None:
    peft_model = peft.get_peft_model(tiny_model(), lora_config())
    perturb_lora_b(peft_model)
    merged = peft_model.to("cpu").merge_and_unload()
    merged.eval()

    input_ids = torch.randint(0, TINY_CONFIG.vocab_size, (1, 8))
    with torch.no_grad():
        expected = merged(input_ids=input_ids).logits

    merged.save_pretrained(str(tmp_path))
    reloaded = AutoModelForCausalLM.from_pretrained(str(tmp_path))
    reloaded.eval()  # type: ignore[no-untyped-call]
    with torch.no_grad():
        actual = reloaded(input_ids=input_ids).logits

    assert torch.allclose(expected, actual, atol=1e-5)


def test_reconstructing_a_saved_adapter_onto_a_fresh_base_matches_the_original(
    tmp_path: object,
) -> None:
    """Mirrors `distill_student`'s mid-training checkpoint path: `on_val` saves
    only the adapter (`PeftModel.save_pretrained`'s own default), and the
    best-val push later reloads a fresh base and reattaches it."""
    peft_model = peft.get_peft_model(tiny_model(), lora_config())
    perturb_lora_b(peft_model)
    peft_model.eval()

    input_ids = torch.randint(0, TINY_CONFIG.vocab_size, (1, 8))
    with torch.no_grad():
        expected = peft_model(input_ids=input_ids).logits

    peft_model.save_pretrained(str(tmp_path))  # adapter-only, same base never saved

    fresh_base = tiny_model()  # same seed -> identical base weights
    reconstructed = peft.PeftModel.from_pretrained(fresh_base, str(tmp_path))
    reconstructed.eval()
    with torch.no_grad():
        actual = reconstructed(input_ids=input_ids).logits
    assert torch.allclose(expected, actual, atol=1e-5)

    remerged = reconstructed.merge_and_unload()
    remerged.eval()
    with torch.no_grad():
        actual_merged = remerged(input_ids=input_ids).logits
    assert torch.allclose(expected, actual_merged, atol=1e-5)
