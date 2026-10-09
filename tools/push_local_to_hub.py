#!/usr/bin/env python3
"""Push a local save_pretrained directory to a Hugging Face model repo."""

from __future__ import annotations

import argparse

from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_dir", help="local directory from save_pretrained")
    parser.add_argument("repo_id", help="HF model repo id (org/name)")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument(
        "--hub-public",
        action="store_true",
        help="create/update repo as public (default: private)",
    )
    args = parser.parse_args()
    private = not args.hub_public
    token = args.hf_token

    api = HfApi(token=token)
    api.create_repo(args.repo_id, repo_type="model", private=private, exist_ok=True)
    if args.hub_public:
        api.update_repo_settings(args.repo_id, private=False, token=token)

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_dir, dtype="auto", device_map="cpu", token=token
    )
    tok = AutoTokenizer.from_pretrained(args.checkpoint_dir, token=token)
    model.push_to_hub(args.repo_id, token=token, private=private)  # type: ignore[arg-type]
    tok.push_to_hub(args.repo_id, token=token, private=private)
    vis = "public" if args.hub_public else "private"
    print(f"pushed {args.checkpoint_dir} -> https://huggingface.co/{args.repo_id} ({vis})")


if __name__ == "__main__":
    main()
