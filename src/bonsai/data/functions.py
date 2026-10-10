"""Function-level TypeScript samples for KD, instead of raw file continuations.

`bonsai.training.kd_run.cache_teacher_logit_file`'s default data source --
`bonsai.data.load_samples` -- truncates each sample to the first `max_length`
tokens of a raw file from `bigcode/the-stack-dedup/data/typescript`, cut
wherever that happens to fall. Phase 4's distillation runs (plan.md,
2026-10-09/10) show training against that shape teaches a student to
*continue* arbitrary code, never to *finish a single bounded function and
stop* -- the exact shape HumanEval-TS evaluates. A pruned-but-undistilled
checkpoint scores a real 0.475 pass@1 zero-shot; the same checkpoint after KD
against raw-file continuations scores 0.0, with or without LoRA. This module
reshapes the KD training pool to match the eval task: one complete, bounded
function (plus its leading comment, if any) per sample.

Parsing is real TypeScript syntax (`tree-sitter-typescript`), not a
brace-counting heuristic -- strings, template literals, and regex literals
can all contain unbalanced `{`/`}`, which a heuristic would mis-split on.
"""

from __future__ import annotations

import random
from typing import Any, Literal

import tree_sitter_typescript as tsts
from tree_sitter import Language, Node, Parser

from bonsai.config import DataConfig
from bonsai.data import load_samples

_LANGUAGE = Language(tsts.language_typescript())


def _unwrap_export(node: Node) -> Node:
    """`export (default)? <decl>` -> `<decl>`; anything else is returned as-is."""
    if node.type != "export_statement":
        return node
    for child in node.children:
        if child.type not in ("export", "default"):
            return child
    return node


def _leading_comment_start(siblings: list[Node], index: int) -> int:
    """Byte offset of the first comment directly above `siblings[index]`, or its own start.

    "Directly above" means no blank source line between one comment, the
    next, and the statement itself -- stops at the first gap or non-comment
    sibling, so an unrelated comment higher up the file isn't swept in.
    """
    node = siblings[index]
    next_start_row = node.start_point[0]
    start_byte = node.start_byte
    i = index - 1
    while i >= 0 and siblings[i].type == "comment":
        comment = siblings[i]
        if next_start_row - comment.end_point[0] > 1:
            break
        start_byte = comment.start_byte
        next_start_row = comment.start_point[0]
        i -= 1
    return start_byte


def extract_functions(content: str, *, min_chars: int = 20, max_chars: int = 4000) -> list[str]:
    """Top-level, bounded function definitions in `content`, each with its leading comment.

    Matches `function foo() {}` (named declarations, `export` and `export
    default` included) and `const foo = (...) => ...` / `const foo =
    function () {}` (single-declarator only -- `const a = 1, b = () => 2` is
    skipped rather than risk emitting an invalid fragment). Class methods and
    nested/non-top-level functions are out of scope for this pass. Returns
    `[]` on unparseable input instead of raising -- `the-stack-dedup` rows
    are untrusted scraped source, some of it already broken, and one bad
    file shouldn't abort a batch.
    """
    try:
        source = content.encode("utf-8")
    except UnicodeEncodeError:
        return []
    tree = Parser(_LANGUAGE).parse(source)
    siblings = list(tree.root_node.children)

    spans: list[tuple[int, int]] = []
    for index, raw in enumerate(siblings):
        node = _unwrap_export(raw)
        if node.type == "function_declaration":
            if node.child_by_field_name("body") is None:
                continue  # ambient `declare function foo(): void;`
        elif node.type == "lexical_declaration":
            declarators = [c for c in node.children if c.type == "variable_declarator"]
            if len(declarators) != 1:
                continue
            value = declarators[0].child_by_field_name("value")
            if value is None or value.type not in ("arrow_function", "function_expression"):
                continue
        else:
            continue
        start = _leading_comment_start(siblings, index)
        spans.append((start, raw.end_byte))

    functions = [source[start:end].decode("utf-8") for start, end in spans]
    return [f for f in functions if min_chars <= len(f) <= max_chars]


def load_function_samples(
    data: DataConfig,
    pool: Literal["calibration", "retraining"],
    n: int,
    *,
    raw_file_budget: int | None = None,
    shard_limit: int | None = None,
    seed: int = 0,
    token: str | None = None,
    min_chars: int = 20,
    max_chars: int = 4000,
) -> list[dict[str, Any]]:
    """`n` function-level samples (`{"content": <comment + signature + body>}`).

    Streams `raw_file_budget` whole files via `bonsai.data.load_samples`
    (same disjoint calibration/retraining shard split), extracts every
    top-level function from each, then shuffles and slices to `n`. Defaults
    `raw_file_budget` to `max(n * 3, 500)`: most files yield zero or a
    handful of matching functions, so getting `n` usable examples needs
    several times `n` source files.
    """
    budget = raw_file_budget if raw_file_budget is not None else max(n * 3, 500)
    raw = load_samples(data, pool, budget, shard_limit=shard_limit, seed=seed, token=token)

    functions: list[str] = []
    for row in raw:
        functions.extend(
            extract_functions(row["content"], min_chars=min_chars, max_chars=max_chars)
        )

    random.Random(seed).shuffle(functions)
    return [{"content": f} for f in functions[:n]]


__all__ = ["extract_functions", "load_function_samples"]
