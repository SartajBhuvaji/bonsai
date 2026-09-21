"""Registers MBPP's MultiPL-E TypeScript translation as a bigcode-evaluation-harness task.

`bigcode_eval.tasks.multiple` wraps every MultiPL-E language as a
`multiple-{language}` task, but only for the *HumanEval* translations --
`GeneralMultiPLE.__init__` hardcodes `DATASET_NAME = f"humaneval-{language}"`.
`nuprl/MultiPL-E` also carries MBPP's translations (`mbpp-{language}`), and
its `mbpp-ts` config has the exact same schema as `humaneval-ts` (confirmed
directly: `doctests`, `language`, `name`, `original`, `prompt`,
`prompt_terminology`, `stop_tokens`, `tests`) -- so `MbppMultiPLE` below reuses
every part of `GeneralMultiPLE` (prompt building, postprocessing, execution,
pass@k scoring) except which dataset config it loads.

`bigcode_eval.tasks.TASK_REGISTRY` is a plain module-level dict built at
import time from static imports in `bigcode_eval/tasks/__init__.py` -- there's
no plugin mechanism for external tasks, so `register()` patches this task
into it directly. Call `register()` before anything calls
`bigcode_eval.tasks.get_task`.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from bigcode_eval.tasks.multiple import GeneralMultiPLE
from datasets import load_dataset

if TYPE_CHECKING:
    from datasets import DatasetDict

TASK_NAME = "multiple-mbpp-ts"

# Matches a standalone `export default foo;` or `export { foo, bar };` line,
# with any leading whitespace -- deliberately scoped to lines *starting* with
# `export` (not e.g. `export function foo() {...}`, which the corrected
# teacher was never observed emitting) so this can't accidentally eat a
# legitimate declaration.
_EXPORT_STATEMENT_RE = re.compile(
    r"^[ \t]*export[ \t]+(?:default[ \t]+\w+|\{[^}]*\})[ \t]*;?[ \t]*\n?", re.MULTILINE
)


def strip_module_exports(code: str) -> str:
    """Remove trailing `export default X;` / `export { X };` lines from a completion.

    The corrected teacher (`3.1`) was fine-tuned on raw TypeScript *files*,
    which routinely end in a module export -- unlike MultiPL-E's
    function-completion scaffold, which expects none. Confirmed directly
    (2026-09-21, a 10-problem HumanEval-TS diagnostic): the corrected
    teacher added an export statement to every single completion, which
    dropped raw pass@1 to 0.0 despite every completion's underlying logic
    being correct -- stripping just this line, no regeneration, raised
    pass@1 back to 0.9 on the same completions. This is a no-op for the
    base (uncorrected) model, which was never observed emitting one, so
    1.2's already-recorded baseline stays comparable.
    """
    return _EXPORT_STATEMENT_RE.sub("", code)


class ExportStrippingMultiPLE(GeneralMultiPLE):
    """`GeneralMultiPLE`, with `strip_module_exports` applied after stop-token truncation."""

    def postprocess_generation(self, generation: str, idx: int) -> str:
        prompt: str = self.get_prompt(self.get_dataset()[idx])
        completion: str = generation[len(prompt) :]
        completion = self._stop_at_stop_token(completion, self.stop_words)
        return prompt + strip_module_exports(completion)


class TsMultiPLE(ExportStrippingMultiPLE):
    """`multiple-ts`, export-stripped -- bigcode-eval's own registration wraps
    only the *HumanEval* MultiPL-E translations natively; this replaces that
    entry so HumanEval-TS gets the same fix `MbppMultiPLE` gets below."""

    def __init__(self) -> None:
        super().__init__("ts")


class MbppMultiPLE(ExportStrippingMultiPLE):
    """`ExportStrippingMultiPLE`, pointed at MultiPL-E's `mbpp-{language}` translation."""

    def __init__(self, language: str = "ts", dataset: DatasetDict | None = None) -> None:
        self.language = language
        self.DATASET_NAME = f"mbpp-{language}"
        # `dataset` lets tests inject a tiny local fixture instead of hitting the Hub.
        self.dataset = dataset or load_dataset(
            GeneralMultiPLE.DATASET_PATH,
            self.DATASET_NAME,
            revision=GeneralMultiPLE.DATASET_REVISION,
        )
        # Not Task.__init__(self, stop_words=..., requires_execution=True): its body
        # does its own load_dataset(path=self.DATASET_PATH, name=self.DATASET_NAME)
        # call, redundant with the one above and a real Hub fetch every time --
        # including inside a test that just injected a local fixture. Setting the
        # two attributes it would have set is the whole of what it does otherwise.
        self.stop_words = self.dataset["test"][0]["stop_tokens"] + ["<file_sep>"]
        self.requires_execution = True


def register() -> None:
    """Patch `TsMultiPLE`/`MbppMultiPLE` into bigcode-evaluation-harness's task registry."""
    import bigcode_eval.tasks as bc_tasks

    bc_tasks.TASK_REGISTRY["multiple-ts"] = TsMultiPLE
    bc_tasks.TASK_REGISTRY[TASK_NAME] = MbppMultiPLE
    bc_tasks.ALL_TASKS = sorted(bc_tasks.TASK_REGISTRY)


__all__ = [
    "TASK_NAME",
    "ExportStrippingMultiPLE",
    "MbppMultiPLE",
    "TsMultiPLE",
    "register",
    "strip_module_exports",
]
