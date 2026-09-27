"""Tests against a tiny local fixture matching `nuprl/MultiPL-E`'s real schema.

`bigcode-eval` lives in the `eval` extra, not `dev` -- it's only needed on
the GPU instance actually running an eval. `importorskip` keeps this file
collectible (and skipped, not failing) for anyone who hasn't installed it.
Not `mbpp-ts` itself: that's still a real Hub fetch, a bad fit for a fast,
always-run test.
"""

from __future__ import annotations

import pytest

bigcode_eval = pytest.importorskip("bigcode_eval")

from datasets import Dataset, DatasetDict  # noqa: E402

from bonsai.eval import (  # noqa: E402
    TASK_NAME,
    MbppMultiPLE,
    TsMultiPLE,
    count_completed_tasks,
    intermediate_generations_path,
    load_intermediate_generations,
    register,
    strip_module_exports,
)


def fixture_dataset() -> DatasetDict:
    # Same columns nuprl/MultiPL-E's real mbpp-ts config carries, confirmed
    # against the actual dataset: doctests, language, name, original, prompt,
    # prompt_terminology, stop_tokens, tests.
    rows = [
        {
            "doctests": "",
            "language": "ts",
            "name": "mbpp_3_is_not_prime",
            "original": "",
            "prompt": "function isNotPrime(n: number): boolean {\n",
            "prompt_terminology": "reworded",
            "stop_tokens": ["\nfunction ", "\n/*", "\n//", "\nclass"],
            "tests": "console.assert(isNotPrime(2) === false);",
        },
        {
            "doctests": "",
            "language": "ts",
            "name": "mbpp_4_unique_elements",
            "original": "",
            "prompt": "function uniqueElements(xs: number[]): boolean {\n",
            "prompt_terminology": "reworded",
            "stop_tokens": ["\nfunction ", "\n/*", "\n//", "\nclass"],
            "tests": "console.assert(uniqueElements([1, 2]) === true);",
        },
    ]
    return DatasetDict({"test": Dataset.from_list(rows)})


def test_dataset_name_targets_mbpp_not_humaneval() -> None:
    task = MbppMultiPLE(dataset=fixture_dataset())

    assert task.DATASET_NAME == "mbpp-ts"


def test_get_dataset_returns_the_test_split() -> None:
    task = MbppMultiPLE(dataset=fixture_dataset())

    assert len(task.get_dataset()) == 2
    assert task.get_dataset()[0]["name"] == "mbpp_3_is_not_prime"


def test_get_prompt_and_reference_match_general_multiple_convention() -> None:
    task = MbppMultiPLE(dataset=fixture_dataset())
    doc = task.get_dataset()[0]

    assert task.get_prompt(doc) == doc["prompt"].strip()
    assert task.get_reference(doc) == doc["tests"]


def test_stop_words_come_from_the_dataset_plus_file_sep() -> None:
    task = MbppMultiPLE(dataset=fixture_dataset())

    assert task.stop_words == ["\nfunction ", "\n/*", "\n//", "\nclass", "<file_sep>"]


def test_requires_execution() -> None:
    task = MbppMultiPLE(dataset=fixture_dataset())

    assert task.requires_execution is True


def test_intermediate_generations_path_matches_bigcode_eval(tmp_path) -> None:
    save_path = tmp_path / "runs" / "foo" / "mbpp-ts-generations.json"
    expected = tmp_path / "runs" / "foo" / "mbpp-ts-generations_multiple-mbpp-ts_intermediate.json"

    assert intermediate_generations_path(save_path, "multiple-mbpp-ts") == expected


def test_load_intermediate_generations_missing_returns_none(tmp_path) -> None:
    path = tmp_path / "missing.json"

    assert load_intermediate_generations(path) is None


def test_count_completed_tasks_skips_empty_slots() -> None:
    intermediate = [[], ["code"], [], ["a", "b"]]

    assert count_completed_tasks(intermediate) == 2


def test_register_adds_task_to_the_harness_registry() -> None:
    register()

    import bigcode_eval.tasks as bc_tasks

    assert bc_tasks.TASK_REGISTRY[TASK_NAME] is MbppMultiPLE
    assert bc_tasks.TASK_REGISTRY["multiple-ts"] is TsMultiPLE
    assert TASK_NAME in bc_tasks.ALL_TASKS
    assert "multiple-ts" in bc_tasks.ALL_TASKS


def test_strip_module_exports_removes_export_default() -> None:
    code = "function f(): void {}\nexport default f;\n"

    assert strip_module_exports(code) == "function f(): void {}\n"


def test_strip_module_exports_removes_named_export() -> None:
    code = "function f(): void {}\nexport { f };\n"

    assert strip_module_exports(code) == "function f(): void {}\n"


def test_strip_module_exports_removes_multiple_export_lines() -> None:
    # Observed directly from the corrected teacher (2026-09-21 diagnostic):
    # some completions emit both a default and a named export for the same
    # function.
    code = "function f(): void {}\nexport default f;\nexport { f };\n"

    assert strip_module_exports(code) == "function f(): void {}\n"


def test_strip_module_exports_is_a_noop_when_no_export_present() -> None:
    code = "function f(): void {}\n"

    assert strip_module_exports(code) == code


def test_strip_module_exports_removes_same_line_export_no_newline() -> None:
    # Regression test: a line-start-anchored version of this regex missed
    # this case on the real 159-problem HumanEval-TS run (2026-09-22) --
    # the model sometimes emits its entire completion as one line, with no
    # newline before the trailing export.
    code = "function f(): void {}export { f };"

    assert strip_module_exports(code) == "function f(): void {}"


def test_postprocess_generation_strips_exports_after_stop_token_truncation() -> None:
    task = MbppMultiPLE(dataset=fixture_dataset())
    prompt = task.get_prompt(task.get_dataset()[0])
    generation = (
        prompt + "\n  return n < 2;\n}\nexport default isNotPrime;\nfunction unrelated() {}"
    )

    result = task.postprocess_generation(generation, 0)

    assert "export" not in result
    assert "unrelated" not in result  # still truncated at the stop token
    assert "return n < 2" in result
