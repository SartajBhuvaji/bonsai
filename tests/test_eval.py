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

from bonsai.eval import TASK_NAME, MbppMultiPLE, register  # noqa: E402


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


def test_register_adds_task_to_the_harness_registry() -> None:
    register()

    import bigcode_eval.tasks as bc_tasks

    assert bc_tasks.TASK_REGISTRY[TASK_NAME] is MbppMultiPLE
    assert TASK_NAME in bc_tasks.ALL_TASKS
