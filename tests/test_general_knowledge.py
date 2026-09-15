"""Tests the shot-count table only -- everything else in this module needs a
real model and a real (if small, ungated) Hub dataset per task, which is what
the actual baseline-eval notebook run is for, not a fast offline unit test.
"""

from __future__ import annotations

import pytest

pytest.importorskip("lm_eval")

from bonsai.eval.general_knowledge import NUM_FEWSHOT


def test_num_fewshot_covers_exactly_the_tracked_tasks() -> None:
    assert set(NUM_FEWSHOT) == {"mmlu", "arc_challenge", "winogrande"}


def test_num_fewshot_values_are_positive() -> None:
    assert all(n > 0 for n in NUM_FEWSHOT.values())
