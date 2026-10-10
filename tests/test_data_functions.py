"""Real TypeScript parsing, not mocked -- the whole point is correct syntax handling."""

from __future__ import annotations

from typing import Any

import pytest

from bonsai.config import DataConfig
from bonsai.data.functions import extract_functions, load_function_samples

# --- extract_functions: matches ---------------------------------------------


def test_extracts_named_function_declaration() -> None:
    src = "function add(a: number, b: number): number {\n  return a + b;\n}\n"

    functions = extract_functions(src)

    assert functions == [src.strip()]


def test_extracts_exported_function_declaration() -> None:
    src = "export function add(a: number, b: number): number {\n  return a + b;\n}"

    functions = extract_functions(src)

    assert functions == [src]


def test_extracts_exported_default_function() -> None:
    src = "export default function run(): number {\n  return 0;\n}"

    functions = extract_functions(src)

    assert functions == [src]


def test_extracts_const_arrow_function_with_block_body() -> None:
    src = "const greet = (name: string): string => {\n  return `hi ${name}`;\n};"

    functions = extract_functions(src)

    assert functions == [src]


def test_extracts_const_arrow_function_with_expression_body() -> None:
    src = "export const mul = (a: number, b: number): number => a * b;"

    functions = extract_functions(src)

    assert functions == [src]


def test_extracts_const_function_expression() -> None:
    src = "const add = function (a: number, b: number): number {\n  return a + b;\n};"

    functions = extract_functions(src)

    assert functions == [src]


def test_leading_comment_is_attached() -> None:
    src = (
        "// Adds two numbers together.\n"
        "function add(a: number, b: number): number {\n"
        "  return a + b;\n"
        "}"
    )

    functions = extract_functions(src)

    assert functions == [src]


def test_leading_jsdoc_comment_is_attached() -> None:
    src = "/**\n * Adds two numbers.\n */\nexport const add = (a: number, b: number) => a + b;"

    functions = extract_functions(src)

    assert functions == [src]


def test_comment_separated_by_blank_line_is_not_attached() -> None:
    src = (
        "// unrelated comment far above\n"
        "\n"
        "function add(a: number, b: number): number {\n"
        "  return a + b;\n"
        "}"
    )

    functions = extract_functions(src)

    assert functions == ["function add(a: number, b: number): number {\n  return a + b;\n}"]


def test_braces_inside_string_and_template_literal_do_not_confuse_the_span() -> None:
    src = (
        "const greet = (name: string): string => {\n"
        '  const brace = "{not a real brace}";\n'
        "  return `hello ${name} {literal}`;\n"
        "};"
    )

    functions = extract_functions(src)

    assert functions == [src]


def test_extracts_multiple_top_level_functions() -> None:
    src = "function first(): number {\n  return 1;\n}\n\nexport const second = (): number => 2;\n"

    functions = extract_functions(src)

    assert len(functions) == 2
    assert "first" in functions[0]
    assert "second" in functions[1]


# --- extract_functions: non-matches / filters -------------------------------


def test_skips_non_function_declarations() -> None:
    src = "const x = 5;\nlet y = 10;\nimport { z } from 'z';\n"

    functions = extract_functions(src)

    assert functions == []


def test_skips_multi_declarator_lexical_declarations() -> None:
    src = "const a = 1, b = () => 2;"

    functions = extract_functions(src)

    assert functions == []


def test_skips_ambient_function_declarations() -> None:
    src = "declare function add(a: number, b: number): number;"

    functions = extract_functions(src)

    assert functions == []


def test_skips_class_methods() -> None:
    src = "class Foo {\n  bar(): void {\n    console.log('hi');\n  }\n}"

    functions = extract_functions(src)

    assert functions == []


def test_does_not_raise_on_unparseable_input() -> None:
    assert extract_functions("{{{ not valid typescript at all (((") == []
    assert extract_functions("") == []


def test_min_chars_filters_out_tiny_functions() -> None:
    src = "const f = () => 1;"

    assert extract_functions(src, min_chars=1000) == []
    assert extract_functions(src, min_chars=1) == [src]


def test_max_chars_filters_out_oversized_functions() -> None:
    src = "function f(): number {\n" + "  const x = 1;\n" * 50 + "  return x;\n}"

    assert extract_functions(src, max_chars=10) == []
    assert extract_functions(src, max_chars=10_000) == [src]


# --- load_function_samples ---------------------------------------------------


def _config() -> DataConfig:
    return DataConfig(
        dataset_id="bigcode/the-stack-dedup",
        data_dir="data/typescript",
        retraining_shards=("data/typescript/data-00000-of-00001.parquet",),
    )


def test_load_function_samples_extracts_and_slices_to_n(monkeypatch: pytest.MonkeyPatch) -> None:
    raw_rows = [
        {
            "content": "function a(): number {\n  return 1;\n}\n\nfunction b(): number {\n  return 2;\n}"
        },
        {"content": "export const c = (): number => 3;"},
        {"content": "const notAFunction = 42;"},
    ]

    def fake_load_samples(
        data: DataConfig,
        pool: str,
        n: int,
        *,
        shard_limit: int | None = None,
        seed: int = 0,
        token: str | None = None,
    ) -> list[dict[str, Any]]:
        return raw_rows

    monkeypatch.setattr("bonsai.data.functions.load_samples", fake_load_samples)

    samples = load_function_samples(_config(), "retraining", n=2, seed=0)

    assert len(samples) == 2
    assert all(set(s.keys()) == {"content"} for s in samples)


def test_load_function_samples_returns_fewer_than_n_if_not_enough_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_rows = [{"content": "export const only = (): number => 1;"}]

    monkeypatch.setattr(
        "bonsai.data.functions.load_samples",
        lambda data, pool, n, **kwargs: raw_rows,
    )

    samples = load_function_samples(_config(), "retraining", n=5, seed=0)

    assert len(samples) == 1


def test_load_function_samples_is_deterministic_for_a_fixed_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_rows = [{"content": f"export const f{i} = (): number => {i};"} for i in range(10)]

    monkeypatch.setattr(
        "bonsai.data.functions.load_samples",
        lambda data, pool, n, **kwargs: raw_rows,
    )

    first = load_function_samples(_config(), "retraining", n=5, seed=0)
    second = load_function_samples(_config(), "retraining", n=5, seed=0)

    assert first == second


def test_load_function_samples_passes_raw_file_budget_to_load_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_load_samples(
        data: DataConfig,
        pool: str,
        n: int,
        *,
        shard_limit: int | None = None,
        seed: int = 0,
        token: str | None = None,
    ) -> list[dict[str, Any]]:
        captured["n"] = n
        return []

    monkeypatch.setattr("bonsai.data.functions.load_samples", fake_load_samples)

    load_function_samples(_config(), "retraining", n=100, seed=0)
    assert captured["n"] == 500  # default: max(n * 3, 500)

    load_function_samples(_config(), "retraining", n=1000, seed=0)
    assert captured["n"] == 3000  # default: max(n * 3, 500)

    load_function_samples(_config(), "retraining", n=10, raw_file_budget=42, seed=0)
    assert captured["n"] == 42
