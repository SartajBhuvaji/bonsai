"""Guard the notebooks' committed outputs.

The outputs are the evidence a notebook actually ran and produced what its
markdown claims. Some of the experiment notebooks need a rented GPU to
reproduce, so CI cannot just re-run them -- the realistic failure is somebody
stripping the outputs before committing, or a notebook that no longer parses.
This catches both.

Run by the notebooks job in CI, and by `make check` locally.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

NOTEBOOKS = Path(__file__).resolve().parent.parent / "notebooks"


def check(path: Path) -> list[str]:
    problems = []
    nb = json.loads(path.read_text())

    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    if not code:
        problems.append("no code cells")

    for i, cell in enumerate(nb["cells"]):
        source = "".join(cell["source"])

        # nbformat wants a trailing newline on every source line but the last.
        # Get this wrong and the notebook renders correctly in an editor while
        # nbconvert collapses every cell to one line and fails on a SyntaxError.
        for line in cell["source"][:-1]:
            if not line.endswith("\n"):
                problems.append(f"cell {i}: source line missing trailing newline")
                break

        if cell["cell_type"] == "code":
            # IPython magics (`!shell command`, `%line`, `%%cell`) aren't valid
            # Python -- blank them out line-for-line (preserving line numbers,
            # for errors that reference one) rather than pulling in IPython's
            # own cell transformer just to make ast.parse tolerate them. A
            # magic line's own trailing `\` continuations are blanked too --
            # they're shell continuations, not Python ones, and left alone
            # they parse as free-floating expressions on the next line(s).
            parseable_lines = []
            in_magic_continuation = False
            for line in source.split("\n"):
                if in_magic_continuation or line.lstrip().startswith(("!", "%")):
                    in_magic_continuation = line.rstrip().endswith("\\")
                    parseable_lines.append("")
                else:
                    parseable_lines.append(line)
            try:
                ast.parse("\n".join(parseable_lines))
            except SyntaxError as err:
                problems.append(f"cell {i}: {err}")

    executed = [c for c in code if c.get("execution_count")]
    if len(executed) != len(code):
        problems.append(
            f"{len(code) - len(executed)} of {len(code)} code cells have no execution "
            "count -- the outputs are the point, do not strip them"
        )

    if not any(c.get("outputs") for c in code):
        problems.append("no outputs at all")

    errors = [o for c in nb["cells"] for o in c.get("outputs", []) if o["output_type"] == "error"]
    if errors:
        problems.append(f"{len(errors)} cell(s) recorded an error")

    return problems


def main() -> int:
    notebooks = sorted(NOTEBOOKS.glob("*.ipynb"))
    if not notebooks:
        print(f"no notebooks found under {NOTEBOOKS}")
        return 1

    failed = False
    for path in notebooks:
        problems = check(path)
        nb = json.loads(path.read_text())
        outputs = sum(len(c.get("outputs", [])) for c in nb["cells"])
        if problems:
            failed = True
            print(f"FAIL {path.relative_to(NOTEBOOKS)}")
            for p in problems:
                print(f"       {p}")
        else:
            print(
                f"ok   {path.relative_to(NOTEBOOKS)}  ({len(nb['cells'])} cells, {outputs} outputs)"
            )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
