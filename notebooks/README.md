# Notebooks

This project is a learning exercise as much as a research one, so most steps
that produce something worth looking at — an architecture, a distribution of
importance scores, a loss curve, an eval table — get a notebook here alongside
the code that does the work.

## What goes in a notebook vs. in `src/bonsai/`

Logic lives in `src/bonsai/`; notebooks call it. A code cell here should
mostly look like:

```python
from bonsai.architecture import summarize

summary = summarize("Qwen/Qwen3-30B-A3B-Base")
```

not the implementation of `summarize` itself. Two reasons: the same
importance-scoring or architecture-inspection code a notebook uses to explain
something also needs to be callable from the actual pruning pipeline later,
so it can't only exist inside a cell — and a notebook stays short enough to
read when it isn't also carrying the implementation.

What belongs directly in a notebook: markdown that connects a paper's claim to
this specific model's numbers, and one-time or visual code — a plot, a
rendered table, poking at a value once — that doesn't need a permanent home in
`src/bonsai/` just for that.

## Naming

Numbered to match the roadmap this project follows, e.g.
`00_model_architecture.ipynb`, `01_calibration_data.ipynb`. Not every step
gets one — plumbing (a provisioning script, a config loader) doesn't produce
anything worth looking at.

## Committed outputs

A committed notebook keeps its execution output. The output is the evidence
the notebook actually ran and produced what its markdown claims — some of
these need a rented GPU to reproduce, so nothing re-runs them in CI.
`tools/check_notebooks.py` fails a notebook whose outputs were stripped, or
that no longer parses; it runs in CI and in `make check`.

## Running one

```bash
pip install -e ".[notebook]"
jupyter lab notebooks/
```
