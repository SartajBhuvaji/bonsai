# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions follow [semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `bonsai.config`: TOML-loaded experiment configuration (model / data /
  compute), plus `configs/qwen3-30b-a3b.toml`.
- `bonsai.tracking`: local run tracking under `runs/` -- metrics as
  JSON-lines, config snapshots, no external service.
- `bonsai.architecture`: inspects a Qwen3-MoE model's parameter breakdown by
  pruning axis (attention / router / experts / embedding) from `config.json`
  alone, via a `meta`-device model build. No weights downloaded.
- `bonsai.viz`: a two-panel architecture diagram (whole-model stack, one
  decoder layer expanded with its residual connections and expert grid),
  drawn entirely from an `ArchitectureSummary` -- no hardcoded model numbers.
- `notebooks/`: explainer notebooks alongside the code they call, with the
  convention documented in `notebooks/README.md`. First one,
  `00_model_architecture.ipynb`, walks Qwen3-30B-A3B-Base's real architecture,
  diagrams it, and states the pruning targets for each axis, including the
  expert axis neither Minitron paper covers.
- `tools/check_notebooks.py` and a `notebooks` tox/CI job: committed notebooks
  must keep their execution outputs.
- Runtime dependencies: `transformers`, `huggingface-hub`, `torch`. A
  `notebook` extra (`jupyter`, `nbconvert`, `ipykernel`, `matplotlib`).

## [0.2.13] - 2026-09-14

## [0.2.12] - 2026-09-13

### Added

- The original Minitron paper (arXiv 2407.14679 v2) in `docs/papers/`, alongside
  the follow-up already there.

## [0.2.11] - 2026-09-13

### Changed

- Dependabot now watches GitHub Actions only, quarterly, grouped into one PR.
  The pip ecosystem is gone: the dependencies here are `>=` floors with no
  lockfile, so bumping them changed nothing about what actually gets installed.

## [0.2.10] - 2026-09-13

## [0.2.9] - 2026-09-13

## [0.2.8] - 2026-09-13

## [0.2.7] - 2026-09-13

## [0.2.6] - 2026-09-13

## [0.2.5] - 2026-09-13

## [0.2.4] - 2026-09-13

## [0.2.3] - 2026-09-13

## [0.2.2] - 2026-09-13

### Fixed

- Dependabot no longer applies its own `major`/`minor`/`patch` labels, which
  collided with the release labels and would have versioned bonsai off a
  dependency's semver bump. Its PRs are now labelled `dependencies` and `patch`.

## [0.2.1] - 2026-09-13

### Added

- `docs/papers/`, with the Minitron paper (arXiv 2408.11796 v3) and an index
  recording the version, retrieval date and checksum of each PDF.

## [0.2.0] - 2026-09-13

### Added

- Project skeleton: hatchling packaging with a `src` layout, ruff, mypy in strict
  mode, pytest, and tox as the single definition of every check.
- CI on Python 3.11, 3.12 and 3.13, plus a distribution build check.
- Label-driven releases: `patch`, `minor` or `major` on a PR bumps the version,
  tags and publishes on merge.
