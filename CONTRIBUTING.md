# Contributing

## Setup

```bash
make install
```

## Before pushing

```bash
make check
```

`make check` runs `tox -e lint,type,py`, which is what CI runs. `make fmt` fixes
formatting and the autofixable lint findings.

## Pull requests

Branch off `main` and open a PR. Every PR needs exactly one of these labels:

| Label | Version bump | Use for |
| --- | --- | --- |
| `patch` | 0.1.0 to 0.1.1 | bug fixes, docs, internal changes |
| `minor` | 0.1.0 to 0.2.0 | new functionality, backwards compatible |
| `major` | 0.1.0 to 1.0.0 | breaking changes to the public API |

The `PR label` check fails until one is set. Note it fails on two or more as
well, so remove the old label when changing your mind rather than adding a second.

When the PR merges, the release workflow bumps the version in `pyproject.toml`,
moves the `Unreleased` changelog entries under the new version, tags `vX.Y.Z` and
publishes a GitHub release. Nothing to do by hand.

Put user-visible changes under `## [Unreleased]` in `CHANGELOG.md` as part of your PR.
