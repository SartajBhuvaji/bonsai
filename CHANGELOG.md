# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions follow [semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Project skeleton: hatchling packaging with a `src` layout, ruff, mypy in strict
  mode, pytest, and tox as the single definition of every check.
- CI on Python 3.11, 3.12 and 3.13, plus a distribution build check.
- Label-driven releases: `patch`, `minor` or `major` on a PR bumps the version,
  tags and publishes on merge.
