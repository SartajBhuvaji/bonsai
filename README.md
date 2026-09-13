# bonsai

<!-- TODO: one line on what this does. -->

## Requirements

Python 3.11 or newer.

## Install

```bash
git clone git@github.com:SartajBhuvaji/bonsai.git
cd bonsai
make install
```

That does an editable install with the dev extras and installs the pre-commit hooks.

## Development

Checks are defined in `tox.ini`. CI runs the same environments, so anything that
passes locally passes there.

```bash
make check   # lint + types + tests on the current interpreter
make fmt     # reformat and autofix
make cov     # tests with a coverage report
tox          # every supported interpreter (3.11, 3.12, 3.13)
```

| Command | What it runs |
| --- | --- |
| `tox -e lint` | `ruff check` and `ruff format --check` |
| `tox -e fmt` | `ruff format` and `ruff check --fix` |
| `tox -e type` | `mypy` in strict mode |
| `tox -e py` | `pytest` |
| `tox -e cov` | `pytest` with coverage |

CI never rewrites files. `tox -e lint` reports formatting problems, `tox -e fmt`
fixes them.

## Layout

```
src/bonsai/     the package
tests/          pytest suite
docs/           documentation
.github/        CI and Dependabot
```

## License

MIT. See [LICENSE](LICENSE).
