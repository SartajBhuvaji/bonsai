# bonsai
#
# Thin wrapper over tox. tox.ini holds the real definitions and CI runs the same
# environments, so `make check` and CI cannot disagree.

.PHONY: help install check lint fmt type test cov notebooks clean

help:
	@echo 'Targets:'
	@echo '  install    editable install with the dev and notebook extras, plus pre-commit hooks'
	@echo '  check      lint + type + tests + notebooks on the current interpreter'
	@echo '  lint       ruff check and format --check (covers notebook cells too)'
	@echo '  fmt        reformat and autofix in place'
	@echo '  type       mypy'
	@echo '  test       pytest on the current interpreter'
	@echo '  cov        tests with a coverage report'
	@echo '  notebooks  check the committed notebooks still carry their outputs'
	@echo '  clean      remove caches and build output'

install:
	pip install -e ".[dev,notebook]"
	pre-commit install

check:
	tox -e lint,type,py,notebooks

lint:
	tox -e lint

fmt:
	tox -e fmt

type:
	tox -e type

test:
	tox -e py

cov:
	tox -e cov

notebooks:
	tox -e notebooks

clean:
	rm -rf build dist .tox .ruff_cache .pytest_cache .mypy_cache htmlcov .coverage coverage.xml
	find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
