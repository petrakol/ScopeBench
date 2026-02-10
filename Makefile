.PHONY: setup lint format test typecheck audit precommit-install precommit-run demo otel-replay

setup:
	python -m pip install -U pip
	python -m pip install -e ".[dev]"

lint:
	ruff check .

format:
	ruff format .

test:
	pytest -q

typecheck:
	mypy scopebench

audit:
	pip-audit

precommit-install:
	pre-commit install

precommit-run:
	pre-commit run --all-files

demo:
	scopebench quickstart

otel-replay:
	python scripts/replay_examples.py --enable-console
