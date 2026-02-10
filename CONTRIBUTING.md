# Contributing to ScopeBench

Thanks for contributing.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

## Local quality gates

Run the same checks used in CI before opening a PR:

```bash
ruff check .
ruff format --check .
mypy scopebench
pytest -q
pytest --cov=scopebench --cov-report=term-missing
```

## Pre-commit hooks

Install and run pre-commit locally:

```bash
pre-commit install
pre-commit run --all-files
```

Configured hooks:

- `ruff` + `ruff-format`
- `check-yaml`
- `end-of-file-fixer`
- `trailing-whitespace`
- `check-added-large-files`

## Test markers

Use pytest markers to focus local runs:

- `unit` for fast, isolated tests
- `integration` for cross-module behavior
- `slow` for longer-running checks

Examples:

```bash
pytest -q -m unit
pytest -q -m "not slow"
```

## Useful commands

- `make setup` — install package with dev dependencies
- `make lint` — run Ruff lint
- `make format` — format code with Ruff
- `make typecheck` — run mypy on `scopebench/`
- `make test` — run tests
- `make audit` — run `pip-audit`
- `make otel-replay` — replay examples with optional OTel console spans

## CI and automation

- CI runs lint, format checks, tests, packaging smoke test, and mypy.
- Dependency audits run in CI via `pip-audit`.
- Dependabot is configured for Python and GitHub Actions updates.
- PRs use a template that captures problem, approach, validation, risk, and rollback.

## Documentation

- Start from `docs/README.md` for a map of all guides.
- Keep documentation updates alongside behavior changes when possible (CLI flags, API payloads, templates, or policy semantics).
- For architecture-impacting changes, add or update an ADR in `docs/adr/`.

## ADRs

Use `docs/adr/template.md` for changes that affect architecture, policy thresholds,
scoring semantics, or telemetry contracts.
