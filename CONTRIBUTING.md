# Contributing to ScopeBench

Thanks for contributing.

## What is in-scope

- New domains (SWE, ops, finance, health) via **tool registry** expansions
- Better axis scoring (still explainable)
- Better aggregation (scope laundering resistance)
- Policy-as-code integrations (OPA/Cedar)
- New ScopeBench cases + clearer labeling rubric
- Better tracing / replay tools

## What is intentionally out-of-scope (for MVP)

- Training new foundation models
- Content moderation policies (toxicity/jailbreak) — other tools already do this well

## Development

```bash
pip install -e ".[dev]"
pytest -q
ruff check .
```
