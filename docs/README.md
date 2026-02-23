# ScopeBench documentation index

Use this page as a quick navigation hub for ScopeBench docs.

## Start here

- [Interactive quickstart walkthrough](tutorials/interactive_quickstart_walkthrough.md) — guided CLI + API tour.
- [Quickstart notebook](notebooks/scopebench_quickstart_tutorial.ipynb) — executable tutorial for template selection, plan editing, and effect annotation.
- [Dataset/calibration/plugin lab guide](tutorials/dataset_calibration_plugin_lab.md) — advanced onboarding for contribution and extension workflows.
- [Dataset/calibration/plugin lab notebook](notebooks/scopebench_dataset_calibration_plugin_lab.ipynb) — executable advanced lab for dataset contribution, calibration tuning, and plugin authoring.
- [README](../README.md) — high-level overview, installation, and command reference.
- [Contributing guide](../CONTRIBUTING.md) — local setup and quality gates for contributors.

## Notebook quickstart

Run the interactive tutorials locally:

```bash
pip install -e ".[dev]"
jupyter lab docs/notebooks/
```

If you prefer a pure-CLI flow, use the paired markdown walkthroughs under `docs/tutorials/`.

## Core concepts

- [Effects modeling](effects.md) — structured effects, precedence, and annotation workflow.
- [Template library](templates.md) — domain presets and template naming grammar.
- [Policy architecture](policy/README.md) — backend model and evaluation flow.
- [Multi-agent sessions](multi_agent_session.md) — global contracts and cross-agent scope accounting.

## Integrations and extension

- [Python integrations SDK](integrations/python_sdk.md) — embed ScopeBench in orchestration frameworks.
- [Integration starters (adoption)](integrations/adoption_starters.md) — copy-paste golden paths for Airflow, Prefect, and Dagster.
- [Plugin ecosystem](plugins.md) — create, sign, and validate plugin bundles.
- [Plugin marketplace catalog](plugin_marketplace.yaml) — community domain/plugin metadata.

## Benchmarking and evaluation

- [Rubric v1](rubric_v1.md) — annotation guidance and benchmark expectations.
- [Architecture blueprint](BLUEPRINT.md) — long-term design, roadmap, and open questions.
- [Adoption telemetry baseline](adoption_telemetry.md) — opt-in events and funnel schema for measuring activation and retention.

## Architecture decisions

- [ADR guide](adr/README.md) — ADR process and index.
- [ADR template](adr/template.md) — starter template for new architecture decisions.
