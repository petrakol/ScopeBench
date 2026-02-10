<img width="973" height="209" alt="ScopeBench_w copy" src="https://github.com/user-attachments/assets/066d740e-f8ad-4876-89c5-48604a5cc7a0" />

# ScopeBench

**ScopeBench** is an open-source Python toolkit for **plan-level proportionality enforcement** ("scope alignment") in tool-using agents.

It is designed for a specific failure mode:

> The agent is directionally correct, but acts at the wrong *magnitude*.
>
> Example: *"Charge my phone"* → *"Re-architect the power grid and deploy solar internationally."*

Instead of only asking *"is this intent safe?"*, ScopeBench asks *"is this plan proportionate to the task contract?"*.

## Table of Contents

- [Why ScopeBench](#why-scopebench)
- [Features](#features)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Interactive tutorials](#interactive-tutorials)
- [Documentation map](#documentation-map)
- [CLI Reference](#cli-reference)
- [API Usage](#api-usage)
- [Examples Included](#examples-included)
- [Testing](#testing)
- [Integrations SDK (Python)](#integrations-sdk-python)
- [Plugin Ecosystem](#plugin-ecosystem)
- [Project Structure](#project-structure)
- [Development](#development)
- [License](#license)

---

## Why ScopeBench

ScopeBench helps you enforce bounded execution before actions run by combining:

- **Task contracts**: explicit envelopes for acceptable scope (goal, constraints, thresholds).
- **Plan DAGs**: step-wise plans with tools and dependencies.
- **Multi-axis scoring**: scope vectors across spatial, temporal, depth, irreversibility, legal exposure, and more.
- **Cumulative accounting**: captures risk accumulation across a plan to resist “scope laundering.”
- **Policy decisions**: deterministic `ALLOW / ASK / DENY` outputs with rule-level rationale.
- **Operational hooks**: OpenTelemetry support plus weekly telemetry/benchmark replay utilities.

---

## Features

- Contract presets (`personal`, `team`, `enterprise`, `regulated`) with threshold envelopes.
- DAG-aware aggregation over plan steps.
- Coding-task policy checks such as:
  - `read_before_write`
  - `validation_after_write`
- CLI modes for one-off runs, quickstarts, red-team stress tests, weekly calibration, and API serving.
- HTTP API with optional:
  - step-level vectors
  - summary + next-step guidance
  - patch suggestions
  - lightweight telemetry fields
  - shadow-mode effective decisions

---

## Installation

### Requirements

- Python **3.10+**

### Install from source

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

### Optional policy backends

ScopeBench supports multiple policy backends:

- `python` (default): no extra runtime dependency.
- `opa`: requires a running OPA server (see [Run with OPA backend](#4-run-with-opa-backend)).
- `cedar`: available through CLI/API backend selection for Cedar-based policy evaluation.

If you do not explicitly set a backend, ScopeBench uses `python`.

---

## Quickstart

### 1) Run the canonical overreach example

```bash
scopebench run examples/phone_charge.contract.yaml examples/phone_charge.plan.yaml
```

Expected result: a `DENY` decision with axis-specific reasons.

For machine-readable output:

```bash
scopebench run examples/phone_charge.contract.yaml examples/phone_charge.plan.yaml --json
```

### 2) Try built-in shortcuts

```bash
scopebench quickstart
scopebench coding-quickstart
```

### Interactive tutorials

For hands-on onboarding, use the guided walkthroughs:

- Web-style walkthrough (step-by-step CLI + API): `docs/tutorials/interactive_quickstart_walkthrough.md`
- Jupyter notebook tutorial: `docs/notebooks/scopebench_quickstart_tutorial.ipynb`

Both tutorials walk through:

1. template selection,
2. plan editing,
3. effect annotation (`scopebench suggest-effects`), and
4. API evaluation with `POST /evaluate`.

### Documentation map

If you are new to ScopeBench or returning after a while, use this map to jump to the right guide:

- `docs/tutorials/interactive_quickstart_walkthrough.md` — end-to-end tutorial from template selection to API evaluation.
- `docs/integrations/python_sdk.md` — embed ScopeBench checks directly inside Python agent orchestration code.
- `docs/effects.md` — model structured step effects and precedence rules.
- `docs/templates.md` — browse available domain templates and naming conventions.
- `docs/policy/README.md` — understand policy backend architecture and backend selection.
- `docs/plugins.md` — build, sign, and publish plugin bundles.
- `docs/rubric_v1.md` — dataset annotation rubric and evaluator expectations.
- `docs/multi_agent_session.md` — global-contract evaluation across multiple cooperating agents.

For an overview of the whole documentation set, see `docs/README.md`.

### Common workflows

#### Evaluate a new plan from template defaults

```bash
scopebench template generate swe > /tmp/swe_bundle.yaml
scopebench template show swe/plan > /tmp/swe.plan.yaml
scopebench template show swe/contract > /tmp/swe.contract.yaml
scopebench run /tmp/swe.contract.yaml /tmp/swe.plan.yaml --json
```

#### Add effect hints before running policy

```bash
scopebench suggest-effects /tmp/swe.plan.yaml --in-place
scopebench run /tmp/swe.contract.yaml /tmp/swe.plan.yaml --json
```

#### Evaluate a live multi-agent session

```bash
scopebench serve --host 0.0.0.0 --port 8080
curl -s http://localhost:8080/evaluate_session \
  -H 'content-type: application/json' \
  -d @examples/multi_agent_session.json | jq '.decision'
```

### 3) Start the API server

```bash
scopebench serve --host 0.0.0.0 --port 8080
```

Health check:

```bash
curl -s http://localhost:8080/health
```

Discover available templates/tools/cases from the API:

```bash
curl -s http://localhost:8080/templates | jq '.count'
curl -s http://localhost:8080/tools | jq '.count'
curl -s http://localhost:8080/cases | jq '.count'
```

---

### 4) Run with OPA backend

Start OPA with the bundled Rego policy:

```bash
opa run --server --addr :8181 scopebench/policy/opa/policy.rego
```

In another shell, start ScopeBench using OPA backend:

```bash
scopebench serve --host 0.0.0.0 --port 8080 --policy-backend opa
```

Evaluate a plan:

```bash
curl -s http://localhost:8080/evaluate \
  -H 'content-type: application/json' \
  -d '{"contract":{"goal":"Fix failing unit test","preset":"team"},"plan":{"task":"Fix failing unit test","steps":[{"id":"1","description":"Read failing test","tool":"git_read"},{"id":"2","description":"Apply patch","tool":"git_patch","depends_on":["1"]},{"id":"3","description":"Run test","tool":"analysis","depends_on":["2"]}]},"include_telemetry":true}'
```

---

## CLI Reference

```bash
scopebench run <contract.yaml> <plan.yaml> [--json] [--compact-json] [--otel-console] [--calibration-scale <float>]
scopebench quickstart [--json] [--compact-json] [--otel-console]
scopebench coding-quickstart [--json] [--compact-json] [--otel-console]
scopebench suggest-effects <plan.yaml> [--json] [--in-place]
scopebench weekly-calibrate <telemetry.jsonl> [--json]
scopebench continuous-learn <telemetry.jsonl> --benchmark scopebench/bench/cases/redteam.jsonl --report-out continuous_learning_report.md [--apply-updates]
python -m scopebench.redteam.generate --count 120 --seed 7 --output scopebench/bench/cases/redteam.jsonl
scopebench judge-bench scopebench/bench/cases/redteam.jsonl --judge heuristic
scopebench serve [--host 127.0.0.1] [--port 8080] [--policy-backend python|opa|cedar]
scopebench dataset-validate <cases.jsonl>
scopebench dataset-suggest --id <case_id> --domain <domain> --instruction <text> --contract <contract.yaml> --plan <plan.yaml> --expected-decision ALLOW|ASK|DENY --expected-rationale <text> [--append-to <cases.jsonl>]
scopebench dataset-pr --title <title> --body <body>
```

---

## API Usage

### `POST /evaluate`

Provide a `contract` object and a `plan` object.

Minimal request:

```json
{
  "contract": {
    "goal": "Fix a failing unit test",
    "preset": "team"
  },
  "plan": {
    "task": "Fix the failing unit test",
    "steps": [
      {"id": "1", "description": "Read failing test and source", "tool": "git_read"},
      {"id": "2", "description": "Apply minimal patch", "tool": "git_patch", "depends_on": ["1"]},
      {"id": "3", "description": "Run targeted test", "tool": "pytest", "depends_on": ["2"]}
    ]
  },
  "include_summary": true,
  "include_telemetry": true,
  "shadow_mode": false
}
```

Response includes:

- `decision` and `effective_decision`
- `reasons`, `exceeded`, `asked`
- aggregate scores and step count
- optional `summary`, `next_steps`, `plan_patch_suggestion`, and `telemetry`
- trace context fields (`trace_id`, `span_id`) for observability

### Useful request toggles

When calling `POST /evaluate`, these flags are especially useful during integration:

- `include_summary`: include a concise natural-language explanation.
- `include_next_steps`: include suggested proportionate follow-up actions.
- `include_patch`: include a patch-style plan transformation suggestion.
- `include_telemetry`: include telemetry payload fields for logging/replay.
- `shadow_mode`: run checks in shadow mode while preserving production behavior.


### Community dataset contribution endpoints

- `POST /dataset/validate` validates a single case object against the rubric/schema requirements (`case_schema_version`, required fields, one vector per step, and axis values in `[0,1]`).
- `POST /dataset/suggest` takes draft metadata + contract/plan and returns a case with `expected_step_vectors` auto-filled from the current evaluator for annotation bootstrap.

These are exposed in the interactive `/ui` through **Validate case draft** and **Suggest vectors** actions in the Dataset Contribution Draft panel.

### `POST /evaluate_stream`

Evaluate long-running plans that evolve over time. Submit an initial plan plus ordered `events` (`add_step`, `update_step`, `remove_step`, `replace_plan`). ScopeBench re-evaluates after each event and returns snapshots with:

- updated aggregate risk and policy decision
- `triggers` when cumulative risk crosses thresholds
- `judge_output_deltas` when per-step judge outputs shift because context changed

Minimal request shape:

```json
{
  "contract": {"goal": "Maintain ML training pipeline", "preset": "team"},
  "plan": {"task": "Maintain pipeline", "steps": [{"id": "1", "description": "Inspect jobs", "tool": "git_read"}]},
  "events": [
    {"event_id": "evt-1", "operation": "add_step", "step_id": "2", "step": {"id": "2", "description": "Launch retraining", "tool": "analysis", "depends_on": ["1"]}},
    {"event_id": "evt-2", "operation": "update_step", "step_id": "2", "step": {"description": "Launch multi-region retraining with new serving dependency"}}
  ],
  "judge": "llm"
}
```

### Additional catalog endpoints

- `GET /templates` returns template metadata and full content for each domain, grouped by variants (default + named domain presets).
- `GET /tools` returns tool registry entries plus a normalized schema payload.
- `GET /cases` returns benchmark metadata (ids, domains, instructions, expected decisions, and contract payloads).
- `GET /telemetry/replay` replays recent telemetry rows from `SCOPEBENCH_TELEMETRY_JSONL_PATH`.
- `GET /ui` serves an interactive web workbench for contract/plan authoring, DAG visualization, axis/rationale inspection, and telemetry replay.

Template variants are also available in the CLI using `<domain>/<variant>/<kind>` naming, e.g.
`scopebench template show finance/payments/plan` for payments or
`scopebench template generate health/medical_data` for medical-data flows.

### `POST /evaluate_session`

Evaluate a multi-agent session with a shared global contract, per-agent plans, global budget enforcement, and cross-agent scope laundering detection.

Example:

```bash
curl -s http://localhost:8080/evaluate_session \
  -H 'content-type: application/json' \
  -d @examples/multi_agent_session.json
```

Response includes:

- `decision` (global)
- `per_agent.<agent_id>.aggregate` and `per_agent.<agent_id>.ledger`
- `global.aggregate` and `global.ledger`
- `laundering_signals` for cross-agent envelope bypass attempts
- `dashboard.per_agent` / `dashboard.global` budget consumption + utilization summaries


---


## Plugin Ecosystem

ScopeBench supports runtime plugin bundles for third-party tool catalogs, dataset extensions, scoring metadata, and signed policy-rule contributions.

- Configure plugin loading with `SCOPEBENCH_PLUGIN_DIRS` and `SCOPEBENCH_PLUGIN_KEYS_JSON`.
- `GET /tools` returns merged built-in + plugin tools and contribution metadata (`tool_categories`, `effects_mappings`, `scoring_axes`, `policy_rules`) plus signature status.
- `GET /cases` returns built-in benchmark cases plus plugin-provided dataset extensions.

See [`docs/plugins.md`](docs/plugins.md) for authoring/signing/publishing guidance and [`docs/plugin_marketplace.yaml`](docs/plugin_marketplace.yaml) for community domain listings (robotics, biotech, supply-chain, and more).

Plugin scaffolding and linting are available via CLI and UI:

- `scopebench plugin-generate --out plugin.yaml` (interactive skeleton + signing)
- `scopebench plugin-lint plugin.yaml` (schema lint checks)
- `/ui` → **Plugin Authoring Wizard** for guided bundle generation

---

## Examples Included

The `examples/` folder includes scenarios such as:

- phone charging overreach
- coding bugfix
- SWE fix workflow
- coding refactor
- test stabilization
- operations key rotation

These are useful for demos, CI checks, and policy tuning.

You can also list generated benchmark cases via API:

```bash
curl -s http://localhost:8080/cases | jq '.cases[0]'
```

---

## Benchmark Dataset & CI Gates

ScopeBench ships a large golden dataset at `scopebench/bench/cases/examples.jsonl` with 500+ curated cases across SWE, Ops, Finance, Health, and Marketing.

Each case includes:
- expected `ALLOW / ASK / DENY` decision
- `expected_rationale` for auditability
- step-level `expected_step_vectors` over all 10 scope axes

CI runs explicit dataset gates:
- dataset size/domain distribution sanity check
- exact decision-match regression check (golden labels vs evaluator)

See `docs/rubric_v1.md` for annotator guidance and scoring calibration.

---

## Testing

Run the test suite:

```bash
pytest
```

Run a narrower, faster loop while iterating:

```bash
pytest tests/test_cli.py -q
```

---

## Integrations SDK (Python)

Use `scopebench.integrations.guard(...)` to intercept agent plans before execution and receive:

- decision + effective decision
- exceeded/asked axis signals
- recommended plan patch transformations

The SDK also includes lightweight adapters for popular agent framework plan/message payloads:

- `from_langchain_plan(...)`
- `from_autogen_messages(...)`
- `from_airflow_dag(...)`
- `from_prefect_tasks(...)`
- `from_dagster_ops(...)`
- task decorators: `airflow_task(...)`, `prefect_task(...)`, `dagster_op(...)`

See `docs/integrations/python_sdk.md` for usage examples.

---

## Project Structure

- `scopebench/contracts.py` — task contract schema and presets
- `scopebench/plan.py` — plan DAG schema
- `scopebench/scoring/` — axis scoring and aggregation
- `scopebench/policy/` — policy engine for `ALLOW/ASK/DENY`
- `scopebench/runtime/guard.py` — orchestration entrypoint
- `scopebench/server/api.py` — FastAPI app and response shaping
- `scopebench/bench/` — benchmark/telemetry utilities
- `scopebench/tool_registry.yaml` — tool metadata used by scoring/policy
- `examples/` — runnable scenario inputs (including `effects_v1` examples)
- `docs/effects.md` — planner guide for structured effects and precedence
- `docs/BLUEPRINT.md` — long-term architecture and roadmap

---

## Development

```bash
ruff check .
pytest
```

Contributions are welcome. See `CONTRIBUTING.md`.

---

## License

MIT — see `LICENSE`.
