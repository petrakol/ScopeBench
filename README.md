<img width="973" height="209" alt="ScopeBench_w copy" src="https://github.com/user-attachments/assets/066d740e-f8ad-4876-89c5-48604a5cc7a0" />

# ScopeBench

**ScopeBench** is an open-source Python toolkit for **plan-level proportionality enforcement** ("scope alignment") in tool-using agents.

It is designed for a specific failure mode:

> The agent is directionally correct, but acts at the wrong *magnitude*.
>
> Example: *"Charge my phone"* → *"Re-architect the power grid and deploy solar internationally."*

Instead of only asking *"is this intent safe?"*, ScopeBench asks *"is this plan proportionate to the task contract?"*.

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
- CLI modes for one-off runs, quickstarts, weekly calibration, and API serving.
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

---

## Quickstart

### 1) Run the canonical overreach example

```bash
scopebench run examples/phone_charge.contract.yaml examples/phone_charge.plan.yaml
```

Expected result: a `DENY` decision with axis-specific reasons.

### 2) Try built-in shortcuts

```bash
scopebench quickstart
scopebench coding-quickstart
```

### 3) Start the API server

```bash
scopebench serve --host 0.0.0.0 --port 8080
```

Health check:

```bash
curl -s http://localhost:8080/health
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
scopebench weekly-calibrate <telemetry.jsonl> [--json]
scopebench serve [--host 127.0.0.1] [--port 8080] [--policy-backend python|opa|cedar]
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


### `POST /evaluate_session`

Evaluate a multi-agent session with a shared global contract, per-agent plans, and global budget enforcement.

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

---

## Testing

Run the test suite:

```bash
pytest
```

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
- `examples/` — runnable scenario inputs
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
