# Interactive Quickstart Walkthrough

This guide is a hands-on companion to `scopebench quickstart` and `scopebench coding-quickstart`.
It is designed for workshop, onboarding, and self-serve experimentation.

## What you will learn

By the end of this walkthrough, you will be able to:

1. Install ScopeBench in a local virtual environment.
2. Run both quickstart commands and inspect decisions.
3. Choose and compare contract templates.
4. Edit a plan and observe decision changes.
5. Evaluate plans over HTTP via `POST /evaluate`.
6. Integrate ScopeBench into Python workflows with the integrations SDK.

## Prerequisites

- Python 3.10+
- `pip`
- `curl`
- Optional: `jq` for pretty JSON output

---

## 1) Install and verify

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
scopebench --help
```

---

## 2) Run quickstarts

### Canonical quickstart

```bash
scopebench quickstart
scopebench quickstart --json | jq
```

### Coding quickstart

```bash
scopebench coding-quickstart
scopebench coding-quickstart --json | jq
```

What to inspect:

- `decision` (`ALLOW`, `ASK`, or `DENY`)
- `reasons` and axis-level limit breaches
- differences between generic and coding-specific policies

---

## 3) Select templates and compare strictness

Inspect available templates:

```bash
scopebench list-templates
```

Create test contracts (example):

```yaml
# personal.contract.yaml
goal: "Fix flaky test"
preset: personal
```

```yaml
# enterprise.contract.yaml
goal: "Fix flaky test"
preset: enterprise
```

Re-run the same plan against each contract and compare output:

```bash
scopebench run personal.contract.yaml examples/coding_small.patch.plan.yaml --json | jq '.decision, .scores'
scopebench run enterprise.contract.yaml examples/coding_small.patch.plan.yaml --json | jq '.decision, .scores'
```

---

## 4) Edit a plan and observe policy behavior

Start from an existing plan and fork it:

```bash
cp examples/phone_charge.plan.yaml /tmp/phone_charge.plan.yaml
```

Run baseline:

```bash
scopebench run examples/phone_charge.contract.yaml /tmp/phone_charge.plan.yaml --json | jq '.decision, .reasons'
```

Now iteratively reduce scope in `/tmp/phone_charge.plan.yaml`:

- remove broad or irreversible actions
- minimize number of steps
- narrow tool choice to local, low-impact tools
- limit external side effects

Re-run after each edit until the plan moves toward `ASK` or `ALLOW`.

---

## 5) Use the API interactively

Start the server:

```bash
scopebench serve --host 0.0.0.0 --port 8080
```

In another shell, use the web docs and HTTP endpoints:

- Open API docs: `http://localhost:8080/docs`
- Health: `curl -s http://localhost:8080/health | jq`
- Templates: `curl -s http://localhost:8080/templates | jq '.count'`

Evaluate a plan:

```bash
curl -s http://localhost:8080/evaluate \
  -H 'content-type: application/json' \
  -d '{
    "contract": {"goal": "Fix failing test", "preset": "team"},
    "plan": {
      "task": "Fix failing test",
      "steps": [
        {"id":"1","description":"Read failing test","tool":"git_read"},
        {"id":"2","description":"Apply minimal patch","tool":"git_patch","depends_on":["1"]},
        {"id":"3","description":"Run targeted test","tool":"pytest","depends_on":["2"]}
      ]
    },
    "include_summary": true,
    "include_next_steps": true,
    "include_patch": true,
    "include_telemetry": true
  }' | jq
```

---

## 6) Python integration walkthrough

Use the Python SDK wrappers to embed checks directly in pipelines and agents:

```python
from scopebench.integrations.sdk import evaluate_plan

contract = {
    "goal": "Fix failing unit test",
    "preset": "team",
}

plan = {
    "task": "Fix failing unit test",
    "steps": [
        {"id": "1", "description": "Read failing test", "tool": "git_read"},
        {"id": "2", "description": "Apply minimal patch", "tool": "git_patch", "depends_on": ["1"]},
        {"id": "3", "description": "Run targeted test", "tool": "pytest", "depends_on": ["2"]},
    ],
}

result = evaluate_plan(contract=contract, plan=plan, include_summary=True)
print(result["decision"], result.get("summary"))
```

You can place this call:

- before autonomous tool execution,
- before PR creation,
- or inside CI checks for plan-level policy conformance.

---

## Next steps

- Run the companion notebook: `docs/notebooks/scopebench_quickstart_tutorial.ipynb`
- Explore templates in `scopebench/templates/`
- Try red-team and calibration flows from the main README
