# Interactive Quickstart Walkthrough

This walkthrough is designed for new ScopeBench users who want to learn by doing.
It guides you through the full lifecycle:

1. Choose a template/contract preset.
2. Edit a plan and observe policy changes.
3. Annotate plan effects.
4. Evaluate through CLI and API.

## Learning path map

Use this walkthrough together with the companion notebook:

- Markdown (this file): best for terminal-first practice and copy/paste commands.
- Notebook: `docs/notebooks/scopebench_quickstart_tutorial.ipynb` for an executable, cell-by-cell flow.

The quickstart focuses on template selection, plan editing, effect annotation, and API evaluation.
For dataset contribution, calibration tuning, and plugin creation, continue to the advanced lab in step 7.

---

## What you will learn

By the end of this tutorial, you will be able to:

- Inspect and pick templates for your domain.
- Run plan checks with `scopebench run`.
- Make constrained plan edits and re-check decisions.
- Auto-suggest and review effect annotations.
- Evaluate plans through `POST /evaluate`.

## Prerequisites

- Python 3.10+
- `pip`
- `curl`
- Optional: `jq` for readable JSON

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

## 2) Explore templates and select a starting point

List available templates:

```bash
scopebench list-templates
```

Pick one template and inspect it. For example:

```bash
cat scopebench/templates/swe/contract.yaml
cat scopebench/templates/swe/plan.yaml
```

Run a template-backed example plan:

```bash
scopebench run examples/coding_bugfix.contract.yaml examples/coding_bugfix.plan.yaml --json | jq '.decision, .scores, .reasons'
```

---

## 3) Edit a plan and observe scope changes

Create an editable copy:

```bash
cp examples/coding_bugfix.plan.yaml /tmp/coding_bugfix.plan.yaml
```

Run baseline evaluation:

```bash
scopebench run examples/coding_bugfix.contract.yaml /tmp/coding_bugfix.plan.yaml --json | jq '.decision, .reasons'
```

Now edit `/tmp/coding_bugfix.plan.yaml` to change scope and re-run.

Try:

- removing broad/high-impact actions,
- reducing number of steps,
- replacing expansive tools with narrower ones.

Re-evaluate after each edit:

```bash
scopebench run examples/coding_bugfix.contract.yaml /tmp/coding_bugfix.plan.yaml --json | jq '.decision, .reasons, .scores'
```

---

## 4) Annotate effects for richer policy signals

Generate suggested effect annotations:

```bash
scopebench suggest-effects /tmp/coding_bugfix.plan.yaml --json | jq
```

Apply annotations in-place when satisfied:

```bash
scopebench suggest-effects /tmp/coding_bugfix.plan.yaml --in-place
```

Re-run evaluation with the updated plan:

```bash
scopebench run examples/coding_bugfix.contract.yaml /tmp/coding_bugfix.plan.yaml --json | jq '.decision, .scores, .reasons'
```

---

## 5) Evaluate with the HTTP API

Start the server:

```bash
scopebench serve --host 0.0.0.0 --port 8080
```

In another shell, inspect API endpoints:

```bash
curl -s http://localhost:8080/health | jq
curl -s http://localhost:8080/templates | jq '.count'
curl -s http://localhost:8080/tools | jq '.count'
curl -s http://localhost:8080/cases | jq '.count'
```

Submit an evaluation request:

```bash
curl -s http://localhost:8080/evaluate \
  -H 'content-type: application/json' \
  -d '{
    "contract": {"goal": "Fix failing unit test", "preset": "team"},
    "plan": {
      "task": "Fix failing unit test",
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
  }' | jq '.decision, .summary, .next_steps, .plan_patch_suggestion'
```

---

## 6) Continue with the notebook tutorial

Open and run the companion notebook for the same workflow in a single interactive environment:

- `docs/notebooks/scopebench_quickstart_tutorial.ipynb`


---

## 7) Continue into advanced onboarding labs

After finishing this quickstart, continue with:

- `docs/tutorials/dataset_calibration_plugin_lab.md` for dataset contribution, calibration tuning, and plugin authoring workflows.
- `docs/notebooks/scopebench_dataset_calibration_plugin_lab.ipynb` for a notebook version of the same lab.

These are also linked from the Workbench UI "Learn" section for in-product discovery.
