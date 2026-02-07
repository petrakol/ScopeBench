# ScopeBench

**ScopeBench** is an open-source MVP for **plan-level proportionality enforcement** ("scope alignment") for tool-using agents.

It targets a failure mode that is *not* "evil intent" or "wrong values", but **scale mismatch**:

> The agent is directionally correct, but acts at an *inappropriate magnitude*, e.g.  
> *"Charge my phone"* → *"Re-architect the power grid and deploy solar on another continent."*

ScopeBench provides:

- A **Task Contract** (human envelope) format with presets (Personal / Team / Enterprise / Regulated)
- A **Plan DAG** format (steps, tools, dependencies)
- A **Scope Vector** scoring engine across multiple axes (space/time/depth/irreversibility/etc.)
- **Cumulative scope accounting** (resists "scope laundering" across steps)
- A **policy engine** that returns `ALLOW / ASK / DENY` with a rationale
- **OpenTelemetry tracing** hooks (OTel-first, replay-friendly)
- A tiny starter **ScopeBench dataset** (JSONL) + tests

## Quickstart

### 1) Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

### 2) Run the demo (phone → solar overreach)

```bash
scopebench run examples/phone_charge.contract.yaml examples/phone_charge.plan.yaml
```

You should see a `DENY` decision with axis-wise reasons.

Quickstart shortcut:

```bash
scopebench quickstart
```

### 3) Start the API

```bash
scopebench serve --host 0.0.0.0 --port 8080
```

Then:

```bash
curl -s http://localhost:8080/health
```

The `/evaluate` response can include a short `summary` and `next_steps` (set `include_summary=true`) to support UI/agent UX flows.

## Repository layout

- `scopebench/contracts.py` — contract schema + presets
- `scopebench/plan.py` — plan DAG schema
- `scopebench/scoring/` — scoring rules and aggregation
- `scopebench/policy/` — policy engine (`ALLOW/ASK/DENY`)
- `scopebench/runtime/guard.py` — the main orchestrator
- `scopebench/server/api.py` — FastAPI service
- `scopebench/bench/` — dataset schema and rubric
- `examples/` — runnable example contracts/plans
- `docs/BLUEPRINT.md` — MVP→full system blueprint and roadmap

## License

MIT — see `LICENSE`.
