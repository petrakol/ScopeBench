# Multi-Agent Session Evaluation

ScopeBench supports multi-agent governance through `POST /evaluate_session`.

## What it enforces

- Schema validation that each plan declares a known `agent_id`.
- Per-agent scope evaluation (aggregate + decision).
- Global envelope evaluation across all agents.
- Global budget ledger checks (`cost_usd`, `time_horizon_days`, `max_tool_calls`) that can trigger `ASK` even when each individual agent remains within its own budget.
- Cross-agent scope laundering detection when aggregate session risk crosses the global envelope while no single agent crosses it alone.
- Dashboard-ready budget telemetry with per-agent/global budget consumption and utilization ratios.

## Run the example end-to-end

Start server:

```bash
scopebench serve --host 0.0.0.0 --port 8080
```

Evaluate the bundled session payload:

```bash
curl -s http://localhost:8080/evaluate_session \
  -H 'content-type: application/json' \
  -d @examples/multi_agent_session.json | python -m json.tool
```

The response includes:

- `decision`
- `per_agent.<agent_id>.aggregate`
- `global.aggregate`
- per-agent and global `ledger` sections with `budget`, `consumed`, `remaining`, and `exceeded`.
- `laundering_signals` entries that flag axes where global aggregate risk exceeds the shared envelope across agent boundaries.
- `dashboard.per_agent` and `dashboard.global` sections with `budget_consumption` and `budget_utilization` for team coordination dashboards.
