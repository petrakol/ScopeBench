# Multi-Agent Session Evaluation

ScopeBench supports multi-agent governance through `POST /evaluate_session`.

## What it enforces

- Schema validation that each plan declares a known `agent_id`.
- Per-agent scope evaluation (aggregate + decision).
- Global envelope evaluation across all agents.
- Global budget ledger checks (`cost_usd`, `time_horizon_days`, `max_tool_calls`) that can trigger `ASK` even when each individual agent remains within its own budget.

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
