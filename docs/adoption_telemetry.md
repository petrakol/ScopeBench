# Adoption Telemetry Baseline (Opt-in)

This schema is a lightweight, opt-in format for adoption measurement.

## Event schema

```json
{
  "event": "first_run_completed",
  "timestamp_utc": "2026-02-23T00:00:00Z",
  "actor": "local_user",
  "properties": {
    "decision": "ASK",
    "n_steps": 3,
    "used_plugin": false
  }
}
```

## Recommended event names

- `install_completed`
- `first_run_started`
- `first_run_completed`
- `plugin_enabled`
- `repeat_run_completed`

## Funnel

Track conversion from `install_completed` -> `first_run_completed` -> `repeat_run_completed`.
