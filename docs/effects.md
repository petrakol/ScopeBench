# Effects (`effects_v1`) guide

ScopeBench planners can attach structured effects to each `PlanStep` using `effects_v1`.

## Schema

Each step can include:

- `resources`
- `legal`
- `stakeholders`
- `irreversible_actions`
- `geo_scope`
- `time_horizon`

Each category has:

- `magnitude`: one of `none | low | medium | high | extreme`
- optional `rationale`
- optional category-specific list fields (`kinds`, `regimes`, `groups`, `actions`, `regions`, `horizons`)

Example:

```yaml
steps:
  - id: "1"
    description: "Proceed with the requested operation."
    tool: "analysis"
    effects:
      version: effects_v1
      legal:
        magnitude: high
        regimes: [gdpr, pci]
        rationale: "contains regulated data"
      stakeholders:
        magnitude: medium
        groups: [customers]
```

## Scoring precedence

For axes covered by effects, scoring precedence is:

1. `effects` (strongest)
2. tool priors
3. keyword heuristics

Effects-covered axes:

- `spatial` (from `geo_scope`)
- `temporal` (from `time_horizon`)
- `irreversibility` (from `irreversible_actions`)
- `resource_intensity` (from `resources`)
- `legal_exposure` (from `legal`)
- `stakeholder_radius` (from `stakeholders`)

## Tool default effects

If a step does not specify `effects`, ScopeBench can infer defaults from `tool_registry.yaml` `default_effects`.
These defaults are merged into the step at scoring time.

This lets neutral descriptions still reflect macro consequences implied by the selected tool.
