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
- `macro_consequences`

Each category has:

- `magnitude`: one of `none | low | medium | high | extreme`
- optional `rationale`
- optional category-specific list fields (`kinds`, `regimes`, `groups`, `actions`, `regions`, `horizons`)

`macro_consequences` entries have:

- `concept`: knowledge-graph concept id
- `channel`: transmission channel (e.g., `climate`, `regulatory`, `social`)
- `confidence`: value in `[0,1]`
- optional `rationale`

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
      macro_consequences:
        - concept: public_trust
          channel: social
          confidence: 0.74
          rationale: "kg:stakeholders (broad trust and legitimacy impact)"
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


## Causal abstraction layer

ScopeBench now supports a causal abstraction layer in `effects_v1`:

- Every tool in `tool_registry.yaml` declares `default_effects` as structured `effects_v1` fields.
- Scoring infers additional `macro_consequences` from step text using a built-in knowledge-graph mapping (concept -> effect axis + channel).
- For covered axes, precedence remains: explicit effects > macro consequence inference > tool priors > keywords.

This means scoring can prioritize declared effects and causal consequence mappings over tool priors and keyword-only heuristics.
