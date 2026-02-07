# ScopeBench blueprint (MVP → full system)

This document maps the research-driven "scale mismatch / scope alignment" concept to concrete system components.

## MVP: Proportionate Planner Gate (PPG)

- **Inputs**
  - `TaskContract` (human envelope)
  - `PlanDAG` (agent plan, tool calls)
  - `ToolRegistry` (tool metadata)
- **Core steps**
  1. Score each step: `ScopeVector`
  2. Aggregate: `ScopeAggregate` (cumulative footprint)
  3. Evaluate policy: `ALLOW/ASK/DENY`
  4. Emit traces: OTel spans for later debugging/replay

## Full system: Scope Alignment Control Plane (SACP)

Extensions:
- Domain template library (SWE, Ops, Finance, Health)
- Causal abstraction layer (tool actions → systems → macro consequences)
- "Knee of curve" instrumentation (detect diminishing returns)
- Red-team suite for scope laundering + policy gaming
- Multi-agent governance (global budgets + per-agent envelopes)

## Design constraints
- **No "evil" assumption**: treat overreach as optimizer dynamics, not malice.
- **No interpretability requirement**: evaluate *plans and effects*, not hidden chain-of-thought.
- **Conservatism under uncertainty**: when scoring confidence is low, the envelope tightens.

## What makes this beyond existing OSS
Existing guardrails typically validate:
- content (toxicity, policy, formatting)
- schema correctness (JSON, types)
- post-hoc eval (observability)

ScopeBench validates:
- **pre-execution** proportionality at the **plan** level
- **cumulative** footprint across steps (laundering-resistant)
- **policy-as-code** enforcement (OPA/Cedar-compatible conceptual model)

## Suggested next increments
1) Add OPA/Cedar integration for real policy enforcement
2) Expand tool registry and domain templates
3) Add calibrated uncertainty scoring (selective prediction/abstention)
4) Build ScopeBench dataset + labeling rubric to drive CI gates
