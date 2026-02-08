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

## Explicit mathematics (lightweight + composable)

ScopeBench is not an optimization system. It is an envelope enforcer with explicit, legible math that resists gaming. The math below is intentionally simple, direct, and translates 1:1 into code.

### 1) Vector spaces & norms (core aggregation)
- Scope is a vector:
  - `S = [spatial, temporal, depth, irreversibility, resource, legal, dependency, stakeholder, power, uncertainty]`
- **Primary aggregation is L∞** to prevent averaging away a catastrophic axis:
  - `||S||∞ = max_i S_i`
- Optional diagnostics only (never enforcement):
  - L1 summaries: `||S||1 = Σ_i S_i`
  - weighted sums for visualization

**Code sketch**
```
scope_max = max(scope_vector.values())
decision = "DENY" if scope_max > threshold else ...
```

### 2) Monotonic accumulation (anti-laundering)
- Scope must be non-decreasing across steps.
- Irreversibility introduces hysteresis (no full rollback).

**Minimal formalism**
```
C_t = C_{t-1} + f(S_t)  (f is monotonic)
```

**Code sketch**
```
cumulative_scope[axis] += step_scope[axis] * irreversibility_factor
```

### 3) Thresholded decision theory (not optimization)
- **Piecewise thresholds** define decision regions per axis:
```
if S_i >= deny_i → DENY
elif S_i >= ask_i → ASK
else → ALLOW
```
- This non-smooth logic is **legible** and **resistant to adversarial optimization**.

### 4) Uncertainty contraction (selective abstention)
- Let `u ∈ [0,1]` be uncertainty.
- Effective thresholds shrink as uncertainty grows:
```
theta_eff = theta_base * (1 - u)
```
```
effective_threshold = base_threshold * (1 - uncertainty)
```

### 5) Knee detection (diminishing returns)
- Detect over-optimization by marginal gain:
```
if delta_benefit / delta_cost < epsilon:
    flag_overreach()
```
- No curve fitting required; change-rate detection is enough.

### 6) Partial orders (power & dominance)
- Power is not purely scalar. Some actions are incomparable and should be classified:
```
if action.category in HARD_POWER_ACTIONS:
    decision = "ASK"
```
- Use **ordered classes** for escalation, not numeric normalization.

### 7) Graph theory (plan-level reasoning)
- Plans are DAGs; scope hides in **paths**, not just nodes.
```
Scope(path) = Σ S(node_i)
```
- Compute max path scope via topological traversal and accumulation.

**Code sketch**
```
for node in topo_order(plan):
    for axis in axes:
        path_scope[node][axis] = max(
            path_scope[pred][axis] for pred in predecessors(node)
        ) + scope[node][axis]
max_path_scope = max(
    max(path_scope[node].values()) for node in plan.nodes
)
```

### 8) Explicit exclusions (what we do *not* use)
- Utility maximization
- Gradient descent
- Reinforcement learning
- Learned reward aggregation

ScopeBench remains a **bounded, conservative, envelope-enforcing system**.

### 9) Summary mapping (math → code → purpose)
| Math | Code pattern | Purpose |
| --- | --- | --- |
| L∞ norm | `max(scope_vector.values())` | Axis dominance without averaging |
| Monotonic sums | `+=` with irreversibility | Anti-laundering |
| Piecewise thresholds | `if/elif/else` | Legible decisions |
| Uncertainty contraction | `theta * (1 - u)` | Conservative envelopes |
| Knee detection | `delta_benefit / delta_cost` | Stop over-optimization |
| Partial orders | category checks | Non-comparable power actions |
| DAG traversal | topological accumulation | Path-level risk |

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
