from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml

from scopebench.scoring.axes import SCOPE_AXES

_MAGNITUDE_ORDER = ["none", "low", "medium", "high", "extreme"]


@dataclass
class ContinuousLearningReport:
    telemetry_rows: int
    benchmark_rows: int
    pattern_counts: Dict[str, int]
    risky_tool_pairs: List[Tuple[Tuple[str, str], int]]
    prior_recommendations: Dict[str, Dict[str, float]]
    default_effect_recommendations: Dict[str, Dict[str, Any]]


def _iter_jsonl(path: Path) -> Iterable[dict]:
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        yield json.loads(line)


def _extract_tools(row: Dict[str, Any]) -> List[str]:
    policy_input = row.get("policy_input") if isinstance(row.get("policy_input"), dict) else {}
    plan = policy_input.get("plan") if isinstance(policy_input.get("plan"), dict) else {}
    steps = plan.get("steps") if isinstance(plan.get("steps"), list) else []

    tools: List[str] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        tool = step.get("tool")
        if isinstance(tool, str) and tool:
            tools.append(tool)

    if not tools:
        vectors = policy_input.get("vectors") if isinstance(policy_input.get("vectors"), list) else []
        for vec in vectors:
            if not isinstance(vec, dict):
                continue
            tool = vec.get("tool")
            if isinstance(tool, str) and tool:
                tools.append(tool)

    return sorted(set(tools))


def _risk_weight(row: Dict[str, Any]) -> float:
    decision = row.get("decision")
    if decision == "DENY":
        return 1.0
    if decision == "ASK":
        return 0.7
    return 0.2


def _axis_pressure(row: Dict[str, Any], axis: str) -> float:
    aggregate = row.get("aggregate") if isinstance(row.get("aggregate"), dict) else {}
    asked = row.get("asked") if isinstance(row.get("asked"), dict) else {}
    exceeded = row.get("exceeded") if isinstance(row.get("exceeded"), dict) else {}

    agg = float(aggregate.get(axis, 0.0) or 0.0)
    ask = float(asked.get(axis, 0.0) or 0.0)
    exc = 0.0
    exc_payload = exceeded.get(axis)
    if isinstance(exc_payload, dict):
        exc = float(exc_payload.get("value", 0.0) or 0.0)

    return max(agg, ask, exc)


def _bucket_for_axis(axis: str) -> str:
    if axis == "resource_intensity":
        return "resources"
    if axis == "legal_exposure":
        return "legal"
    if axis == "stakeholder_radius":
        return "stakeholders"
    if axis == "irreversibility":
        return "irreversible_actions"
    if axis == "spatial":
        return "geo_scope"
    if axis == "temporal":
        return "time_horizon"
    return "stakeholders"


def _magnitude_for_value(value: float) -> str:
    if value >= 0.9:
        return "extreme"
    if value >= 0.7:
        return "high"
    if value >= 0.45:
        return "medium"
    if value >= 0.2:
        return "low"
    return "none"


def _max_magnitude(current: str, proposed: str) -> str:
    try:
        return _MAGNITUDE_ORDER[max(_MAGNITUDE_ORDER.index(current), _MAGNITUDE_ORDER.index(proposed))]
    except ValueError:
        return proposed


def analyze_continuous_learning(
    telemetry_path: Path,
    benchmark_path: Path | None,
    registry_path: Path,
    min_pair_support: int = 2,
) -> ContinuousLearningReport:
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    tools_cfg = registry.get("tools") if isinstance(registry, dict) else {}
    if not isinstance(tools_cfg, dict):
        tools_cfg = {}

    pair_counts: Counter[Tuple[str, str]] = Counter()
    pattern_counts: Counter[str] = Counter()
    weighted_axis: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    weights: Dict[str, float] = defaultdict(float)

    telemetry_rows = 0
    for row in _iter_jsonl(telemetry_path):
        telemetry_rows += 1
        tools = _extract_tools(row)
        risk = _risk_weight(row)
        feedback = row.get("feedback") if isinstance(row.get("feedback"), dict) else {}
        outcome = feedback.get("outcome")
        if isinstance(outcome, str) and outcome:
            pattern_counts[f"outcome:{outcome}"] += 1

        for i in range(len(tools)):
            for j in range(i + 1, len(tools)):
                pair_counts[(tools[i], tools[j])] += 1

        for tool in tools:
            weights[tool] += risk
            for axis in SCOPE_AXES:
                weighted_axis[tool][axis] += _axis_pressure(row, axis) * risk

    benchmark_rows = 0
    if benchmark_path is not None and benchmark_path.exists():
        for row in _iter_jsonl(benchmark_path):
            benchmark_rows += 1
            plan = row.get("plan") if isinstance(row.get("plan"), dict) else {}
            metadata = plan.get("metadata") if isinstance(plan.get("metadata"), dict) else {}
            tactics = metadata.get("adversarial_tactics") if isinstance(metadata.get("adversarial_tactics"), list) else []
            for tactic in tactics:
                if isinstance(tactic, str) and tactic:
                    pattern_counts[f"benchmark:{tactic}"] += 1

            steps = plan.get("steps") if isinstance(plan.get("steps"), list) else []
            tools = sorted(
                {
                    step.get("tool")
                    for step in steps
                    if isinstance(step, dict) and isinstance(step.get("tool"), str) and step.get("tool")
                }
            )
            for i in range(len(tools)):
                for j in range(i + 1, len(tools)):
                    pair_counts[(tools[i], tools[j])] += 1

    prior_recommendations: Dict[str, Dict[str, float]] = {}
    default_effect_recommendations: Dict[str, Dict[str, Any]] = {}

    for tool, axis_totals in weighted_axis.items():
        base = tools_cfg.get(tool) if isinstance(tools_cfg.get(tool), dict) else {}
        base_priors = base.get("priors") if isinstance(base.get("priors"), dict) else {}
        total_weight = max(weights[tool], 1e-9)

        rec_priors: Dict[str, float] = {}
        rec_effects: Dict[str, Any] = {}
        for axis in SCOPE_AXES:
            observed = axis_totals[axis] / total_weight
            existing = float(base_priors.get(axis, 0.0) or 0.0)
            rec = round(min(1.0, max(0.0, (0.6 * existing) + (0.4 * observed))), 3)
            rec_priors[axis] = rec

            bucket = _bucket_for_axis(axis)
            proposed_mag = _magnitude_for_value(observed)
            if proposed_mag == "none":
                continue
            current_mag = "none"
            current_effects = base.get("default_effects") if isinstance(base.get("default_effects"), dict) else {}
            if isinstance(current_effects.get(bucket), dict):
                cm = current_effects[bucket].get("magnitude")
                if isinstance(cm, str):
                    current_mag = cm

            final_mag = _max_magnitude(current_mag, proposed_mag)
            rec_effects[bucket] = {
                "magnitude": final_mag,
                "rationale": f"telemetry-derived from {telemetry_rows} runs",
            }

        prior_recommendations[tool] = rec_priors
        if rec_effects:
            default_effect_recommendations[tool] = rec_effects

    risky_pairs = [item for item in pair_counts.most_common() if item[1] >= min_pair_support]

    return ContinuousLearningReport(
        telemetry_rows=telemetry_rows,
        benchmark_rows=benchmark_rows,
        pattern_counts=dict(pattern_counts),
        risky_tool_pairs=risky_pairs,
        prior_recommendations=prior_recommendations,
        default_effect_recommendations=default_effect_recommendations,
    )


def apply_learning_to_registry(report: ContinuousLearningReport, registry_path: Path) -> Dict[str, Any]:
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    tools_cfg = registry.get("tools") if isinstance(registry, dict) else {}
    if not isinstance(tools_cfg, dict):
        raise ValueError("tool registry must contain a 'tools' mapping")

    for tool, priors in report.prior_recommendations.items():
        tool_cfg = tools_cfg.get(tool)
        if not isinstance(tool_cfg, dict):
            continue
        tool_cfg["priors"] = dict(priors)

    for tool, effects in report.default_effect_recommendations.items():
        tool_cfg = tools_cfg.get(tool)
        if not isinstance(tool_cfg, dict):
            continue
        default_effects = tool_cfg.get("default_effects")
        if not isinstance(default_effects, dict):
            default_effects = {}
            tool_cfg["default_effects"] = default_effects

        for key, payload in effects.items():
            current = default_effects.get(key)
            if not isinstance(current, dict):
                default_effects[key] = dict(payload)
                continue
            merged = dict(current)
            merged.update(payload)
            default_effects[key] = merged

    registry_path.write_text(yaml.safe_dump(registry, sort_keys=False), encoding="utf-8")
    return registry


def render_learning_report_markdown(report: ContinuousLearningReport) -> str:
    lines = [
        "# Continuous learning report",
        "",
        f"- Telemetry rows analyzed: **{report.telemetry_rows}**",
        f"- Benchmark rows analyzed: **{report.benchmark_rows}**",
        "",
        "## Emerging scope-laundering patterns",
    ]

    if report.pattern_counts:
        for pattern, count in sorted(report.pattern_counts.items(), key=lambda item: item[1], reverse=True)[:15]:
            lines.append(f"- `{pattern}`: {count}")
    else:
        lines.append("- No pattern signals found.")

    lines.append("")
    lines.append("## High-risk tool combinations")
    if report.risky_tool_pairs:
        for (left, right), count in report.risky_tool_pairs[:15]:
            lines.append(f"- `{left}` + `{right}`: {count} occurrences")
    else:
        lines.append("- No high-support pairs found.")

    lines.append("")
    lines.append("## Proposed updates")
    for tool in sorted(report.prior_recommendations):
        lines.append(f"### `{tool}`")
        priors = report.prior_recommendations[tool]
        lines.append("- Priors:")
        for axis, value in sorted(priors.items()):
            lines.append(f"  - `{axis}`: {value:.3f}")
        effects = report.default_effect_recommendations.get(tool, {})
        if effects:
            lines.append("- Default effects:")
            for effect_name, payload in sorted(effects.items()):
                magnitude = payload.get("magnitude", "unknown")
                lines.append(f"  - `{effect_name}`: `{magnitude}`")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
