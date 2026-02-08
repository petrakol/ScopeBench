from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import yaml

from scopebench.scoring.axes import AxisScore, ScopeVector


@dataclass(frozen=True)
class CausalRule:
    category: str
    axis_minimums: Dict[str, float]
    rationale: str


_CAUSAL_RULE_CACHE: Optional[Dict[str, CausalRule]] = None


def _default_rules_path() -> Path:
    return Path(__file__).resolve().parent / "causal_rules.yaml"


def load_causal_rules(path: Optional[Path] = None) -> Dict[str, CausalRule]:
    rules_path = path or _default_rules_path()
    raw = yaml.safe_load(rules_path.read_text(encoding="utf-8")) or {}
    rules: Dict[str, CausalRule] = {}
    for key, payload in (raw.get("categories") or {}).items():
        payload = payload or {}
        rules[key.strip().lower()] = CausalRule(
            category=payload.get("category", key),
            axis_minimums=payload.get("axis_minimums") or {},
            rationale=payload.get("rationale", ""),
        )
    return rules


def get_causal_rule(category: Optional[str]) -> Optional[CausalRule]:
    global _CAUSAL_RULE_CACHE
    if not category:
        return None
    if _CAUSAL_RULE_CACHE is None:
        _CAUSAL_RULE_CACHE = load_causal_rules()
    return _CAUSAL_RULE_CACHE.get(category.strip().lower())


def apply_causal_adjustments(vector: ScopeVector, tool_category: Optional[str]) -> ScopeVector:
    rule = get_causal_rule(tool_category)
    if not rule:
        return vector

    updates: Dict[str, AxisScore] = {}
    for axis, minimum in rule.axis_minimums.items():
        if axis not in vector.axes:
            continue
        current = getattr(vector, axis)
        if current.value >= float(minimum):
            continue
        rationale = current.rationale
        if rule.rationale:
            rationale = f"{rationale}; {rule.rationale}".strip("; ")
        updates[axis] = current.model_copy(update={"value": float(minimum), "rationale": rationale})
    if not updates:
        return vector
    return vector.model_copy(update=updates)


def list_causal_rules() -> Dict[str, CausalRule]:
    global _CAUSAL_RULE_CACHE
    if _CAUSAL_RULE_CACHE is None:
        _CAUSAL_RULE_CACHE = load_causal_rules()
    return dict(_CAUSAL_RULE_CACHE)
