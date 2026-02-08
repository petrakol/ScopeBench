from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Set

import yaml

from pydantic import BaseModel, Field


class DomainTemplate(BaseModel):
    name: str
    description: str
    forbidden_tool_categories: Set[str] = Field(default_factory=set)
    escalation_tool_categories: Set[str] = Field(default_factory=set)
    thresholds: Dict[str, float] = Field(default_factory=dict)
    escalation: Dict[str, float] = Field(default_factory=dict)
    budgets: Dict[str, float] = Field(default_factory=dict)
    allowed_tools: Optional[Set[str]] = None
    notes: Dict[str, str] = Field(default_factory=dict)


_DOMAIN_TEMPLATE_CACHE: Optional[Dict[str, DomainTemplate]] = None


def _default_templates_path() -> Path:
    return Path(__file__).resolve().parent / "domain_templates.yaml"


def load_domain_templates(path: Optional[Path] = None) -> Dict[str, DomainTemplate]:
    templates_path = path or _default_templates_path()
    raw = yaml.safe_load(templates_path.read_text(encoding="utf-8")) or {}
    templates: Dict[str, DomainTemplate] = {}
    for key, payload in (raw.get("domains") or {}).items():
        payload = payload or {}
        templates[key.strip().lower()] = DomainTemplate(
            name=payload.get("name", key),
            description=payload.get("description", ""),
            forbidden_tool_categories=set(payload.get("forbidden_tool_categories") or []),
            escalation_tool_categories=set(payload.get("escalation_tool_categories") or []),
            thresholds=payload.get("thresholds") or {},
            escalation=payload.get("escalation") or {},
            budgets=payload.get("budgets") or {},
            allowed_tools=set(payload.get("allowed_tools") or []) or None,
            notes=payload.get("notes") or {},
        )
    return templates


def get_domain_template(name: Optional[str]) -> Optional[DomainTemplate]:
    global _DOMAIN_TEMPLATE_CACHE
    if not name:
        return None
    if _DOMAIN_TEMPLATE_CACHE is None:
        _DOMAIN_TEMPLATE_CACHE = load_domain_templates()
    return _DOMAIN_TEMPLATE_CACHE.get(name.strip().lower())


def list_domain_templates() -> Dict[str, DomainTemplate]:
    global _DOMAIN_TEMPLATE_CACHE
    if _DOMAIN_TEMPLATE_CACHE is None:
        _DOMAIN_TEMPLATE_CACHE = load_domain_templates()
    return dict(_DOMAIN_TEMPLATE_CACHE)
