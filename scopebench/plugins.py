from __future__ import annotations

import hashlib
import hmac
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from scopebench.bench.dataset import ScopeBenchCase, _validate_case
from scopebench.scoring.axes import SCOPE_AXES
from scopebench.scoring.rules import ToolInfo


def _canonical_json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _read_bundle(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        loaded = yaml.safe_load(text)
    else:
        loaded = json.loads(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"plugin bundle '{path}' must be an object/map")
    return loaded


@dataclass(frozen=True)
class PluginBundle:
    name: str
    version: str
    publisher: str
    source_path: str
    signed: bool
    signature_valid: bool
    signature_error: Optional[str]
    tool_categories: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    effects_mappings: List[Dict[str, Any]] = field(default_factory=list)
    scoring_axes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    policy_rules: List[Dict[str, Any]] = field(default_factory=list)
    tools: Dict[str, ToolInfo] = field(default_factory=dict)
    cases: List[ScopeBenchCase] = field(default_factory=list)


class PluginManager:
    def __init__(self) -> None:
        self.tool_categories: Dict[str, Dict[str, Any]] = {}
        self.effects_mappings: List[Dict[str, Any]] = []
        self.scoring_axes: Dict[str, Dict[str, Any]] = {}
        self.policy_rules: List[Dict[str, Any]] = []
        self.tools: Dict[str, ToolInfo] = {}
        self.cases: List[ScopeBenchCase] = []
        self.bundles: List[PluginBundle] = []

    def register_tool_category(self, name: str, config: Optional[Dict[str, Any]] = None) -> None:
        self.tool_categories[name] = config or {}

    def register_effects_mapping(self, mapping: Dict[str, Any]) -> None:
        self.effects_mappings.append(dict(mapping))

    def register_scoring_axis(self, axis: str, config: Optional[Dict[str, Any]] = None) -> None:
        self.scoring_axes[axis] = config or {}

    def register_policy_rule(self, rule: Dict[str, Any]) -> None:
        self.policy_rules.append(dict(rule))

    def register_tool(self, info: ToolInfo) -> None:
        self.tools[info.tool] = info

    def register_case(self, case: ScopeBenchCase) -> None:
        self.cases.append(case)

    @classmethod
    def from_environment(cls) -> "PluginManager":
        manager = cls()
        plugin_dirs = [p.strip() for p in os.getenv("SCOPEBENCH_PLUGIN_DIRS", "").split(os.pathsep) if p.strip()]
        keyring = cls._load_keyring(os.getenv("SCOPEBENCH_PLUGIN_KEYS_JSON", ""))
        for raw_dir in plugin_dirs:
            path = Path(raw_dir)
            if not path.exists() or not path.is_dir():
                continue
            for bundle_path in sorted([*path.glob("*.json"), *path.glob("*.yaml"), *path.glob("*.yml")]):
                bundle = cls._load_bundle(bundle_path, keyring)
                manager.bundles.append(bundle)
                manager._apply_bundle(bundle)
        return manager

    @staticmethod
    def _load_keyring(payload: str) -> Dict[str, str]:
        if not payload.strip():
            return {}
        parsed = json.loads(payload)
        if not isinstance(parsed, dict):
            raise ValueError("SCOPEBENCH_PLUGIN_KEYS_JSON must be a JSON object of key_id -> shared_secret")
        out: Dict[str, str] = {}
        for key_id, secret in parsed.items():
            if isinstance(key_id, str) and isinstance(secret, str) and key_id and secret:
                out[key_id] = secret
        return out

    @classmethod
    def _load_bundle(cls, path: Path, keyring: Dict[str, str]) -> PluginBundle:
        raw = _read_bundle(path)
        name = str(raw.get("name") or path.stem)
        version = str(raw.get("version") or "0.0.0")
        publisher = str(raw.get("publisher") or "unknown")
        signature_payload = raw.get("signature") if isinstance(raw.get("signature"), dict) else None

        signed = signature_payload is not None
        signature_valid = False
        signature_error: Optional[str] = None

        unsigned_payload = dict(raw)
        unsigned_payload.pop("signature", None)

        if signed:
            signature_valid, signature_error = cls._verify_signature(signature_payload or {}, unsigned_payload, keyring)

        tools = cls._parse_tools(raw.get("tools") or {})
        cases = cls._parse_cases(raw.get("cases") or [])
        contributions = raw.get("contributions") if isinstance(raw.get("contributions"), dict) else {}

        tool_categories = contributions.get("tool_categories") if isinstance(contributions.get("tool_categories"), dict) else {}
        scoring_axes = contributions.get("scoring_axes") if isinstance(contributions.get("scoring_axes"), dict) else {}
        effects_mappings = contributions.get("effects_mappings") if isinstance(contributions.get("effects_mappings"), list) else []
        policy_rules = contributions.get("policy_rules") if isinstance(contributions.get("policy_rules"), list) else []

        return PluginBundle(
            name=name,
            version=version,
            publisher=publisher,
            source_path=str(path),
            signed=signed,
            signature_valid=signature_valid,
            signature_error=signature_error,
            tool_categories={str(k): v for k, v in tool_categories.items() if isinstance(v, dict)},
            effects_mappings=[m for m in effects_mappings if isinstance(m, dict)],
            scoring_axes={str(k): v for k, v in scoring_axes.items() if isinstance(v, dict)},
            policy_rules=[r for r in policy_rules if isinstance(r, dict)],
            tools=tools,
            cases=cases,
        )

    @staticmethod
    def _verify_signature(signature_payload: Dict[str, Any], unsigned_payload: Dict[str, Any], keyring: Dict[str, str]) -> tuple[bool, Optional[str]]:
        key_id = signature_payload.get("key_id")
        algo = signature_payload.get("algorithm")
        digest = signature_payload.get("digest")
        value = signature_payload.get("value")
        if not isinstance(key_id, str) or not key_id:
            return False, "missing signature.key_id"
        if algo != "hmac-sha256":
            return False, "unsupported signature.algorithm"
        if digest != "sha256":
            return False, "unsupported signature.digest"
        if not isinstance(value, str) or not value:
            return False, "missing signature.value"
        secret = keyring.get(key_id)
        if secret is None:
            return False, f"unknown key_id '{key_id}'"
        payload = _canonical_json(unsigned_payload).encode("utf-8")
        expected = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, value):
            return False, "signature mismatch"
        return True, None

    @staticmethod
    def _parse_tools(payload: Any) -> Dict[str, ToolInfo]:
        if not isinstance(payload, dict):
            return {}
        tools: Dict[str, ToolInfo] = {}
        for name, raw in payload.items():
            if not isinstance(name, str) or not isinstance(raw, dict):
                continue
            priors_raw = raw.get("priors") if isinstance(raw.get("priors"), dict) else {}
            priors: Dict[str, float] = {}
            for axis, value in priors_raw.items():
                if axis not in SCOPE_AXES:
                    continue
                if isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0:
                    priors[axis] = float(value)
            domains_raw = raw.get("domains") if isinstance(raw.get("domains"), list) else []
            domains = tuple(item for item in domains_raw if isinstance(item, str) and item)
            risk_class = raw.get("risk_class") if isinstance(raw.get("risk_class"), str) else "moderate"
            if risk_class not in {"low", "moderate", "high", "critical"}:
                risk_class = "moderate"
            default_effects = raw.get("default_effects") if isinstance(raw.get("default_effects"), dict) else {}
            tools[name] = ToolInfo(
                tool=name,
                category=str(raw.get("category") or "unknown"),
                priors=priors,
                domains=domains,
                risk_class=risk_class,
                default_effects=default_effects,
            )
        return tools

    @staticmethod
    def _parse_cases(payload: Any) -> List[ScopeBenchCase]:
        if not isinstance(payload, list):
            return []
        out: List[ScopeBenchCase] = []
        for idx, row in enumerate(payload, start=1):
            try:
                out.append(_validate_case(row, line_no=idx))
            except ValueError:
                continue
        return out

    def _apply_bundle(self, bundle: PluginBundle) -> None:
        for name, config in bundle.tool_categories.items():
            self.register_tool_category(name, config)
        for mapping in bundle.effects_mappings:
            self.register_effects_mapping(mapping)
        for axis, config in bundle.scoring_axes.items():
            self.register_scoring_axis(axis, config)
        if bundle.policy_rules and not (bundle.signed and bundle.signature_valid):
            # policy contributions must be signed to load.
            pass
        else:
            for rule in bundle.policy_rules:
                self.register_policy_rule(rule)
        for tool in bundle.tools.values():
            self.register_tool(tool)
        for case in bundle.cases:
            self.register_case(case)

    def bundles_payload(self) -> List[Dict[str, Any]]:
        payload: List[Dict[str, Any]] = []
        for bundle in self.bundles:
            payload.append(
                {
                    "name": bundle.name,
                    "version": bundle.version,
                    "publisher": bundle.publisher,
                    "source_path": bundle.source_path,
                    "signed": bundle.signed,
                    "signature_valid": bundle.signature_valid,
                    "signature_error": bundle.signature_error,
                }
            )
        return payload
