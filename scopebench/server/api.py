from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from pydantic import BaseModel, ConfigDict, Field

from scopebench.contracts import TaskContract
from scopebench.plan import PlanDAG
from scopebench.runtime.guard import evaluate
from scopebench.scoring.axes import combine_aggregates
from scopebench.scoring.calibration import CalibratedDecisionThresholds
from scopebench.tracing.otel import current_trace_context, init_tracing
from scopebench.scoring.rules import build_budget_ledger
from scopebench.session import MultiAgentSession
from scopebench.tracing.otel import get_tracer, init_tracing

SWE_READ_TOOLS = {"git_read", "file_read"}
SWE_WRITE_TOOLS = {"git_patch", "git_rewrite", "file_write"}
VALIDATION_TOOLS = {"analysis", "test_run", "pytest"}
VALIDATION_HINTS = ("test", "verify", "validation", "assert", "check")


class EvaluateRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "contract": {
                    "goal": "Fix flaky auth test in CI",
                    "non_goals": ["Refactor unrelated auth modules"],
                    "constraints": ["Do not change production auth behavior"],
                    "acceptance": ["Auth test passes consistently"],
                    "preset": "balanced",
                },
                "plan": {
                    "task": "Diagnose and fix flaky token refresh test",
                    "steps": [
                        {
                            "id": "read-failing-test",
                            "description": "Inspect the failing test and auth refresh implementation.",
                            "tool": "git_read",
                        },
                        {
                            "id": "patch-refresh-window",
                            "description": "Patch the refresh timing logic to avoid race conditions.",
                            "tool": "git_patch",
                        },
                        {
                            "id": "validate-auth",
                            "description": "Run targeted auth tests.",
                            "tool": "pytest",
                        },
                    ],
                },
                "include_steps": True,
                "include_summary": True,
                "include_telemetry": True,
                "shadow_mode": False,
                "policy_backend": "python",
            }
        },
    )
    contract: Dict[str, Any] = Field(..., description="TaskContract as dict")
    plan: Dict[str, Any] = Field(..., description="PlanDAG as dict")
    include_steps: bool = Field(False, description="Include step-level vectors and rationales.")
    include_summary: bool = Field(False, description="Include summary and next-step guidance.")
    include_telemetry: bool = Field(
        True, description="Include lightweight evaluation telemetry fields."
    )
    shadow_mode: bool = Field(
        False, description="If true, never block execution; return what enforcement would decide."
    )
    ask_action: Optional[str] = Field(
        None, description="Optional feedback: accepted/replanned/ignored."
    )
    outcome: Optional[str] = Field(
        None,
        description="Optional outcome feedback: tests_pass/tests_fail/rollback/manual_override.",
    )
    calibration_scale: Optional[float] = Field(
        None, ge=0.0, description="Optional scale for aggregate scores."
    )
    policy_backend: Optional[str] = Field(
        None, description="Policy backend override: python|opa|cedar."
    )


class AxisDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: float
    rationale: str
    confidence: float


class StepDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")
    step_id: Optional[str]
    tool: Optional[str]
    tool_category: Optional[str]
    est_cost_usd: Optional[float] = None
    est_time_days: Optional[float] = None
    est_benefit: Optional[float] = None
    benefit_unit: Optional[str] = None
    axes: Dict[str, AxisDetail]


class TelemetryDetail(BaseModel):
    schema_version: str = "telemetry_v1"
    model_config = ConfigDict(extra="forbid")
    preset: str
    policy_input_version: str
    task_type: str
    plan_size: int
    decision: str
    triggered_rules: List[str]
    has_read_before_write: bool
    has_validation_after_write: bool
    ask_action: Optional[str] = None
    outcome: Optional[str] = None


class EvaluateResponse(BaseModel):
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "decision": "ALLOW",
                "policy_backend": "python",
                "policy_version": "v1",
                "policy_hash": "sha256:demo",
                "effective_decision": "ALLOW",
                "shadow_mode": False,
                "reasons": ["Aggregate risk stayed below threshold."],
                "exceeded": {},
                "asked": {},
                "aggregate": {
                    "spatial": 0.08,
                    "temporal": 0.12,
                    "depth": 0.2,
                    "irreversibility": 0.06,
                    "resource_intensity": 0.1,
                    "legal_exposure": 0.02,
                    "dependency_creation": 0.04,
                    "stakeholder_radius": 0.05,
                    "power_concentration": 0.03,
                    "uncertainty": 0.22,
                },
                "n_steps": 3,
                "steps": [
                    {
                        "step_id": "patch-refresh-window",
                        "tool": "git_patch",
                        "tool_category": "write",
                        "axes": {
                            "uncertainty": {
                                "value": 0.35,
                                "rationale": "Timing behavior is partially inferred from flaky logs.",
                                "confidence": 0.72,
                            }
                        },
                    }
                ],
                "summary": "Decision ALLOW (effective: ALLOW). Top axes: uncertainty=0.22, depth=0.20, temporal=0.12.",
                "next_steps": ["Proceed; plan appears proportionate to the contract."],
                "plan_patch_suggestion": [],
                "telemetry": {
                    "preset": "balanced",
                    "policy_input_version": "v1",
                    "task_type": "bug_fix",
                    "plan_size": 3,
                    "decision": "ALLOW",
                    "triggered_rules": [],
                    "has_read_before_write": True,
                    "has_validation_after_write": True,
                    "ask_action": None,
                    "outcome": None,
                },
                "policy_input": {
                    "task_type": "bug_fix",
                    "read_before_write": True,
                    "validation_after_write": True,
                },
            }
        },
    )
    decision: str
    policy_backend: str
    policy_version: str
    policy_hash: str
    effective_decision: str
    shadow_mode: bool
    reasons: list[str]
    exceeded: Dict[str, Dict[str, float]]
    asked: Dict[str, float]
    aggregate: Dict[str, float]
    n_steps: int
    steps: Optional[List[StepDetail]] = None
    summary: Optional[str] = None
    next_steps: Optional[List[str]] = None
    plan_patch_suggestion: Optional[List[Dict[str, Any]]] = None
    telemetry: Optional[TelemetryDetail] = None
    policy_input: Optional[Dict[str, Any]] = None


class EvaluateSessionRequest(BaseModel):
    session: Dict[str, Any] = Field(..., description="MultiAgentSession as dict")
    include_steps: bool = Field(False, description="Include step-level vectors and rationales.")
    include_telemetry: bool = Field(
        True, description="Include lightweight evaluation telemetry fields."
    )
    policy_backend: Optional[str] = Field(
        None, description="Policy backend override: python|opa|cedar."
    )


class SessionAggregateDetail(BaseModel):
    aggregate: Dict[str, float]
    ledger: Dict[str, Dict[str, float]]
    decision: str


class EvaluateSessionResponse(BaseModel):
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    decision: str
    per_agent: Dict[str, SessionAggregateDetail]
    global_: SessionAggregateDetail = Field(..., alias="global")

    model_config = {"populate_by_name": True}


def _summarize_response(policy, aggregate, effective_decision: str) -> str:
    top_axes = sorted(aggregate.items(), key=lambda item: item[1], reverse=True)[:3]
    axes_text = ", ".join(f"{axis}={value:.2f}" for axis, value in top_axes)
    return f"Decision {policy.decision.value} (effective: {effective_decision}). Top axes: {axes_text}."


def _next_steps_from_policy(policy) -> List[str]:
    suggestions: List[str] = []
    for axis, (_, threshold) in policy.exceeded.items():
        suggestions.append(
            f"Reduce {axis} below {float(threshold):.2f} or split into smaller steps."
        )

    if "read_before_write" in policy.asked:
        suggestions.append(
            "Add an explicit read step before patching code (for example: git_read on failing files)."
        )
    if "validation_after_write" in policy.asked:
        suggestions.append(
            "Add a downstream validation step after patching (for example: run targeted tests)."
        )

    for axis, threshold in policy.asked.items():
        if axis in {"read_before_write", "validation_after_write"}:
            continue
        suggestions.append(f"Consider approval or mitigating {axis} below {float(threshold):.2f}.")

    if any("Tool category" in reason for reason in policy.reasons):
        suggestions.append("Remove high-risk tool categories or get explicit approval.")
    if not suggestions:
        suggestions.append("Proceed; plan appears proportionate to the contract.")
    return suggestions[:5]


def _suggest_plan_patch(policy, plan: PlanDAG) -> List[Dict[str, Any]]:
    patches: List[Dict[str, Any]] = []
    triggered = set(policy.exceeded.keys()) | set(policy.asked.keys())
    if "read_before_write" in policy.asked:
        first_write = next((step for step in plan.steps if step.tool in SWE_WRITE_TOOLS), None)
        if first_write is not None:
            patches.append(
                {
                    "op": "insert_before",
                    "target_step_id": first_write.id,
                    "step": {
                        "id": "read_before_write",
                        "description": "Read failing test and impacted source files.",
                        "tool": "git_read",
                    },
                }
            )
    if "validation_after_write" in policy.asked:
        first_write = next((step for step in plan.steps if step.tool in SWE_WRITE_TOOLS), None)
        if first_write is not None:
            patches.append(
                {
                    "op": "insert_after",
                    "target_step_id": first_write.id,
                    "step": {
                        "id": "validate_after_patch",
                        "description": "Run targeted tests for modified behavior.",
                        "tool": "analysis",
                    },
                }
            )

    if "dependency_creation" in triggered:
        high_dependency = next(
            (
                step
                for step in plan.steps
                if (step.tool_category or "") in {"infra", "payments", "finance", "health"}
            ),
            None,
        )
        if high_dependency is not None:
            patches.append(
                {
                    "op": "replace",
                    "target_step_id": high_dependency.id,
                    "step": {
                        "id": f"{high_dependency.id}_reduced_tooling",
                        "description": "Use lower-risk analysis-first workflow before introducing external dependencies.",
                        "tool": "analysis",
                    },
                    "rationale": "Reduce tool category risk and defer new dependency creation.",
                }
            )

    if "depth" in triggered and len(plan.steps) >= 2:
        split_after = max(1, len(plan.steps) // 2)
        patches.append(
            {
                "op": "split_plan",
                "after_step_id": plan.steps[split_after - 1].id,
                "chunks": [
                    {
                        "name": "phase_1_safe_discovery",
                        "step_ids": [step.id for step in plan.steps[:split_after]],
                    },
                    {
                        "name": "phase_2_execution",
                        "step_ids": [step.id for step in plan.steps[split_after:]],
                    },
                ],
                "rationale": "Split high-depth execution into staged checkpoints.",
            }
        )

    if "power_concentration" in triggered:
        first_privileged = next(
            (
                step
                for step in plan.steps
                if (step.tool_category or "") in {"iam", "infra", "payments"}
            ),
            None,
        )
        if first_privileged is not None:
            patches.append(
                {
                    "op": "insert_before",
                    "target_step_id": first_privileged.id,
                    "step": {
                        "id": "approval_gate",
                        "description": "Request human approval with blast-radius summary before privileged action.",
                        "tool": "analysis",
                    },
                    "rationale": "Add approval gate before high-power operations.",
                }
            )

    if "irreversibility" in triggered:
        irreversible = next(
            (
                step
                for step in plan.steps
                if (step.tool_category or "") in {"infra", "iam", "payments", "health"}
                or any(keyword in step.description.lower() for keyword in ("delete", "destroy", "drop", "rotate"))
            ),
            None,
        )
        if irreversible is not None:
            patches.append(
                {
                    "op": "replace",
                    "target_step_id": irreversible.id,
                    "step": {
                        "id": f"{irreversible.id}_reversible_preview",
                        "description": "Run dry-run/preview and create rollback artifact before applying irreversible change.",
                        "tool": "analysis",
                    },
                    "rationale": "Reduce irreversible operations by introducing reversible preview.",
                }
            )
    return patches


def _looks_like_validation(description: str) -> bool:
    text = description.lower()
    return any(hint in text for hint in VALIDATION_HINTS)


def _has_read_before_write(plan: PlanDAG) -> bool:
    read_seen = False
    for step in plan.steps:
        if step.tool in SWE_READ_TOOLS:
            read_seen = True
        if step.tool in SWE_WRITE_TOOLS and not read_seen:
            return False
    return True


def _has_validation_after_write(plan: PlanDAG) -> bool:
    write_seen = False
    for step in plan.steps:
        if step.tool in SWE_WRITE_TOOLS:
            write_seen = True
            continue
        if write_seen and (
            (step.tool in VALIDATION_TOOLS) or _looks_like_validation(step.description)
        ):
            return True
    return not write_seen


def _infer_task_type(contract: TaskContract, plan: PlanDAG) -> str:
    text = f"{contract.goal} {plan.task}".lower()
    if "bug" in text or "fix" in text:
        return "bug_fix"
    if "test" in text:
        return "test_stabilization"
    if "refactor" in text:
        return "refactor"
    return "general_coding"


def _build_telemetry(
    contract: TaskContract, plan: PlanDAG, policy, ask_action: Optional[str], outcome: Optional[str]
) -> TelemetryDetail:
    triggered = sorted(set(policy.exceeded.keys()) | set(policy.asked.keys()))
    return TelemetryDetail(
        preset=contract.preset.value,
        policy_input_version="v1",
        task_type=_infer_task_type(contract, plan),
        plan_size=len(plan.steps),
        decision=policy.decision.value,
        triggered_rules=triggered,
        has_read_before_write=_has_read_before_write(plan),
        has_validation_after_write=_has_validation_after_write(plan),
        ask_action=ask_action,
        outcome=outcome,
    )


def _effective_decision(policy_decision: str, shadow_mode: bool) -> str:
    if shadow_mode and policy_decision in {"ASK", "DENY"}:
        return "ALLOW"
    return policy_decision


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def _telemetry_row(
    telemetry: TelemetryDetail,
    policy_input: Optional[Dict[str, Any]],
    aggregate: Dict[str, float],
    asked: Dict[str, float],
    exceeded: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    return {
        "schema_version": "telemetry_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "policy_input": policy_input or {},
        "decision": telemetry.decision,
        "aggregate": aggregate,
        "asked": asked,
        "exceeded": exceeded,
        "feedback": {
            "ask_action": telemetry.ask_action,
            "outcome": telemetry.outcome,
        },
        "telemetry": telemetry.model_dump(),
    }


def _load_telemetry_rows(path: Path, limit: int = 200) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if limit <= 0:
        return rows
    return rows[-limit:]


def create_app(default_policy_backend: str = "python", telemetry_jsonl_path: Optional[str] = None) -> FastAPI:
    init_tracing(enable_console=False)
    tracer = get_tracer("scopebench")
    app = FastAPI(title="ScopeBench", version="0.1.0")
    configured_telemetry_path = telemetry_jsonl_path or os.getenv("SCOPEBENCH_TELEMETRY_JSONL_PATH")

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/ui", response_class=HTMLResponse)
    def ui_index():
        from pathlib import Path

        ui_path = Path(__file__).resolve().parents[1] / "ui" / "index.html"
        return ui_path.read_text(encoding="utf-8")

    @app.get("/templates")
    def templates_endpoint():
        from pathlib import Path
        import yaml

        templates_root = Path(__file__).resolve().parents[1] / "templates"
        payload: List[Dict[str, Any]] = []
        for domain_dir in sorted(p for p in templates_root.iterdir() if p.is_dir()):
            variants: Dict[str, Dict[str, Any]] = {}

            def load_variant(variant_name: str, contract_path: Path, plan_path: Path, notes_path: Path) -> None:
                variants[variant_name] = {
                    "metadata": {
                        "has_contract": contract_path.exists(),
                        "has_plan": plan_path.exists(),
                        "has_notes": notes_path.exists(),
                    },
                    "content": {
                        "contract": yaml.safe_load(contract_path.read_text(encoding="utf-8")) if contract_path.exists() else None,
                        "plan": yaml.safe_load(plan_path.read_text(encoding="utf-8")) if plan_path.exists() else None,
                        "notes": notes_path.read_text(encoding="utf-8") if notes_path.exists() else None,
                    },
                }

            load_variant(
                "default",
                domain_dir / "contract.yaml",
                domain_dir / "plan.yaml",
                domain_dir / "notes.md",
            )

            for contract_path in sorted(domain_dir.glob("*.contract.yaml")):
                variant = contract_path.name[: -len(".contract.yaml")]
                if not variant:
                    continue
                load_variant(
                    variant,
                    contract_path,
                    domain_dir / f"{variant}.plan.yaml",
                    domain_dir / f"{variant}.notes.md",
                )

            payload.append({"domain": domain_dir.name, "variants": variants})
        return {"templates": payload}

    @app.get("/tools")
    def tools_endpoint():
        from scopebench.scoring.rules import ToolRegistry

        registry = ToolRegistry.load_default()
        tools = []
        for tool_name, tool_info in sorted(registry._tools.items()):  # noqa: SLF001
            tools.append(
                {
                    "tool": tool_name,
                    "category": tool_info.category,
                    "domains": list(tool_info.domains),
                    "risk_class": tool_info.risk_class,
                    "priors": dict(tool_info.priors),
                    "default_effects": dict(tool_info.default_effects),
                }
            )

        schema = {
            "type": "object",
            "required": ["tool", "category", "domains", "risk_class", "priors", "default_effects"],
            "properties": {
                "tool": {"type": "string"},
                "category": {"type": "string"},
                "domains": {"type": "array", "items": {"type": "string"}},
                "risk_class": {"type": "string", "enum": ["low", "moderate", "high", "critical"]},
                "priors": {"type": "object", "additionalProperties": {"type": "number", "minimum": 0.0, "maximum": 1.0}},
                "default_effects": {"type": "object"},
            },
            "additionalProperties": False,
        }
        return {"tools": tools, "normalized_schema": schema}

    @app.get("/cases")
    def cases_endpoint():
        from scopebench.bench.dataset import default_cases_path, load_cases

        try:
            cases = load_cases(default_cases_path())
        except ValueError as exc:
            return {
                "datasets": [],
                "domains": [],
                "count": 0,
                "cases": [],
                "error": str(exc),
            }

        return {
            "datasets": sorted({case.id for case in cases}),
            "domains": sorted({case.domain for case in cases}),
            "count": len(cases),
            "cases": [
                {
                    "id": case.id,
                    "domain": case.domain,
                    "instruction": case.instruction,
                    "expected_decision": case.expected_decision,
                    "contract": case.contract,
                }
                for case in cases
            ],
        }

    @app.get("/telemetry/replay")
    def telemetry_replay(limit: int = 50):
        if not configured_telemetry_path:
            return {
                "enabled": False,
                "rows": [],
                "message": "Set SCOPEBENCH_TELEMETRY_JSONL_PATH to enable replay.",
            }

        rows = _load_telemetry_rows(Path(configured_telemetry_path), limit=limit)
        return {
            "enabled": True,
            "source": configured_telemetry_path,
            "count": len(rows),
            "rows": rows,
        }

    @app.post("/evaluate", response_model=EvaluateResponse)
    @app.post(
        "/evaluate",
        response_model=EvaluateResponse,
        openapi_extra={
            "requestBody": {
                "content": {
                    "application/json": {
                        "example": EvaluateRequest.model_config["json_schema_extra"]["example"]
                    }
                }
            },
            "responses": {
                "200": {
                    "content": {
                        "application/json": {
                            "example": EvaluateResponse.model_config["json_schema_extra"]["example"]
                        }
                    }
                }
            },
        },
    )
    def evaluate_endpoint(req: EvaluateRequest):
        contract = TaskContract.model_validate(req.contract)
        plan = PlanDAG.model_validate(req.plan)
        calibration = None
        if req.calibration_scale is not None:
            calibration = CalibratedDecisionThresholds(global_scale=req.calibration_scale)
        backend = req.policy_backend or default_policy_backend
        res = evaluate(contract, plan, calibration=calibration, policy_backend=backend)
        pol = res.policy

        decision = pol.decision.value
        effective_decision = _effective_decision(decision, req.shadow_mode)

        steps = None
        if req.include_steps:
            steps = []
            plan_steps_by_id = {step.id: step for step in plan.steps}
            for vec in res.vectors:
                axes = {
                    "spatial": AxisDetail(**vec.spatial.model_dump()),
                    "temporal": AxisDetail(**vec.temporal.model_dump()),
                    "depth": AxisDetail(**vec.depth.model_dump()),
                    "irreversibility": AxisDetail(**vec.irreversibility.model_dump()),
                    "resource_intensity": AxisDetail(**vec.resource_intensity.model_dump()),
                    "legal_exposure": AxisDetail(**vec.legal_exposure.model_dump()),
                    "dependency_creation": AxisDetail(**vec.dependency_creation.model_dump()),
                    "stakeholder_radius": AxisDetail(**vec.stakeholder_radius.model_dump()),
                    "power_concentration": AxisDetail(**vec.power_concentration.model_dump()),
                    "uncertainty": AxisDetail(**vec.uncertainty.model_dump()),
                }
                plan_step = plan_steps_by_id.get(vec.step_id or "")
                steps.append(
                    StepDetail(
                        step_id=vec.step_id,
                        tool=vec.tool,
                        tool_category=vec.tool_category,
                        est_cost_usd=plan_step.est_cost_usd if plan_step else None,
                        est_time_days=plan_step.est_time_days if plan_step else None,
                        est_benefit=plan_step.est_benefit if plan_step else None,
                        benefit_unit=plan_step.benefit_unit if plan_step else None,
                        axes=axes,
                    )
                )

        summary = None
        next_steps = None
        patch_suggestion = None
        if req.include_summary:
            summary = _summarize_response(pol, res.aggregate.as_dict(), effective_decision)
            next_steps = _next_steps_from_policy(pol)
            patch_suggestion = _suggest_plan_patch(pol, plan)

        telemetry = (
            _build_telemetry(contract, plan, pol, req.ask_action, req.outcome)
            if req.include_telemetry
            else None
        )

        reasons = list(pol.reasons)
        if req.shadow_mode and decision != effective_decision:
            reasons.append(
                "Shadow mode enabled: returning effective_decision=ALLOW while preserving policy decision for analysis"
            )

        with tracer.start_as_current_span("scopebench.evaluate.response"):
            trace_context = current_trace_context()
        exceeded_payload = {
            k: {"value": float(v[0]), "threshold": float(v[1])} for k, v in pol.exceeded.items()
        }
        asked_payload = {k: float(v) for k, v in pol.asked.items()}
        policy_input_payload = pol.policy_input.__dict__ if (req.include_telemetry and pol.policy_input) else None

        if req.include_telemetry and telemetry and configured_telemetry_path:
            _append_jsonl(
                Path(configured_telemetry_path),
                _telemetry_row(
                    telemetry=telemetry,
                    policy_input=policy_input_payload,
                    aggregate=res.aggregate.as_dict(),
                    asked=asked_payload,
                    exceeded=exceeded_payload,
                ),
            )

        return EvaluateResponse(
            trace_id=trace_context.get("trace_id"),
            span_id=trace_context.get("span_id"),
            decision=decision,
            policy_backend=pol.policy_backend,
            policy_version=pol.policy_version,
            policy_hash=pol.policy_hash,
            effective_decision=effective_decision,
            shadow_mode=req.shadow_mode,
            reasons=reasons,
            exceeded=exceeded_payload,
            asked=asked_payload,
            aggregate=res.aggregate.as_dict(),
            n_steps=res.aggregate.n_steps,
            steps=steps,
            summary=summary,
            next_steps=next_steps,
            plan_patch_suggestion=patch_suggestion,
            telemetry=telemetry,
            policy_input=policy_input_payload,
        )

    @app.post("/evaluate_session", response_model=EvaluateSessionResponse)
    def evaluate_session_endpoint(req: EvaluateSessionRequest):
        session = MultiAgentSession.model_validate(req.session)
        backend = req.policy_backend or default_policy_backend

        per_agent: Dict[str, SessionAggregateDetail] = {}
        agent_aggregates = []
        global_plans: List[PlanDAG] = []
        global_decision = "ALLOW"

        for agent in sorted(session.agents, key=lambda item: item.agent_id):
            agent_plans = session.plans_for(agent.agent_id)
            contract = session.contract_for(agent.agent_id)
            global_plans.extend(agent_plans)

            with tracer.start_as_current_span("scopebench.evaluate_session.agent") as span:
                span.set_attribute("scopebench.agent_id", agent.agent_id)
                span.set_attribute("scopebench.agent_plan_count", len(agent_plans))
                agent_results = [evaluate(contract, plan, policy_backend=backend) for plan in agent_plans]
            agent_aggregate = combine_aggregates([result.aggregate for result in agent_results])
            agent_decision = "ALLOW"
            for result in agent_results:
                if result.policy.decision.value == "DENY":
                    agent_decision = "DENY"
                    break
                if result.policy.decision.value == "ASK":
                    agent_decision = "ASK"

            agent_ledger = build_budget_ledger(contract, agent_plans)
            if any(entry["exceeded"] > 0 for entry in agent_ledger.values()) and agent_decision != "DENY":
                agent_decision = "ASK"

            per_agent[agent.agent_id] = SessionAggregateDetail(
                aggregate=agent_aggregate.as_dict(),
                ledger=agent_ledger,
                decision=agent_decision,
            )
            agent_aggregates.append(agent_aggregate)

            if agent_decision == "DENY":
                global_decision = "DENY"
            elif agent_decision == "ASK" and global_decision != "DENY":
                global_decision = "ASK"

        global_aggregate = combine_aggregates(agent_aggregates)
        global_ledger = build_budget_ledger(session.global_contract, global_plans)
        if any(entry["exceeded"] > 0 for entry in global_ledger.values()) and global_decision != "DENY":
            global_decision = "ASK"

        global_scope = SessionAggregateDetail(
            aggregate=global_aggregate.as_dict(),
            ledger=global_ledger,
            decision=global_decision,
        )
        with tracer.start_as_current_span("scopebench.evaluate_session.response"):
            trace_context = current_trace_context()
        return EvaluateSessionResponse(
            trace_id=trace_context.get("trace_id"),
            span_id=trace_context.get("span_id"),
            decision=global_decision,
            per_agent=per_agent,
            global_=global_scope,
        )

    return app
