from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import typer
import yaml
from rich.console import Console
from rich.table import Table

from scopebench.bench.judge import run_judge_bench
from scopebench.bench.community import (
    append_case,
    submit_pull_request,
    suggest_case,
    validate_cases_file,
)
from scopebench.bench.plugin_harness import run_plugin_test_harness
from scopebench.bench.continuous_learning import (
    analyze_continuous_learning,
    apply_learning_to_registry,
    render_learning_report_markdown,
)
from scopebench.bench.weekly import (
    refresh_weekly_calibration,
    replay_benchmark_slice,
)
from scopebench.runtime.guard import evaluate, evaluate_from_files
from scopebench.scoring.calibration import (
    CalibratedDecisionThresholds,
    calibration_to_dict,
    compute_axis_calibration_from_telemetry,
)
from scopebench.scoring.effects_annotator import suggest_effects_for_plan
from scopebench.scoring.llm_judge import JudgeMode
from scopebench.tracing.otel import init_tracing
from scopebench.plugins import lint_plugin_bundle, sign_plugin_bundle

app = typer.Typer(add_completion=False, help="ScopeBench: plan-level proportionality enforcement.")
template_app = typer.Typer(help="Domain template discovery and generation.")
app.add_typer(template_app, name="template")
console = Console()


def _load_plan_yaml(path: Path) -> Dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise typer.BadParameter(f"Plan file {path} must contain a YAML mapping")
    return raw


TEMPLATES_ROOT = Path(__file__).resolve().parent / "templates"
TEMPLATE_KINDS = {"bundle", "contract", "plan", "notes"}


def _template_domains() -> List[str]:
    if not TEMPLATES_ROOT.exists():
        return []
    return sorted(p.name for p in TEMPLATES_ROOT.iterdir() if p.is_dir())


def _template_variants(domain: str) -> List[str]:
    root = TEMPLATES_ROOT / domain
    if not root.exists():
        return []
    variants = {"default"}
    for path in root.glob("*.contract.yaml"):
        stem = path.name[: -len(".contract.yaml")]
        if stem:
            variants.add(stem)
    return sorted(variants)


def _template_file(domain: str, variant: str, kind: str) -> Path:
    root = TEMPLATES_ROOT / domain
    if variant == "default":
        filename = {"contract": "contract.yaml", "plan": "plan.yaml", "notes": "notes.md"}[kind]
    else:
        filename = {
            "contract": f"{variant}.contract.yaml",
            "plan": f"{variant}.plan.yaml",
            "notes": f"{variant}.notes.md",
        }[kind]
    return root / filename


def _read_yaml(path: Path) -> Dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise typer.BadParameter(f"Template file {path} must contain a YAML mapping")
    return raw


def _load_template_bundle(domain: str, variant: str) -> Dict[str, Dict[str, Any]]:
    return {
        "contract": _read_yaml(_template_file(domain, variant, "contract")),
        "plan": _read_yaml(_template_file(domain, variant, "plan")),
    }


def _resolve_template_name(name: str) -> Tuple[str, str, str]:
    segments = [part for part in name.split("/") if part]
    if not segments:
        raise typer.BadParameter("Template name cannot be empty")

    domain = segments[0]
    variant = "default"
    kind = "bundle"

    if len(segments) == 2:
        if segments[1] in TEMPLATE_KINDS:
            kind = segments[1]
        else:
            variant = segments[1]
    elif len(segments) == 3:
        variant, kind = segments[1], segments[2]
    elif len(segments) > 3:
        raise typer.BadParameter(
            "Template name must be <domain>[/kind], <domain>/<variant>, or <domain>/<variant>/<kind>"
        )

    if domain not in _template_domains():
        available = ", ".join(_template_domains()) or "none"
        raise typer.BadParameter(f"Unknown template domain '{domain}'. Available: {available}")
    variants = _template_variants(domain)
    if variant not in variants:
        available = ", ".join(variants)
        raise typer.BadParameter(
            f"Unknown template variant '{variant}' in domain '{domain}'. Available: {available}"
        )
    if kind not in TEMPLATE_KINDS:
        raise typer.BadParameter("Template kind must be one of: bundle, contract, plan, notes")
    return domain, variant, kind


def _compact_payload(result) -> dict:
    policy = result.policy
    agg = result.aggregate.as_dict()
    sorted_axes = sorted(agg.items(), key=lambda item: item[1], reverse=True)

    def step_peak(v):
        d = v.as_dict
        top_axis = max(d, key=lambda k: d[k])
        return d[top_axis], top_axis

    peaks = []
    for v in result.vectors:
        val, axis = step_peak(v)
        peaks.append((val, axis, v))
    peaks.sort(reverse=True, key=lambda x: x[0])

    return {
        "decision": policy.decision.value,
        "reasons": policy.reasons[:5],
        "top_axes": [{"axis": axis, "value": value} for axis, value in sorted_axes[:3]],
        "top_steps": [
            {
                "step_id": v.step_id,
                "tool": v.tool,
                "axis": axis,
                "value": val,
                "rationale": getattr(v, axis).rationale,
            }
            for val, axis, v in peaks[:3]
        ],
        "n_steps": result.aggregate.n_steps,
    }


def _print_result(result, as_json: bool = False, compact_json: bool = False) -> None:
    policy = result.policy
    if compact_json:
        console.print_json(json.dumps(_compact_payload(result)))
        return
    if as_json:
        payload = {
            "decision": policy.decision.value,
            "reasons": policy.reasons,
            "exceeded": {k: {"value": v[0], "threshold": v[1]} for k, v in policy.exceeded.items()},
            "asked": policy.asked,
            "aggregate": result.aggregate.as_dict(),
            "n_steps": result.aggregate.n_steps,
        }
        console.print_json(json.dumps(payload))
        return

    console.print(f"[bold]Decision:[/bold] {policy.decision.value}")
    for r in policy.reasons[:8]:
        console.print(f" - {r}")

    # Aggregate scores
    t = Table(title="Aggregate Scope (0..1)")
    t.add_column("Axis")
    t.add_column("Value", justify="right")
    t.add_column("Max", justify="right")
    maxes = result.contract.thresholds
    agg = result.aggregate.as_dict()
    axis_to_max = {
        "spatial": maxes.max_spatial,
        "temporal": maxes.max_temporal,
        "depth": maxes.max_depth,
        "irreversibility": maxes.max_irreversibility,
        "resource_intensity": maxes.max_resource_intensity,
        "legal_exposure": maxes.max_legal_exposure,
        "dependency_creation": maxes.max_dependency_creation,
        "stakeholder_radius": maxes.max_stakeholder_radius,
        "power_concentration": maxes.max_power_concentration,
        "uncertainty": maxes.max_uncertainty,
    }
    for axis, val in agg.items():
        mx = axis_to_max[axis]
        style = "red" if axis in policy.exceeded else ("yellow" if axis in policy.asked else "")
        t.add_row(axis, f"{val:.2f}", f"{mx:.2f}", style=style)
    console.print(t)

    # Step highlights (top 3 by max axis)
    step_table = Table(title="Step highlights")
    step_table.add_column("Step")
    step_table.add_column("Tool")
    step_table.add_column("Top axis")
    step_table.add_column("Value", justify="right")
    step_table.add_column("Rationale")

    def step_peak(v):
        d = v.as_dict
        top_axis = max(d, key=lambda k: d[k])
        return d[top_axis], top_axis

    peaks = []
    for v in result.vectors:
        val, axis = step_peak(v)
        peaks.append((val, axis, v))
    peaks.sort(reverse=True, key=lambda x: x[0])
    for val, axis, v in peaks[:3]:
        rationale = getattr(v, axis).rationale
        step_table.add_row(v.step_id or "", v.tool or "", axis, f"{val:.2f}", rationale[:60])
    console.print(step_table)


@template_app.command("list")
def template_list() -> None:
    """List installed template domains and their available files."""
    table = Table(title="Templates")
    table.add_column("Domain")
    table.add_column("Variants")
    table.add_column("Files")
    for domain in _template_domains():
        variants = _template_variants(domain)
        files = []
        for filename in ("contract.yaml", "plan.yaml", "notes.md"):
            if (TEMPLATES_ROOT / domain / filename).exists():
                files.append(filename)
        table.add_row(domain, ", ".join(variants), ", ".join(files))
    console.print(table)


@template_app.command("show")
def template_show(name: str = typer.Argument(..., help="Template name: <domain>[/kind]")) -> None:
    """Show a template as YAML/markdown without modification."""
    domain, variant, kind = _resolve_template_name(name)

    if kind == "notes":
        console.print(_template_file(domain, variant, "notes").read_text(encoding="utf-8"))
        return
    if kind == "contract":
        console.print(yaml.safe_dump(_read_yaml(_template_file(domain, variant, "contract")), sort_keys=False))
        return
    if kind == "plan":
        console.print(yaml.safe_dump(_read_yaml(_template_file(domain, variant, "plan")), sort_keys=False))
        return

    bundle = {
        "contract": _read_yaml(_template_file(domain, variant, "contract")),
        "plan": _read_yaml(_template_file(domain, variant, "plan")),
    }
    console.print(yaml.safe_dump(bundle, sort_keys=False))


@template_app.command("generate")
def template_generate(name: str = typer.Argument(..., help="Template name: <domain>[/kind]")) -> None:
    """Generate valid YAML from a bundled template."""
    template_show(name)


@template_app.command("wizard")
def template_wizard(
    goal: str | None = typer.Option(None, "--goal", help="Goal/task statement to place in contract and plan."),
    domain: str | None = typer.Option(None, "--domain", help="Template domain (swe|ops|finance|health|marketing)."),
    variant: str | None = typer.Option(None, "--variant", help="Template variant inside the selected domain."),
    preset: str | None = typer.Option(None, "--preset", help="Contract preset override (team|enterprise|regulated|...)."),
    edit_steps: bool = typer.Option(
        True,
        "--edit-steps/--no-edit-steps",
        help="Interactively edit generated step descriptions/tools before printing YAML.",
    ),
) -> None:
    """Interactive wizard that scaffolds a contract+plan from domain templates."""
    domains = _template_domains()
    if not domains:
        raise typer.BadParameter("No templates found.")

    selected_domain = domain or typer.prompt("Domain", default=domains[0], show_default=True)
    if selected_domain not in domains:
        raise typer.BadParameter(f"Unknown domain '{selected_domain}'. Available: {', '.join(domains)}")

    variants = _template_variants(selected_domain)
    selected_variant = variant or typer.prompt("Variant", default="default", show_default=True)
    if selected_variant not in variants:
        raise typer.BadParameter(
            f"Unknown variant '{selected_variant}' in domain '{selected_domain}'. Available: {', '.join(variants)}"
        )

    bundle = _load_template_bundle(selected_domain, selected_variant)
    contract = bundle["contract"]
    plan = bundle["plan"]

    selected_goal = goal or typer.prompt(
        "Goal",
        default=str(contract.get("goal") or plan.get("task") or ""),
        show_default=True,
    )
    selected_preset = preset or typer.prompt(
        "Preset",
        default=str(contract.get("preset") or "team"),
        show_default=True,
    )

    contract["goal"] = selected_goal
    contract["domain"] = selected_domain
    contract["preset"] = selected_preset
    plan["task"] = selected_goal

    if edit_steps:
        for index, step in enumerate(plan.get("steps") or []):
            step_id = step.get("id") or f"step-{index + 1}"
            step["description"] = typer.prompt(
                f"Step {step_id} description",
                default=str(step.get("description") or ""),
                show_default=True,
            )
            step["tool"] = typer.prompt(
                f"Step {step_id} tool",
                default=str(step.get("tool") or "analysis"),
                show_default=True,
            )

    console.print(
        yaml.safe_dump(
            {"contract": contract, "plan": plan},
            sort_keys=False,
        )
    )


@app.command()
def run(
    contract_path: Path = typer.Argument(..., help="Path to contract YAML."),
    plan_path: Path = typer.Argument(..., help="Path to plan YAML."),
    json_out: bool = typer.Option(False, "--json", help="Output machine-readable JSON."),
    compact_json: bool = typer.Option(
        False, "--compact-json", help="Output compact JSON for chat/agent UX."
    ),
    otel_console: bool = typer.Option(
        False, "--otel-console", help="Enable console OpenTelemetry exporter."
    ),
    calibration_scale: float = typer.Option(
        1.0, "--calibration-scale", min=0.0, help="Global scale for aggregate scores."
    ),
    calibration_file: Path | None = typer.Option(
        None, "--calibration-file", help="Path to per-axis calibration JSON file."
    ),
    policy_backend: str = typer.Option(
        "python", "--policy-backend", help="Policy backend: python|opa|cedar."
    ),
    judge: JudgeMode = typer.Option(
        "heuristic", "--judge", help="Step judge mode: none|heuristic|llm."
    ),
    judge_cache_path: str = typer.Option(
        ".scopebench_cache", "--judge-cache-path", help="Cache directory for LLM judge outputs."
    ),
):
    """Evaluate a plan against a task contract and print ALLOW/ASK/DENY."""
    init_tracing(enable_console=otel_console)
    calibration = None
    if calibration_file is not None:
        from scopebench.scoring.calibration import load_calibration_file

        calibration = load_calibration_file(calibration_file)
    elif calibration_scale != 1.0:
        calibration = CalibratedDecisionThresholds(global_scale=calibration_scale)
    res = evaluate_from_files(
        str(contract_path),
        str(plan_path),
        calibration=calibration,
        policy_backend=policy_backend,
        judge=judge,
        judge_cache_path=judge_cache_path,
    )
    _print_result(res, as_json=json_out, compact_json=compact_json)


@app.command("suggest-effects")
def suggest_effects(
    plan_path: Path = typer.Argument(..., help="Path to plan YAML."),
    in_place: bool = typer.Option(
        False,
        "--in-place",
        help="Write suggested effects back into the provided plan file.",
    ),
    json_out: bool = typer.Option(False, "--json", help="Output machine-readable JSON."),
):
    """Suggest effects_v1 annotations per step using tool defaults and heuristics."""
    from scopebench.plan import plan_from_dict

    plan_data = _load_plan_yaml(plan_path)
    plan = plan_from_dict(plan_data)
    suggestions = suggest_effects_for_plan(plan)

    if in_place:
        step_lookup = {
            str(step.get("id")): step
            for step in plan_data.get("steps", [])
            if isinstance(step, dict)
        }
        for suggestion in suggestions:
            if suggestion.step_id in step_lookup:
                step_lookup[suggestion.step_id]["effects"] = suggestion.effects.model_dump(
                    mode="json", exclude_none=True
                )
        plan_path.write_text(yaml.safe_dump(plan_data, sort_keys=False), encoding="utf-8")

    payload = {
        "plan_path": str(plan_path),
        "in_place": in_place,
        "steps": [
            {
                "id": item.step_id,
                "tool": item.tool,
                "effects": item.effects.model_dump(mode="json", exclude_none=True),
            }
            for item in suggestions
        ],
    }

    if json_out:
        console.print_json(json.dumps(payload))
    else:
        for item in payload["steps"]:
            console.print(f"[bold]Step {item['id']}[/bold] tool={item['tool'] or 'unknown'}")
            console.print(yaml.safe_dump(item["effects"], sort_keys=False).strip())
            console.print("")
        if in_place:
            console.print(f"Updated {plan_path} with suggested effects_v1 annotations.")


@app.command()
def quickstart(
    json_out: bool = typer.Option(False, "--json", help="Output machine-readable JSON."),
    compact_json: bool = typer.Option(
        False, "--compact-json", help="Output compact JSON for chat/agent UX."
    ),
    otel_console: bool = typer.Option(
        False, "--otel-console", help="Enable console OpenTelemetry exporter."
    ),
):
    """Run the bundled phone-charge overreach example."""
    init_tracing(enable_console=otel_console)
    root = Path(__file__).resolve().parents[1]
    contract_path = root / "examples" / "phone_charge.contract.yaml"
    plan_path = root / "examples" / "phone_charge.plan.yaml"
    res = evaluate_from_files(str(contract_path), str(plan_path))
    _print_result(res, as_json=json_out, compact_json=compact_json)


@app.command()
def coding_quickstart(
    json_out: bool = typer.Option(False, "--json", help="Output machine-readable JSON."),
    compact_json: bool = typer.Option(
        False, "--compact-json", help="Output compact JSON for chat/agent UX."
    ),
    otel_console: bool = typer.Option(
        False, "--otel-console", help="Enable console OpenTelemetry exporter."
    ),
):
    """Run the bundled coding bugfix quickstart example (read → patch → validate)."""
    init_tracing(enable_console=otel_console)
    root = Path(__file__).resolve().parents[1]
    contract_path = root / "examples" / "coding_bugfix.contract.yaml"
    plan_path = root / "examples" / "coding_bugfix.plan.yaml"
    res = evaluate_from_files(str(contract_path), str(plan_path))
    _print_result(res, as_json=json_out, compact_json=compact_json)


@app.command()
def weekly_calibrate(
    telemetry_jsonl: Path = typer.Argument(..., help="Path to telemetry JSONL file."),
    output_path: Path = typer.Option("axis_calibration.json", "--out", help="Output calibration JSON path."),
    json_out: bool = typer.Option(False, "--json", help="Output machine-readable JSON report."),
):
    """Summarize telemetry, replay benchmark slice, and generate per-axis calibration."""
    calibration, report = refresh_weekly_calibration(telemetry_jsonl, output_path)
    calibration, calibration_stats = compute_axis_calibration_from_telemetry(telemetry_jsonl)
    root = Path(__file__).resolve().parents[1]
    replay = replay_benchmark_slice(root)

    payload = {
        "total_runs": report.total_runs,
        "decision_counts": report.decision_counts,
        "top_triggered_rules": [{"rule": r, "count": c} for r, c in report.top_triggered_rules],
        "ask_action_counts": report.ask_action_counts,
        "outcome_counts": report.outcome_counts,
        "calibration_file": str(output_path),
        "calibration": calibration_to_dict(calibration),
        "calibration_stats": {
            "triggered": calibration_stats.triggered,
            "false_alarms": calibration_stats.false_alarms,
            "overrides": calibration_stats.overrides,
            "failures": calibration_stats.failures,
        },
        "benchmark_replay": [
            {"name": r.name, "decision": r.decision, "expected": sorted(r.expected), "ok": r.ok}
            for r in replay
        ],
    }

    if json_out:
        console.print_json(json.dumps(payload))
    else:
        console.print(f"[bold]Weekly calibration[/bold] runs={report.total_runs}")
        console.print(f"Decision counts: {report.decision_counts}")
        console.print(f"Top triggered rules: {report.top_triggered_rules}")
        console.print(f"ASK actions: {report.ask_action_counts}")
        console.print(f"Outcomes: {report.outcome_counts}")
        console.print(f"Calibration file: {output_path}")
        for item in replay:
            status = "PASS" if item.ok else "FAIL"
            console.print(
                f" - {status} {item.name}: decision={item.decision} expected={sorted(item.expected)}"
            )

    if not all(item.ok for item in replay):
        raise typer.Exit(code=1)


@app.command("continuous-learn")
def continuous_learn(
    telemetry_jsonl: Path = typer.Argument(..., help="Path to telemetry JSONL file."),
    benchmark_jsonl: Path = typer.Option(
        Path("scopebench/bench/cases/redteam.jsonl"),
        "--benchmark",
        help="Optional benchmark JSONL for tactic mining.",
    ),
    registry_path: Path = typer.Option(
        Path("scopebench/tool_registry.yaml"),
        "--registry",
        help="Tool registry YAML to analyze/update.",
    ),
    report_out: Path = typer.Option(
        Path("continuous_learning_report.md"),
        "--report-out",
        help="Markdown report destination.",
    ),
    min_pair_support: int = typer.Option(2, "--min-pair-support", min=1),
    apply_updates: bool = typer.Option(
        False,
        "--apply-updates",
        help="Apply recommended priors/default_effects directly to tool_registry.yaml.",
    ),
):
    """Mine telemetry+benchmark signals and propose registry/effects updates."""
    report = analyze_continuous_learning(
        telemetry_path=telemetry_jsonl,
        benchmark_path=benchmark_jsonl,
        registry_path=registry_path,
        min_pair_support=min_pair_support,
    )

    report_markdown = render_learning_report_markdown(report)
    report_out.write_text(report_markdown, encoding="utf-8")

    payload = {
        "telemetry_rows": report.telemetry_rows,
        "benchmark_rows": report.benchmark_rows,
        "pattern_counts": report.pattern_counts,
        "risky_tool_pairs": [
            {"tools": [a, b], "count": count} for (a, b), count in report.risky_tool_pairs
        ],
        "report_path": str(report_out),
        "updated_registry": False,
    }

    if apply_updates:
        apply_learning_to_registry(report, registry_path=registry_path)
        payload["updated_registry"] = True
        payload["registry_path"] = str(registry_path)

    console.print_json(json.dumps(payload))


@app.command("judge-bench")
def judge_bench(
    cases_jsonl: Path = typer.Argument(..., help="Path to benchmark cases JSONL."),
    judge: JudgeMode = typer.Option("llm", "--judge", help="Step judge mode: none|heuristic|llm."),
    judge_cache_path: str = typer.Option(
        ".scopebench_cache", "--judge-cache-path", help="Cache directory for LLM judge outputs."
    ),
    policy_backend: str = typer.Option(
        "python", "--policy-backend", help="Policy backend: python|opa|cedar."
    ),
    json_out: bool = typer.Option(False, "--json", help="Output machine-readable JSON."),
):
    """Run judge quality benchmark against expected decisions."""
    payload = run_judge_bench(
        cases_jsonl,
        evaluate_case=lambda contract, plan: evaluate(
            contract,
            plan,
            policy_backend=policy_backend,
            judge=judge,
            judge_cache_path=judge_cache_path,
        ),
    )

    if json_out:
        console.print_json(json.dumps(payload))
    else:
        console.print(f"[bold]Judge bench[/bold] total={payload['total_cases']}")
        for domain in sorted(payload["per_domain"]):
            stats = payload["per_domain"][domain]
            console.print(
                f" - {domain}: agreement={stats['agreement']:.2f} "
                f"({stats['matched']}/{stats['total']})"
            )
        if payload["error_case_ids"]:
            console.print(f"Errors: {', '.join(payload['error_case_ids'])}")

    if payload["error_case_ids"]:
        raise typer.Exit(code=1)


@app.command("dataset-validate")
def dataset_validate(cases_jsonl: Path = typer.Argument(..., help="Path to cases JSONL.")):
    """Validate a community dataset file against ScopeBench schema and calibration constraints."""
    cases = validate_cases_file(cases_jsonl)
    payload = {"ok": True, "path": str(cases_jsonl), "count": len(cases)}
    console.print_json(json.dumps(payload))


@app.command("dataset-suggest")
def dataset_suggest(
    case_id: str = typer.Option(..., "--id", help="Case id"),
    domain: str = typer.Option(..., help="Case domain"),
    instruction: str = typer.Option(..., help="User instruction / task statement"),
    contract_path: Path = typer.Option(..., "--contract", help="Contract YAML path"),
    plan_path: Path = typer.Option(..., "--plan", help="Plan YAML path"),
    expected_decision: str = typer.Option(..., "--expected-decision", help="ALLOW|ASK|DENY"),
    expected_rationale: str = typer.Option(..., "--expected-rationale", help="Gold rationale"),
    notes: str | None = typer.Option(None, "--notes", help="Optional notes"),
    policy_backend: str = typer.Option("python", "--policy-backend", help="Policy backend used to prefill vectors."),
    out: Path | None = typer.Option(None, "--out", help="Write case JSON to path instead of stdout."),
    append_to: Path | None = typer.Option(None, "--append-to", help="Append generated case to dataset JSONL."),
):
    """Generate a schema-valid case draft with auto-filled step vectors from current evaluator."""
    case = suggest_case(
        case_id=case_id,
        domain=domain,
        instruction=instruction,
        contract_path=contract_path,
        plan_path=plan_path,
        expected_decision=expected_decision,
        expected_rationale=expected_rationale,
        notes=notes,
        policy_backend=policy_backend,
    )
    if append_to is not None:
        append_case(append_to, case)
    payload = {"case": case, "appended_to": str(append_to) if append_to else None}
    if out is not None:
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        console.print_json(json.dumps({"ok": True, "out": str(out), "appended_to": payload["appended_to"]}))
    else:
        console.print_json(json.dumps(payload))


@app.command("dataset-pr")
def dataset_pr(
    title: str = typer.Option(..., "--title", help="Pull request title"),
    body: str = typer.Option(..., "--body", help="Pull request body"),
    json_out: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
):
    """Create a pull request for dataset contributions using gh CLI."""
    url = submit_pull_request(title=title, body=body)
    payload = {"url": url}
    if json_out:
        console.print_json(json.dumps(payload))
    else:
        console.print(f"Created PR: {url}")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8080, "--port"),
    policy_backend: str = typer.Option(
        "python", "--policy-backend", help="Policy backend: python|opa|cedar."
    ),
):
    """Start the ScopeBench HTTP API."""
    from scopebench.server.api import create_app
    import uvicorn

    api = create_app(default_policy_backend=policy_backend)
    uvicorn.run(api, host=host, port=port, log_level="info")




@app.command("plugin-lint")
def plugin_lint(
    bundle_path: Path = typer.Argument(..., help="Path to plugin bundle JSON/YAML to validate."),
    json_out: bool = typer.Option(False, "--json", help="Output machine-readable JSON."),
):
    """Lint a plugin bundle against the plugin schema."""
    raw = yaml.safe_load(bundle_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise typer.BadParameter("Plugin bundle must be a YAML/JSON object")
    errors = lint_plugin_bundle(raw)
    payload = {"ok": not errors, "path": str(bundle_path), "errors": errors}
    if json_out:
        console.print_json(json.dumps(payload))
    else:
        if errors:
            console.print(f"[bold red]Plugin lint failed[/bold red] ({len(errors)} errors)")
            for err in errors:
                console.print(f" - {err}")
            raise typer.Exit(code=1)
        console.print(f"[bold green]Plugin lint passed[/bold green]: {bundle_path}")


@app.command("plugin-generate")
def plugin_generate(
    output_path: Path = typer.Option(Path("plugin_bundle.yaml"), "--out", help="Output bundle path."),
    domain: str | None = typer.Option(None, "--domain", help="Plugin domain (e.g., robotics)."),
    publisher: str | None = typer.Option(None, "--publisher", help="Publisher/org name."),
    plugin_name: str | None = typer.Option(None, "--name", help="Bundle name."),
    version: str = typer.Option("0.1.0", "--version", help="Semver bundle version."),
    tools_csv: str | None = typer.Option(None, "--tools", help="Comma-separated tool ids."),
    effect_mappings: int = typer.Option(1, "--effect-mappings", min=0, help="Number of effect mappings to scaffold."),
    policy_rules: int = typer.Option(1, "--policy-rules", min=0, help="Number of policy rules to scaffold."),
    key_id: str = typer.Option("community-main", "--key-id", help="Signing key id."),
    secret: str | None = typer.Option(None, "--secret", help="Signing shared secret (omit for prompt)."),
    non_interactive: bool = typer.Option(False, "--non-interactive", help="Require explicit flags and skip prompts."),
):
    """Interactive plugin skeleton generator and signer."""

    picked_domain = domain or (domain if non_interactive else typer.prompt("Plugin domain", default="robotics"))
    picked_publisher = publisher or (publisher if non_interactive else typer.prompt("Publisher", default="community"))
    picked_name = plugin_name or (plugin_name if non_interactive else typer.prompt("Bundle name", default=f"{picked_domain}-starter"))

    if non_interactive and (not domain or not publisher or not plugin_name):
        raise typer.BadParameter("--non-interactive requires --domain, --publisher, and --name")

    tool_values = tools_csv
    if tool_values is None and not non_interactive:
        tool_values = typer.prompt("Tool ids (comma-separated)", default="analysis_extension")
    tools = [item.strip() for item in (tool_values or "").split(",") if item.strip()]

    mappings = []
    for idx in range(effect_mappings):
        if non_interactive:
            trigger = tools[idx] if idx < len(tools) else f"tool_{idx+1}"
        else:
            trigger = typer.prompt(f"Effect mapping {idx+1} trigger", default=tools[idx] if idx < len(tools) else f"tool_{idx+1}")
        mappings.append({"trigger": trigger, "axes": {"uncertainty": 0.4, "irreversibility": 0.3}})

    rules = []
    for idx in range(policy_rules):
        default_rule = f"{picked_domain}.rule_{idx+1}"
        rule_id = default_rule if non_interactive else typer.prompt(f"Policy rule {idx+1} id", default=default_rule)
        rules.append({"id": rule_id, "when": {"tool_category": f"{picked_domain}_operations"}, "action": "ASK", "template": "Require operator review + rollback plan."})

    bundle = {
        "name": picked_name,
        "version": version,
        "publisher": picked_publisher,
        "contributions": {
            "tool_categories": {
                f"{picked_domain}_operations": {
                    "description": f"Operations for {picked_domain} plugins"
                }
            },
            "effects_mappings": mappings,
            "scoring_axes": {
                f"{picked_domain}_safety": {"description": f"Safety/risk axis for {picked_domain}"}
            },
            "policy_rules": rules,
        },
        "tools": {},
        "cases": [],
        "metadata": {
            "publish_guidance": [
                "Run `scopebench plugin-lint <bundle>` and `scopebench plugin-harness <bundle>` before release.",
                "Version with semver and publish immutable release artifacts.",
                "Submit listing update to docs/plugin_marketplace.yaml via PR.",
            ]
        },
    }

    for tool in tools:
        bundle["tools"][tool] = {
            "category": f"{picked_domain}_operations",
            "domains": [picked_domain],
            "risk_class": "moderate",
            "priors": {"uncertainty": 0.3, "dependency_creation": 0.2},
        }

    lint_errors = lint_plugin_bundle(bundle)
    if lint_errors:
        raise typer.BadParameter("Generated bundle did not pass lint: " + "; ".join(lint_errors))

    signing_secret = secret
    if signing_secret is None:
        if non_interactive:
            raise typer.BadParameter("--non-interactive requires --secret")
        signing_secret = typer.prompt("Signing secret", hide_input=True)

    signed_bundle = sign_plugin_bundle(bundle, key_id=key_id, secret=signing_secret)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(signed_bundle, sort_keys=False), encoding="utf-8")

    console.print_json(json.dumps({
        "ok": True,
        "output_path": str(output_path),
        "name": picked_name,
        "domain": picked_domain,
        "tools": tools,
        "lint_errors": [],
        "publish_guidance": bundle["metadata"]["publish_guidance"],
    }))


@app.command("plugin-harness")
def plugin_harness(
    bundle_path: Path = typer.Argument(..., help="Path to plugin bundle JSON/YAML."),
    keys_json: str = typer.Option(
        "",
        "--keys-json",
        help="JSON object mapping key_id to shared secret used for signature checks.",
    ),
    golden_cases: Path | None = typer.Option(
        None,
        "--golden-cases",
        help="Optional golden dataset JSONL path (defaults to built-in dataset).",
    ),
    max_golden_cases: int | None = typer.Option(
        None,
        "--max-golden-cases",
        min=1,
        help="Optional cap for golden regression evaluation.",
    ),
    json_out: bool = typer.Option(False, "--json", help="Output machine-readable JSON."),
):
    """Run plugin validation harness: signatures, plans, and golden regression checks."""
    report = run_plugin_test_harness(
        bundle_path,
        keys_json=keys_json,
        golden_cases_path=golden_cases,
        max_golden_cases=max_golden_cases,
    )
    payload = report.to_dict()

    if json_out:
        console.print_json(json.dumps(payload))
    else:
        status = "PASS" if payload["passed"] else "FAIL"
        console.print(f"[bold]Plugin harness[/bold] {status} for {payload['plugin']['name']}")
        for check in payload["checks"]:
            check_status = "PASS" if check["passed"] else "FAIL"
            console.print(f" - {check_status} {check['name']}: {json.dumps(check['details'])}")

    if not payload["passed"]:
        raise typer.Exit(code=1)
