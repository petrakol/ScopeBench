from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from scopebench.bench.judge import run_judge_bench
from scopebench.bench.weekly import replay_benchmark_slice, summarize_weekly_telemetry
from scopebench.runtime.guard import evaluate, evaluate_from_files
from scopebench.scoring.calibration import CalibratedDecisionThresholds
from scopebench.scoring.llm_judge import JudgeMode
from scopebench.tracing.otel import init_tracing

app = typer.Typer(add_completion=False, help="ScopeBench: plan-level proportionality enforcement.")
console = Console()


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
        1.0, "--calibration-scale", min=0.0, help="Scale aggregate scores."
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
    if calibration_scale != 1.0:
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
    json_out: bool = typer.Option(False, "--json", help="Output machine-readable JSON report."),
):
    """Summarize weekly telemetry and replay a fixed benchmark slice."""
    report = summarize_weekly_telemetry(telemetry_jsonl)
    root = Path(__file__).resolve().parents[1]
    replay = replay_benchmark_slice(root)

    payload = {
        "total_runs": report.total_runs,
        "decision_counts": report.decision_counts,
        "top_triggered_rules": [{"rule": r, "count": c} for r, c in report.top_triggered_rules],
        "ask_action_counts": report.ask_action_counts,
        "outcome_counts": report.outcome_counts,
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
        for item in replay:
            status = "PASS" if item.ok else "FAIL"
            console.print(
                f" - {status} {item.name}: decision={item.decision} expected={sorted(item.expected)}"
            )

    if not all(item.ok for item in replay):
        raise typer.Exit(code=1)


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
