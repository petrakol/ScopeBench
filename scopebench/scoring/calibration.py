from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from scopebench.scoring.axes import SCOPE_AXES, ScopeAggregate


DEFAULT_DOMAIN = "global"


def _default_axis_scales() -> Dict[str, float]:
    return {axis: 1.0 for axis in SCOPE_AXES}


def _default_axis_biases() -> Dict[str, float]:
    return {axis: 0.0 for axis in SCOPE_AXES}


def _default_axis_threshold_factors() -> Dict[str, float]:
    return {axis: 1.0 for axis in SCOPE_AXES}


@dataclass(frozen=True)
class CalibratedDecisionThresholds:
    """Calibration knobs derived from production telemetry.

    `axis_scale/axis_bias` calibrate aggregate axis values.
    `axis_threshold_factor` calibrates contract thresholds per axis.
    `abstain_uncertainty_threshold` can tighten/loosen selective abstention.
    """

    global_scale: float = 1.0
    axis_scale: Dict[str, float] | None = None
    axis_bias: Dict[str, float] | None = None
    axis_threshold_factor: Dict[str, float] | None = None
    abstain_uncertainty_threshold: float | None = None

    def resolved_axis_scale(self) -> Dict[str, float]:
        values = _default_axis_scales()
        if self.axis_scale:
            values.update({axis: float(scale) for axis, scale in self.axis_scale.items() if axis in values})
        return values

    def resolved_axis_bias(self) -> Dict[str, float]:
        values = _default_axis_biases()
        if self.axis_bias:
            values.update({axis: float(bias) for axis, bias in self.axis_bias.items() if axis in values})
        return values

    def resolved_axis_threshold_factor(self) -> Dict[str, float]:
        values = _default_axis_threshold_factors()
        if self.axis_threshold_factor:
            values.update(
                {
                    axis: float(factor)
                    for axis, factor in self.axis_threshold_factor.items()
                    if axis in values
                }
            )
        return values


@dataclass(frozen=True)
class AxisCalibrationStats:
    triggered: Dict[str, int]
    false_alarms: Dict[str, int]
    overrides: Dict[str, int]
    failures: Dict[str, int]


@dataclass(frozen=True)
class DomainCalibrationStats:
    domain: str
    runs: int
    triggered: Dict[str, int]
    false_alarms: Dict[str, int]
    overrides: Dict[str, int]
    failures: Dict[str, int]


def _iter_jsonl(path: Path) -> Iterable[dict]:
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        yield json.loads(raw)


def _outcome_bucket(row: dict) -> str:
    ask_action = row.get("ask_action")
    feedback = row.get("feedback")
    if isinstance(feedback, dict):
        ask_action = feedback.get("ask_action", ask_action)
    outcome = row.get("outcome")
    if isinstance(feedback, dict):
        outcome = feedback.get("outcome", outcome)
    if isinstance(outcome, str) and outcome in {"tests_fail", "rollback"}:
        return "failure"
    if isinstance(outcome, str) and outcome == "manual_override":
        return "override"
    if isinstance(ask_action, str) and ask_action == "ignored":
        return "override"
    if isinstance(outcome, str) and outcome == "tests_pass":
        return "false_alarm"
    return "neutral"


def _infer_domain(row: dict) -> str:
    for key in ("domain", "task_type"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    telemetry = row.get("telemetry")
    if isinstance(telemetry, dict):
        task_type = telemetry.get("task_type")
        if isinstance(task_type, str) and task_type:
            return task_type
    policy_input = row.get("policy_input")
    if isinstance(policy_input, dict):
        task_type = policy_input.get("task_type")
        if isinstance(task_type, str) and task_type:
            return task_type
    return DEFAULT_DOMAIN


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _fit_single_feature_logistic(samples: List[Tuple[float, int]]) -> Tuple[float, float]:
    """Fit a tiny logistic regression with 1 feature using SGD.

    The model predicts `P(failure | axis_signal)` where feature is in [0, 1].
    """
    if not samples:
        return -2.2, 0.0

    b0 = -2.2
    b1 = 0.0
    lr = 0.15
    for _ in range(40):
        for x, y in samples:
            pred = _sigmoid(b0 + b1 * x)
            err = pred - float(y)
            b0 -= lr * err
            b1 -= lr * err * x
        lr *= 0.94
    return b0, b1


def _axis_signal(row: dict, axis: str) -> float:
    values: List[float] = []
    for container_name in ("asked", "aggregate"):
        container = row.get(container_name)
        if isinstance(container, dict) and axis in container:
            value = container.get(axis)
            if isinstance(value, (int, float)):
                values.append(float(value))
    exceeded = row.get("exceeded")
    if isinstance(exceeded, dict) and axis in exceeded:
        value = exceeded.get(axis)
        if isinstance(value, dict):
            raw = value.get("value")
            if isinstance(raw, (int, float)):
                values.append(float(raw))
        elif isinstance(value, (int, float)):
            values.append(float(value))
    if not values:
        return 0.0
    return _clamp(max(values), 0.0, 1.0)


def _collect_axis_stats(rows: Iterable[dict]) -> Tuple[AxisCalibrationStats, Dict[str, List[Tuple[float, int]]], int]:
    triggered = {axis: 0 for axis in SCOPE_AXES}
    false_alarms = {axis: 0 for axis in SCOPE_AXES}
    overrides = {axis: 0 for axis in SCOPE_AXES}
    failures = {axis: 0 for axis in SCOPE_AXES}
    training: Dict[str, List[Tuple[float, int]]] = {axis: [] for axis in SCOPE_AXES}

    run_count = 0
    for row in rows:
        run_count += 1
        bucket = _outcome_bucket(row)
        label = 1 if bucket == "failure" else 0
        include_sample = bucket in {"failure", "false_alarm", "override"}

        for axis in SCOPE_AXES:
            signal = _axis_signal(row, axis)
            if signal <= 0:
                continue
            triggered[axis] += 1
            if bucket == "false_alarm":
                false_alarms[axis] += 1
            elif bucket == "override":
                overrides[axis] += 1
            elif bucket == "failure":
                failures[axis] += 1
            if include_sample:
                training[axis].append((signal, label))

    return (
        AxisCalibrationStats(
            triggered=triggered,
            false_alarms=false_alarms,
            overrides=overrides,
            failures=failures,
        ),
        training,
        run_count,
    )


def _fit_calibration_from_stats(
    stats: AxisCalibrationStats,
    training: Dict[str, List[Tuple[float, int]]],
) -> CalibratedDecisionThresholds:
    axis_scale: Dict[str, float] = {}
    axis_bias: Dict[str, float] = {}
    axis_threshold_factor: Dict[str, float] = {}

    uncertainty_risk = 0.2
    for axis in SCOPE_AXES:
        denom = max(1, stats.triggered[axis])
        pressure = (stats.failures[axis] - stats.false_alarms[axis] - stats.overrides[axis]) / denom

        b0, b1 = _fit_single_feature_logistic(training[axis])
        predicted_failure = _sigmoid(b0 + b1 * 1.0)
        # 0.2 baseline -> risk appetite where higher failure risk tightens thresholds.
        ml_pressure = (predicted_failure - 0.2) / 0.8
        combined_pressure = 0.55 * pressure + 0.45 * ml_pressure

        if axis == "uncertainty":
            uncertainty_risk = predicted_failure

        axis_scale[axis] = _clamp(1.0 + 0.22 * combined_pressure, 0.6, 1.4)
        axis_bias[axis] = _clamp(0.07 * combined_pressure, -0.2, 0.2)
        axis_threshold_factor[axis] = _clamp(1.0 - 0.24 * combined_pressure, 0.65, 1.3)

    abstain_threshold = _clamp(0.72 - 0.3 * (uncertainty_risk - 0.2), 0.3, 0.95)
    return CalibratedDecisionThresholds(
        global_scale=1.0,
        axis_scale=axis_scale,
        axis_bias=axis_bias,
        axis_threshold_factor=axis_threshold_factor,
        abstain_uncertainty_threshold=abstain_threshold,
    )


def compute_axis_calibration_from_telemetry(path: Path) -> Tuple[CalibratedDecisionThresholds, AxisCalibrationStats]:
    stats, training, _ = _collect_axis_stats(_iter_jsonl(path))
    calibration = _fit_calibration_from_stats(stats, training)
    return calibration, stats


def compute_domain_calibration_from_telemetry(
    path: Path,
) -> Dict[str, Tuple[CalibratedDecisionThresholds, DomainCalibrationStats]]:
    rows_by_domain: Dict[str, List[dict]] = {}
    for row in _iter_jsonl(path):
        domain = _infer_domain(row)
        rows_by_domain.setdefault(domain, []).append(row)

    result: Dict[str, Tuple[CalibratedDecisionThresholds, DomainCalibrationStats]] = {}
    for domain, rows in rows_by_domain.items():
        stats, training, run_count = _collect_axis_stats(rows)
        calibration = _fit_calibration_from_stats(stats, training)
        result[domain] = (
            calibration,
            DomainCalibrationStats(
                domain=domain,
                runs=run_count,
                triggered=stats.triggered,
                false_alarms=stats.false_alarms,
                overrides=stats.overrides,
                failures=stats.failures,
            ),
        )
    return result


def apply_manual_adjustments(
    calibration: CalibratedDecisionThresholds,
    adjustments: dict | None,
) -> CalibratedDecisionThresholds:
    if not adjustments:
        return calibration

    axis_threshold_factor = calibration.resolved_axis_threshold_factor()
    for axis, delta in (adjustments.get("axis_threshold_factor_delta") or {}).items():
        if axis in axis_threshold_factor and isinstance(delta, (int, float)):
            axis_threshold_factor[axis] = _clamp(axis_threshold_factor[axis] + float(delta), 0.5, 1.5)

    axis_scale = calibration.resolved_axis_scale()
    for axis, delta in (adjustments.get("axis_scale_delta") or {}).items():
        if axis in axis_scale and isinstance(delta, (int, float)):
            axis_scale[axis] = _clamp(axis_scale[axis] + float(delta), 0.5, 1.5)

    abstain = calibration.abstain_uncertainty_threshold
    abstain_delta = adjustments.get("abstain_uncertainty_threshold_delta")
    if abstain is not None and isinstance(abstain_delta, (int, float)):
        abstain = _clamp(abstain + float(abstain_delta), 0.2, 0.99)

    return CalibratedDecisionThresholds(
        global_scale=calibration.global_scale,
        axis_scale=axis_scale,
        axis_bias=calibration.resolved_axis_bias(),
        axis_threshold_factor=axis_threshold_factor,
        abstain_uncertainty_threshold=abstain,
    )


def calibration_to_dict(calibration: CalibratedDecisionThresholds) -> dict:
    return {
        "schema_version": "axis_calibration_v3",
        "global_scale": calibration.global_scale,
        "axis_scale": calibration.resolved_axis_scale(),
        "axis_bias": calibration.resolved_axis_bias(),
        "axis_threshold_factor": calibration.resolved_axis_threshold_factor(),
        "abstain_uncertainty_threshold": calibration.abstain_uncertainty_threshold,
    }


def write_calibration_file(path: Path, calibration: CalibratedDecisionThresholds) -> None:
    path.write_text(json.dumps(calibration_to_dict(calibration), indent=2) + "\n", encoding="utf-8")


def load_calibration_file(path: Path) -> CalibratedDecisionThresholds:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return CalibratedDecisionThresholds(
        global_scale=float(payload.get("global_scale", 1.0)),
        axis_scale={axis: float(v) for axis, v in payload.get("axis_scale", {}).items()},
        axis_bias={axis: float(v) for axis, v in payload.get("axis_bias", {}).items()},
        axis_threshold_factor={
            axis: float(v) for axis, v in payload.get("axis_threshold_factor", {}).items()
        },
        abstain_uncertainty_threshold=(
            float(payload["abstain_uncertainty_threshold"])
            if payload.get("abstain_uncertainty_threshold") is not None
            else None
        ),
    )


def apply_calibration(agg: ScopeAggregate, cal: CalibratedDecisionThresholds) -> ScopeAggregate:
    d = agg.as_dict()
    axis_scale = cal.resolved_axis_scale()
    axis_bias = cal.resolved_axis_bias()
    scaled = {
        k: _clamp(v * cal.global_scale * axis_scale[k] + axis_bias[k], 0.0, 1.0) for k, v in d.items()
    }
    return ScopeAggregate(**scaled, n_steps=agg.n_steps)
