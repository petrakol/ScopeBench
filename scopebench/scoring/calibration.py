from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

from scopebench.scoring.axes import SCOPE_AXES, ScopeAggregate


def _default_axis_scales() -> Dict[str, float]:
    return {axis: 1.0 for axis in SCOPE_AXES}


def _default_axis_biases() -> Dict[str, float]:
    return {axis: 0.0 for axis in SCOPE_AXES}


@dataclass(frozen=True)
class CalibratedDecisionThresholds:
    """A small hook to tune thresholding based on observed evals.

    MVP: simple affine adjustments; later: learned calibration, selective prediction.
    """

    global_scale: float = 1.0  # multiply all axis values by this (clipped)
    axis_scale: Dict[str, float] | None = None
    axis_bias: Dict[str, float] | None = None

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


@dataclass(frozen=True)
class AxisCalibrationStats:
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
    outcome = row.get("outcome")
    if isinstance(outcome, str) and outcome in {"tests_fail", "rollback"}:
        return "failure"
    if isinstance(outcome, str) and outcome == "manual_override":
        return "override"
    if isinstance(ask_action, str) and ask_action == "ignored":
        return "override"
    if isinstance(outcome, str) and outcome == "tests_pass":
        return "false_alarm"
    return "neutral"


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def compute_axis_calibration_from_telemetry(path: Path) -> Tuple[CalibratedDecisionThresholds, AxisCalibrationStats]:
    triggered = {axis: 0 for axis in SCOPE_AXES}
    false_alarms = {axis: 0 for axis in SCOPE_AXES}
    overrides = {axis: 0 for axis in SCOPE_AXES}
    failures = {axis: 0 for axis in SCOPE_AXES}

    for row in _iter_jsonl(path):
        bucket = _outcome_bucket(row)
        active_axes = set()
        asked = row.get("asked")
        if isinstance(asked, dict):
            active_axes.update(axis for axis in asked if axis in SCOPE_AXES)
        exceeded = row.get("exceeded")
        if isinstance(exceeded, dict):
            active_axes.update(axis for axis in exceeded if axis in SCOPE_AXES)

        for axis in active_axes:
            triggered[axis] += 1
            if bucket == "false_alarm":
                false_alarms[axis] += 1
            elif bucket == "override":
                overrides[axis] += 1
            elif bucket == "failure":
                failures[axis] += 1

    axis_scale: Dict[str, float] = {}
    axis_bias: Dict[str, float] = {}
    for axis in SCOPE_AXES:
        denom = max(1, triggered[axis])
        pressure = (failures[axis] - false_alarms[axis] - overrides[axis]) / denom
        axis_scale[axis] = _clamp(1.0 + 0.25 * pressure, 0.6, 1.4)
        axis_bias[axis] = _clamp(0.08 * pressure, -0.2, 0.2)

    calibration = CalibratedDecisionThresholds(
        global_scale=1.0,
        axis_scale=axis_scale,
        axis_bias=axis_bias,
    )
    stats = AxisCalibrationStats(
        triggered=triggered,
        false_alarms=false_alarms,
        overrides=overrides,
        failures=failures,
    )
    return calibration, stats


def calibration_to_dict(calibration: CalibratedDecisionThresholds) -> dict:
    return {
        "schema_version": "axis_calibration_v1",
        "global_scale": calibration.global_scale,
        "axis_scale": calibration.resolved_axis_scale(),
        "axis_bias": calibration.resolved_axis_bias(),
    }


def write_calibration_file(path: Path, calibration: CalibratedDecisionThresholds) -> None:
    path.write_text(json.dumps(calibration_to_dict(calibration), indent=2) + "\n", encoding="utf-8")


def load_calibration_file(path: Path) -> CalibratedDecisionThresholds:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return CalibratedDecisionThresholds(
        global_scale=float(payload.get("global_scale", 1.0)),
        axis_scale={axis: float(v) for axis, v in payload.get("axis_scale", {}).items()},
        axis_bias={axis: float(v) for axis, v in payload.get("axis_bias", {}).items()},
    )


def apply_calibration(agg: ScopeAggregate, cal: CalibratedDecisionThresholds) -> ScopeAggregate:
    d = agg.as_dict()
    axis_scale = cal.resolved_axis_scale()
    axis_bias = cal.resolved_axis_bias()
    scaled = {
        k: _clamp(v * cal.global_scale * axis_scale[k] + axis_bias[k], 0.0, 1.0) for k, v in d.items()
    }
    return ScopeAggregate(**scaled, n_steps=agg.n_steps)
