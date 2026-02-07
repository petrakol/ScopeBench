from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

from scopebench.scoring.axes import ScopeAggregate


@dataclass(frozen=True)
class CalibratedDecisionThresholds:
    """A small hook to tune thresholding based on observed evals.

    MVP: simple affine adjustments; later: learned calibration, selective prediction.
    """

    global_scale: float = 1.0  # multiply all axis values by this (clipped)


def apply_calibration(agg: ScopeAggregate, cal: CalibratedDecisionThresholds) -> ScopeAggregate:
    d = agg.as_dict()
    scaled = {k: min(1.0, v * cal.global_scale) for k, v in d.items()}
    return ScopeAggregate(**scaled, n_steps=agg.n_steps)
