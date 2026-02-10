from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Protocol, Tuple

from scopebench.contracts import TaskContract
from scopebench.plan import PlanDAG
from scopebench.scoring.axes import ScopeAggregate, ScopeVector


class Decision(str, Enum):
    ALLOW = "ALLOW"
    ASK = "ASK"
    DENY = "DENY"


@dataclass
class PolicyInput:
    policy_input_version: str
    contract: dict
    aggregate: dict
    plan: dict
    vectors: List[dict]
    metadata: Dict[str, object]


@dataclass
class PolicyResult:
    decision: Decision
    reasons: List[str]
    exceeded: Dict[str, Tuple[float, float]]
    asked: Dict[str, float]
    step_vectors: Optional[List[ScopeVector]] = None
    aggregate: Optional[ScopeAggregate] = None
    policy_backend: str = "python"
    policy_version: str = "v1"
    policy_hash: str = ""
    policy_input: Optional[PolicyInput] = None


class PolicyBackend(Protocol):
    name: str

    def evaluate(
        self,
        contract: TaskContract,
        agg: ScopeAggregate,
        step_vectors: Optional[List[ScopeVector]] = None,
        plan: Optional[PlanDAG] = None,
    ) -> PolicyResult: ...
