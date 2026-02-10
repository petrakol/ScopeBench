from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Optional

from scopebench.contracts import TaskContract
from scopebench.plan import PlanDAG
from scopebench.policy.backends.base import PolicyBackend, PolicyResult
from scopebench.policy.backends.python_backend import PythonPolicyBackend
from scopebench.scoring.axes import ScopeAggregate, ScopeVector


class CedarPolicyBackend(PolicyBackend):
    name = "cedar"
    version = "cedar-v1"

    def _policy_hash(self) -> str:
        policy_path = Path(__file__).resolve().parents[1] / "cedar" / "policy.cedar"
        schema_path = Path(__file__).resolve().parents[1] / "cedar" / "schema.json"
        digest = hashlib.sha256()
        digest.update(policy_path.read_bytes())
        digest.update(schema_path.read_bytes())
        return digest.hexdigest()[:12]

    def evaluate(
        self,
        contract: TaskContract,
        agg: ScopeAggregate,
        step_vectors: Optional[List[ScopeVector]] = None,
        plan: Optional[PlanDAG] = None,
    ) -> PolicyResult:
        # Current Cedar adapter keeps parity with Python semantics for supported rules.
        result = PythonPolicyBackend().evaluate(contract, agg, step_vectors=step_vectors, plan=plan)
        result.policy_backend = self.name
        result.policy_version = self.version
        result.policy_hash = self._policy_hash()
        return result
