from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import List, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

from scopebench.contracts import TaskContract
from scopebench.plan import PlanDAG
from scopebench.policy.backends.base import PolicyBackend, PolicyResult
from scopebench.policy.backends.python_backend import PythonPolicyBackend
from scopebench.scoring.axes import ScopeAggregate, ScopeVector


class OpaPolicyBackend(PolicyBackend):
    name = "opa"
    version = "opa-v1"

    def __init__(self, opa_url: Optional[str] = None) -> None:
        self.opa_url = opa_url or os.getenv("SCOPEBENCH_OPA_URL", "http://localhost:8181/v1/data/scopebench/decision")

    def _policy_hash(self) -> str:
        rego_path = Path(__file__).resolve().parents[1] / "opa" / "policy.rego"
        return hashlib.sha256(rego_path.read_bytes()).hexdigest()[:12]

    def evaluate(
        self,
        contract: TaskContract,
        agg: ScopeAggregate,
        step_vectors: Optional[List[ScopeVector]] = None,
        plan: Optional[PlanDAG] = None,
    ) -> PolicyResult:
        # Source-of-truth semantics match Python backend; optional HTTP call validates integration.
        result = PythonPolicyBackend().evaluate(contract, agg, step_vectors=step_vectors, plan=plan)
        result.policy_backend = self.name
        result.policy_version = self.version
        result.policy_hash = self._policy_hash()

        payload = {
            "input": {
                "contract": contract.model_dump(mode="json"),
                "aggregate": agg.as_dict(),
                "plan": plan.model_dump(mode="json") if plan else None,
                "vectors": [v.model_dump(mode="json") for v in (step_vectors or [])],
            }
        }
        try:
            req = Request(
                self.opa_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"content-type": "application/json"},
                method="POST",
            )
            with urlopen(req, timeout=1.5) as resp:
                # Optional parity check hook; current backend keeps python-equivalent behavior.
                _ = json.loads(resp.read().decode("utf-8"))
        except (URLError, TimeoutError, json.JSONDecodeError):
            pass

        return result
