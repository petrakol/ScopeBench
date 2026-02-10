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
from scopebench.policy.backends.base import Decision, PolicyBackend, PolicyResult
from scopebench.policy.backends.external_common import (
    build_external_input,
    normalize_policy_result,
    parse_json_response,
)
from scopebench.scoring.axes import ScopeAggregate, ScopeVector


class OpaPolicyBackend(PolicyBackend):
    name = "opa"
    version = "opa-v1"

    def __init__(self, opa_url: Optional[str] = None) -> None:
        self.opa_url = opa_url or os.getenv("SCOPEBENCH_OPA_URL", "http://localhost:8181/v1/data/scopebench/decision")
        self.fail_open = os.getenv("SCOPEBENCH_POLICY_FAIL_OPEN", "0").lower() in {"1", "true", "yes"}

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
        policy_hash = self._policy_hash()
        payload = {"input": build_external_input(contract, agg, step_vectors=step_vectors, plan=plan)}
        try:
            req = Request(
                self.opa_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"content-type": "application/json"},
                method="POST",
            )
            with urlopen(req, timeout=1.5) as resp:
                parsed = parse_json_response(resp.read())
            result = normalize_policy_result(parsed, self.name, self.version, policy_hash)
            result.step_vectors = step_vectors
            result.aggregate = agg
            return result
        except (URLError, TimeoutError, json.JSONDecodeError):
            if not self.fail_open:
                return PolicyResult(
                    decision=Decision.DENY,
                    reasons=["OPA policy engine unavailable or invalid response"],
                    exceeded={"policy_engine": (1.0, 0.0)},
                    asked={},
                    step_vectors=step_vectors,
                    aggregate=agg,
                    policy_backend=self.name,
                    policy_version=self.version,
                    policy_hash=policy_hash,
                )
            return PolicyResult(
                decision=Decision.ASK,
                reasons=["OPA policy engine unavailable; fail-open escalation"],
                exceeded={},
                asked={"policy_engine": 1.0},
                step_vectors=step_vectors,
                aggregate=agg,
                policy_backend=self.name,
                policy_version=self.version,
                policy_hash=policy_hash,
            )
