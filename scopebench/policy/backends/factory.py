from __future__ import annotations

import os

from scopebench.policy.backends.base import PolicyBackend
from scopebench.policy.backends.cedar_backend import CedarPolicyBackend
from scopebench.policy.backends.opa_backend import OpaPolicyBackend
from scopebench.policy.backends.python_backend import PythonPolicyBackend


def get_policy_backend(name: str | None = None) -> PolicyBackend:
    selected = (name or os.getenv("SCOPEBENCH_POLICY_BACKEND", "python")).strip().lower()
    if selected == "python":
        return PythonPolicyBackend()
    if selected == "opa":
        return OpaPolicyBackend()
    if selected == "cedar":
        return CedarPolicyBackend()
    raise ValueError(f"Unknown policy backend: {selected}")
