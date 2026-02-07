from __future__ import annotations

"""
Executor stub.

ScopeBench is primarily a *pre-execution proportionality gate*.
In a full system, the executor would:

- run tool calls in a sandbox
- enforce capability-based permissions
- log all actions and artifacts
- support deterministic replay

This file exists as a placeholder to encourage a clean separation:
Planner -> ScopeGate -> Executor
"""
