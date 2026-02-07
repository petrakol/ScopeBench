from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional


@dataclass(frozen=True)
class ScopeBenchCase:
    id: str
    domain: str
    instruction: str
    contract: dict
    plan: dict
    expected_decision: str  # ALLOW / ASK / DENY
    notes: Optional[str] = None


def load_cases(path: Path) -> List[ScopeBenchCase]:
    cases: List[ScopeBenchCase] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            cases.append(
                ScopeBenchCase(
                    id=obj["id"],
                    domain=obj.get("domain", "unknown"),
                    instruction=obj["instruction"],
                    contract=obj["contract"],
                    plan=obj["plan"],
                    expected_decision=obj["expected_decision"],
                    notes=obj.get("notes"),
                )
            )
    return cases


def default_cases_path() -> Path:
    return Path(__file__).resolve().parent / "cases" / "examples.jsonl"
