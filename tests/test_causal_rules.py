from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scopebench.scoring.causal import list_causal_rules  # noqa: E402


def test_causal_rules_list_contains_lab():
    rules = list_causal_rules()
    assert "lab" in rules
