from __future__ import annotations

from typing import Optional

from scopebench.plan import PlanStep
from scopebench.scoring.axes import ScopeVector

# LLM-judge stub.
#
# This repository is offline and does not call external model APIs by default.
# However, in a production system you may want an LLM-based classifier to:
#
# - refine axis scoring from unstructured plans
# - produce better rationales
# - estimate uncertainty/confidence
#
# Implement `judge_step` to plug in your provider of choice.


def judge_step(step: PlanStep) -> Optional[ScopeVector]:
    # Returning None means "no override".
    return None
