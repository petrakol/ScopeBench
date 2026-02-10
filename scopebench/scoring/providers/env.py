from __future__ import annotations

import os

from scopebench.scoring.providers.base import ProviderResult


class EnvJSONProvider:
    """Offline-friendly provider reading a JSON response from env var."""

    def __init__(self, env_var: str = "SCOPEBENCH_LLM_JUDGE_RESPONSE"):
        self.env_var = env_var

    def complete(self, prompt: str) -> ProviderResult:
        del prompt
        raw = os.getenv(self.env_var, "")
        if not raw:
            raise RuntimeError(f"{self.env_var} is not set")
        return ProviderResult(raw_text=raw, provider_name="env_json")
