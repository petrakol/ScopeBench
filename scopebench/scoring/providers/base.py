from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class ProviderResult:
    raw_text: str
    provider_name: str


class LLMProvider(Protocol):
    def complete(self, prompt: str) -> ProviderResult:
        ...
