from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional


class JudgeCache:
    """Content-addressed cache for LLM judge outputs."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_path(cls, path: Optional[str] = None) -> "JudgeCache":
        return cls(Path(path or ".scopebench_cache"))

    @staticmethod
    def make_key(payload: Dict[str, Any]) -> str:
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _entry_path(self, key: str) -> Path:
        return self.root / "llm_judge" / f"{key}.json"

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        path = self._entry_path(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def set(self, key: str, payload: Dict[str, Any]) -> None:
        path = self._entry_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
