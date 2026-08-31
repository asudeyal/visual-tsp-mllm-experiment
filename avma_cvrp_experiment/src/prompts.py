"""Prompt loading and provenance hashing."""

from __future__ import annotations

import hashlib
from pathlib import Path


class PromptSet:
    def __init__(self, root: str | Path, version: str) -> None:
        self.root = Path(root) / version
        if not self.root.is_dir():
            raise FileNotFoundError(f"Prompt set bulunamadı: {self.root}")
        self.version = version
        self.common = self._read("common_policy.txt")

    def _read(self, name: str) -> str:
        return (self.root / name).read_text(encoding="utf-8").strip()

    def role(self, role_name: str) -> str:
        return self._read(f"{role_name}.txt")

    def combined(self, role_name: str) -> str:
        return f"{self.common}\n\n{self.role(role_name)}".strip()

    def hashes(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for path in sorted(self.root.glob("*.txt")):
            result[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return result
