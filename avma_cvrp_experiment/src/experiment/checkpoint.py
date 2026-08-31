"""Atomic checkpoint persistence."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from ..schemas import CheckpointState


def save_checkpoint(path: str | Path, state: CheckpointState) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(asdict(state), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_checkpoint(path: str | Path) -> CheckpointState:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return CheckpointState(**data)
