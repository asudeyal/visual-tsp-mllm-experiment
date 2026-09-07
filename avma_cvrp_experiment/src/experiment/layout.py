"""Run-layout helpers for shared multi-provider AVMA experiments."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Iterable


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")


def problem_alias(problem_name: str) -> str:
    """Compact readable problem label, e.g. ``eil51`` -> ``eil_51``."""

    value = problem_name.strip().lower()
    value = re.sub(r"(?<=[a-z])(?=\d)", "_", value)
    return slug(value)


def automatic_run_id(
    problem_name: str,
    run_label: str,
    *,
    now: datetime | None = None,
) -> str:
    stamp = (now or datetime.now()).strftime("%y%m%d")
    return f"{stamp}-{problem_alias(problem_name)}_{slug(run_label.lower())}"


def provider_model_dir(run_root: str | Path, provider: str, model: str) -> Path:
    root = Path(run_root)
    return root / "providers" / slug(provider.lower()) / slug(model)


def is_legacy_run(run_dir: str | Path) -> bool:
    path = Path(run_dir)
    return (path / "run_manifest.json").exists() and (
        (path / "iterations").exists()
        or (path / "initializer").exists()
        or (path / "summary.json").exists()
    )


def is_compact_run_root(run_dir: str | Path) -> bool:
    path = Path(run_dir)
    return (path / "run.json").exists() and (path / "providers").exists()


def is_compact_model_run(run_dir: str | Path) -> bool:
    path = Path(run_dir)
    return (path / "state.json").exists() or (path / "trace.jsonl").exists()


def discover_model_runs(run_root: str | Path) -> list[Path]:
    root = Path(run_root) / "providers"
    if not root.exists():
        return []
    found: list[Path] = []
    for state_path in root.glob("*/*/state.json"):
        found.append(state_path.parent)
    for manifest in root.glob("*/*/run_manifest.json"):
        model_dir = manifest.parent
        if model_dir not in found and ((model_dir / "iterations").exists() or (model_dir / "summary.json").exists()):
            found.append(model_dir)
    return sorted(found)


def model_run_labels(paths: Iterable[Path]) -> list[str]:
    labels: list[str] = []
    for path in paths:
        try:
            provider = path.parent.name
            model = path.name
            labels.append(f"{provider}/{model}")
        except Exception:
            labels.append(str(path))
    return labels
