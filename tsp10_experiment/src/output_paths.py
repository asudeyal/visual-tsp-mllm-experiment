"""Deney çıktıları için geriye uyumlu ve güvenli klasör yolları."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")


@dataclass(frozen=True)
class ExperimentPaths:
    """Aynı mantıksal çalıştırmaya ait yöntem klasörleri."""

    output_root: Path
    run_root: Path
    baseline: Path
    zero_shot: Path
    multi_agent1: Path
    multi_agent2: Path


def validate_run_id(run_id: str) -> str:
    """Klasör dışına taşamayan, okunabilir bir çalışma kimliği doğrular."""

    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(
            "--run-id 1-100 karakter olmalı; yalnızca harf, rakam, nokta, "
            "alt çizgi ve tire içerebilir ve harf veya rakamla başlamalıdır."
        )
    return run_id


def build_experiment_paths(
    output_root: Path,
    run_id: str | None,
) -> ExperimentPaths:
    """Her çalıştırmayı runs/<run-id> altında yöntem klasörlerine ayırır."""

    output_root = Path(output_root)
    if run_id is None:
        safe_run_id = "default"
    else:
        safe_run_id = validate_run_id(run_id)

    run_root = output_root / "runs" / safe_run_id
    return ExperimentPaths(
        output_root=output_root,
        run_root=run_root,
        baseline=run_root / "baseline",
        zero_shot=run_root / "zero_shot",
        multi_agent1=run_root / "multi_agent1",
        multi_agent2=run_root / "multi_agent2",
    )
