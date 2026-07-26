"""Bir deney koşusunun değişmez problem manifestini yönetir."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.core import read_json, write_json
from src.problem_instance import (
    ProblemInstance,
    ProblemSource,
    ReferenceSolution,
    ReferenceType,
)


MANIFEST_SCHEMA_VERSION = "1.0"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_problem_payload(
    problem: ProblemInstance,
) -> dict[str, Any]:
    return {
        "name": problem.name,
        "source_type": problem.source_type.value,
        "dimension": problem.dimension,
        "depot_id": problem.depot_id,
        "edge_weight_type": problem.edge_weight_type,
        "seed": problem.seed,
        "coordinates": [
            [
                node_id,
                problem.coordinates[node_id][0],
                problem.coordinates[node_id][1],
            ]
            for node_id in problem.node_ids
        ],
    }


def problem_fingerprint(
    problem: ProblemInstance,
) -> str:
    encoded = json.dumps(
        _canonical_problem_payload(problem),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _copy_input(
    source: Path,
    destination: Path,
) -> Path:
    source = source.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if source.read_bytes() != destination.read_bytes():
            raise FileExistsError(
                f"Run girdisi aynı adla farklı içerikte mevcut: {destination}"
            )
        return destination
    shutil.copy2(source, destination)
    return destination


def snapshot_problem_inputs(
    problem: ProblemInstance,
    run_dir: Path,
) -> ProblemInstance:
    """TSPLIB girdilerini koşu klasörüne kopyalayıp yolları sabitler."""

    if problem.source_type is not ProblemSource.TSPLIB:
        return problem
    if problem.source_file is None:
        raise ValueError("TSPLIB probleminde source_file bulunmalıdır.")

    inputs_dir = run_dir / "inputs"
    copied_instance = _copy_input(
        problem.source_file,
        inputs_dir / problem.source_file.name,
    )
    copied_tour: Path | None = None
    if problem.optimal_tour_file is not None:
        copied_tour = _copy_input(
            problem.optimal_tour_file,
            inputs_dir / problem.optimal_tour_file.name,
        )

    reference = problem.reference
    if reference is not None and copied_tour is not None:
        reference = replace(
            reference,
            source_file=copied_tour,
        )

    return replace(
        problem,
        source_file=copied_instance,
        optimal_tour_file=copied_tour,
        reference=reference,
    )


def _relative_path(
    path: Path | None,
    run_dir: Path,
) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(run_dir.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"Manifest yolu run klasörü dışında: {path}"
        ) from exc


def problem_to_manifest(
    problem: ProblemInstance,
    run_dir: Path,
) -> dict[str, Any]:
    reference = problem.reference
    return {
        "name": problem.name,
        "source_type": problem.source_type.value,
        "dimension": problem.dimension,
        "depot_id": problem.depot_id,
        "edge_weight_type": problem.edge_weight_type,
        "seed": problem.seed,
        "fingerprint_sha256": problem_fingerprint(problem),
        "source_file": _relative_path(problem.source_file, run_dir),
        "optimal_tour_file": _relative_path(
            problem.optimal_tour_file,
            run_dir,
        ),
        "coordinates": [
            {
                "node_id": node_id,
                "x": problem.coordinates[node_id][0],
                "y": problem.coordinates[node_id][1],
            }
            for node_id in problem.node_ids
        ],
        "reference": (
            {
                "type": reference.reference_type.value,
                "distance": reference.distance,
                "is_proven_optimal": reference.is_proven_optimal,
                "route": (
                    list(reference.route)
                    if reference.route is not None
                    else None
                ),
                "source_file": _relative_path(
                    reference.source_file,
                    run_dir,
                ),
            }
            if reference is not None
            else None
        ),
    }


def problem_from_manifest(
    manifest: dict[str, Any],
    run_dir: Path,
) -> ProblemInstance:
    data = manifest["problem"]
    reference_data = data.get("reference")
    reference = (
        ReferenceSolution(
            reference_type=ReferenceType(reference_data["type"]),
            distance=float(reference_data["distance"]),
            is_proven_optimal=bool(
                reference_data["is_proven_optimal"]
            ),
            route=(
                tuple(int(node) for node in reference_data["route"])
                if reference_data.get("route") is not None
                else None
            ),
            source_file=(
                run_dir / reference_data["source_file"]
                if reference_data.get("source_file")
                else None
            ),
        )
        if reference_data is not None
        else None
    )
    problem = ProblemInstance(
        name=data["name"],
        source_type=ProblemSource(data["source_type"]),
        dimension=int(data["dimension"]),
        depot_id=int(data["depot_id"]),
        edge_weight_type=data["edge_weight_type"],
        coordinates={
            int(item["node_id"]): (
                float(item["x"]),
                float(item["y"]),
            )
            for item in data["coordinates"]
        },
        seed=(
            int(data["seed"])
            if data.get("seed") is not None
            else None
        ),
        source_file=(
            run_dir / data["source_file"]
            if data.get("source_file")
            else None
        ),
        optimal_tour_file=(
            run_dir / data["optimal_tour_file"]
            if data.get("optimal_tour_file")
            else None
        ),
        reference=reference,
    )
    actual = problem_fingerprint(problem)
    if actual != data["fingerprint_sha256"]:
        raise ValueError(
            "Manifest problem fingerprint doğrulaması başarısız."
        )
    return problem


def build_run_manifest(
    *,
    run_id: str,
    problem: ProblemInstance,
    run_dir: Path,
    input_request: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "created_at_utc": _utc_now_iso(),
        "problem": problem_to_manifest(problem, run_dir),
        "input_request": input_request,
        "baseline": baseline,
    }


def write_run_manifest(
    path: Path,
    manifest: dict[str, Any],
) -> None:
    if path.exists():
        existing = read_json(path)
        old_fingerprint = existing.get("problem", {}).get(
            "fingerprint_sha256"
        )
        new_fingerprint = manifest.get("problem", {}).get(
            "fingerprint_sha256"
        )
        if old_fingerprint != new_fingerprint:
            raise FileExistsError(
                "Bu run-id farklı bir problem için zaten kullanılmış."
            )
        manifest["created_at_utc"] = existing.get(
            "created_at_utc",
            manifest["created_at_utc"],
        )
    write_json(path, manifest)


def load_run_problem(
    manifest_path: Path,
) -> tuple[dict[str, Any], ProblemInstance]:
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "Desteklenmeyen run manifest şema sürümü: "
            f"{manifest.get('schema_version')}"
        )
    run_dir = manifest_path.parent
    return manifest, problem_from_manifest(manifest, run_dir)
