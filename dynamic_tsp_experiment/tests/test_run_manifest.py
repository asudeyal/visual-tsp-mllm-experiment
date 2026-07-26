from dataclasses import replace
from pathlib import Path

import pytest

from src.problem_instance import (
    ReferenceSolution,
    ReferenceType,
)
from src.problem_loader import (
    generate_random_problem,
    load_tsplib_problem,
)
from src.run_manifest import (
    build_run_manifest,
    load_run_problem,
    problem_fingerprint,
    snapshot_problem_inputs,
    write_run_manifest,
)


ROOT = Path(__file__).resolve().parents[1]


def test_random_problem_manifest_round_trip(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "runs" / "random20"
    problem = generate_random_problem(20, seed=42)
    problem = replace(
        problem,
        reference=ReferenceSolution(
            reference_type=ReferenceType.OR_TOOLS_HEURISTIC,
            distance=12.5,
            is_proven_optimal=False,
            route=tuple(
                [
                    *problem.node_ids,
                    problem.depot_id,
                ]
            ),
        ),
    )
    manifest = build_run_manifest(
        run_id="random20",
        problem=problem,
        run_dir=run_dir,
        input_request={
            "mode": "random",
            "num_nodes": 20,
            "seed": 42,
        },
        baseline={
            "method": "or_tools",
            "distance": 12.5,
        },
    )
    manifest_path = run_dir / "run_manifest.json"
    write_run_manifest(manifest_path, manifest)

    loaded_manifest, loaded_problem = load_run_problem(
        manifest_path
    )

    assert loaded_manifest["run_id"] == "random20"
    assert loaded_problem.coordinates == problem.coordinates
    assert loaded_problem.reference is not None
    assert loaded_problem.reference.is_proven_optimal is False
    assert (
        problem_fingerprint(loaded_problem)
        == problem_fingerprint(problem)
    )


def test_tsplib_inputs_are_snapshotted_and_round_trip(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "runs" / "eil51"
    original = load_tsplib_problem(
        ROOT / "data" / "eil51.tsp",
        optimal_tour_file=ROOT / "data" / "eil51.opt.tour",
    )
    problem = snapshot_problem_inputs(
        original,
        run_dir,
    )
    manifest = build_run_manifest(
        run_id="eil51",
        problem=problem,
        run_dir=run_dir,
        input_request={"mode": "tsplib"},
        baseline={"method": "or_tools"},
    )
    manifest_path = run_dir / "run_manifest.json"
    write_run_manifest(manifest_path, manifest)

    _, loaded = load_run_problem(manifest_path)

    assert (run_dir / "inputs" / "eil51.tsp").exists()
    assert (run_dir / "inputs" / "eil51.opt.tour").exists()
    assert loaded.reference is not None
    assert loaded.reference.distance == 426
    assert loaded.coordinates == original.coordinates


def test_existing_run_id_rejects_different_problem(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "runs" / "same_id"
    first = generate_random_problem(10, seed=42)
    second = generate_random_problem(10, seed=43)
    first_manifest = build_run_manifest(
        run_id="same_id",
        problem=first,
        run_dir=run_dir,
        input_request={"mode": "random"},
        baseline={},
    )
    second_manifest = build_run_manifest(
        run_id="same_id",
        problem=second,
        run_dir=run_dir,
        input_request={"mode": "random"},
        baseline={},
    )
    manifest_path = run_dir / "run_manifest.json"
    write_run_manifest(
        manifest_path,
        first_manifest,
    )

    with pytest.raises(
        FileExistsError,
        match="farklı bir problem",
    ):
        write_run_manifest(
            manifest_path,
            second_manifest,
        )
