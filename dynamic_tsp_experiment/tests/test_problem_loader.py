from pathlib import Path

import pytest

from src.problem_instance import (
    ProblemSource,
    ReferenceType,
)
from src.problem_loader import (
    generate_random_problem,
    load_tsplib_problem,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "num_nodes",
    [10, 20, 50],
)
def test_random_problem_uses_requested_node_count(
    num_nodes: int,
) -> None:
    problem = generate_random_problem(
        num_nodes,
        seed=42,
    )

    assert problem.dimension == num_nodes
    assert problem.node_ids == tuple(range(num_nodes))
    assert problem.depot_id == 0
    assert problem.source_type is ProblemSource.RANDOM
    assert problem.edge_weight_type == "EUC_2D_FLOAT"


def test_random_problem_is_reproducible_with_same_seed() -> None:
    first = generate_random_problem(
        20,
        seed=42,
    )
    second = generate_random_problem(
        20,
        seed=42,
    )

    assert first.coordinates == second.coordinates


def test_random_problem_changes_with_different_seed() -> None:
    first = generate_random_problem(
        20,
        seed=42,
    )
    second = generate_random_problem(
        20,
        seed=43,
    )

    assert first.coordinates != second.coordinates


def test_random_problem_rejects_invalid_node_count() -> None:
    with pytest.raises(
        ValueError,
        match="en az 2",
    ):
        generate_random_problem(1)


def test_eil51_is_loaded_as_generic_tsplib_problem() -> None:
    problem = load_tsplib_problem(
        ROOT / "data" / "eil51.tsp",
        optimal_tour_file=(
            ROOT / "data" / "eil51.opt.tour"
        ),
    )

    assert problem.name == "eil51"
    assert problem.dimension == 51
    assert problem.node_ids == tuple(range(1, 52))
    assert problem.depot_id == 1
    assert problem.edge_weight_type == "EUC_2D"
    assert problem.source_type is ProblemSource.TSPLIB

    assert problem.reference is not None
    assert (
        problem.reference.reference_type
        is ReferenceType.TSPLIB_KNOWN_OPTIMUM
    )
    assert problem.reference.is_proven_optimal is True
    assert problem.reference.distance == 426.0
    assert problem.reference.route is not None
    assert problem.reference.route[0] == 1
    assert problem.reference.route[-1] == 1
    assert len(problem.reference.route) == 52


def test_unsupported_tsplib_distance_type_is_rejected(
    tmp_path: Path,
) -> None:
    instance_file = tmp_path / "unsupported.tsp"

    instance_file.write_text(
        "\n".join(
            [
                "NAME: unsupported",
                "TYPE: TSP",
                "DIMENSION: 2",
                "EDGE_WEIGHT_TYPE: ATT",
                "NODE_COORD_SECTION",
                "1 0 0",
                "2 1 1",
                "EOF",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="ATT desteklenmiyor",
    ):
        load_tsplib_problem(instance_file)
