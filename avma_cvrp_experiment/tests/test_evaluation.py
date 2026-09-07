from src.evaluation import validate_cvrp_routes
from src.schemas import ProblemInstance


def make_problem():
    return ProblemInstance(
        name="test",
        dimension=6,
        node_ids=(1, 2, 3, 4, 5, 6),
        coordinates={
            1: (0.0, 0.0),
            2: (1.0, 0.0),
            3: (2.0, 0.0),
            4: (0.0, 1.0),
            5: (1.0, 1.0),
            6: (2.0, 1.0),
        },
        depot=1,
        capacity=10,
        demands={
            1: 0,
            2: 2,
            3: 3,
            4: 2,
            5: 1,
            6: 2,
        },
        max_vehicles=3,
    )


def test_depot_is_not_counted_as_customer():
    problem = make_problem()

    routes = [
        [1, 2, 3, 1],
        [1, 4, 5, 6, 1],
    ]

    result = validate_cvrp_routes(problem, routes)

    assert result.valid is True
    assert result.duplicate_nodes == ()
    assert result.missing_nodes == ()
    assert result.vehicle_count == 2


def test_duplicate_customer_is_invalid():
    problem = make_problem()

    routes = [
        [1, 2, 3, 1],
        [1, 3, 4, 5, 6, 1],
    ]

    result = validate_cvrp_routes(problem, routes)

    assert result.valid is False
    assert 3 in result.duplicate_nodes
    assert "duplicate_nodes" in result.reasons


def test_missing_customer_is_invalid():
    problem = make_problem()

    routes = [
        [1, 2, 3, 1],
        [1, 4, 5, 1],
    ]

    result = validate_cvrp_routes(problem, routes)

    assert result.valid is False
    assert 6 in result.missing_nodes
    assert "missing_nodes" in result.reasons


def test_capacity_violation_is_invalid():
    problem = make_problem()

    problem = ProblemInstance(
        name=problem.name,
        dimension=problem.dimension,
        node_ids=problem.node_ids,
        coordinates=problem.coordinates,
        depot=problem.depot,
        capacity=9,
        demands=problem.demands,
        max_vehicles=problem.max_vehicles,
        edge_weight_type=problem.edge_weight_type,
        source_path=problem.source_path,
        source_sha256=problem.source_sha256,
        reference_optimum=problem.reference_optimum,
    )

    routes = [
        [1, 2, 3, 4, 5, 6, 1],
    ]

    result = validate_cvrp_routes(problem, routes)

    assert result.valid is False
    assert "capacity_exceeded" in result.reasons
    assert result.route_loads == (10,)
    assert result.capacity_exceeded_route_indices == (0,)


def test_wrong_start_depot_is_invalid():
    problem = make_problem()

    routes = [
        [2, 3, 1],
        [1, 4, 5, 6, 1],
    ]

    result = validate_cvrp_routes(problem, routes)

    assert result.valid is False
    assert "wrong_start_depot" in result.reasons


def test_wrong_end_depot_is_invalid():
    problem = make_problem()

    routes = [
        [1, 2, 3],
        [1, 4, 5, 6, 1],
    ]

    result = validate_cvrp_routes(problem, routes)

    assert result.valid is False
    assert "wrong_end_depot" in result.reasons


def test_vehicle_limit_is_enforced():
    problem = make_problem()

    routes = [
        [1, 2, 1],
        [1, 3, 1],
        [1, 4, 1],
        [1, 5, 6, 1],
    ]

    result = validate_cvrp_routes(problem, routes)

    assert result.valid is False
    assert result.vehicle_count == 4
    assert "max_vehicles_exceeded" in result.reasons