from src.evaluation import evaluate_route, validate_route


def test_valid_hamiltonian_cycle(square_problem):
    result = validate_route(square_problem, (1, 2, 3, 4, 1))
    assert result.valid
    assert result.renderable
    assert result.reasons == ()


def test_missing_node_is_renderable_but_invalid(square_problem):
    result = validate_route(square_problem, (1, 2, 4, 1))
    assert not result.valid
    assert result.renderable
    assert result.missing_nodes == (3,)


def test_unknown_node_is_unrenderable(square_problem):
    result = validate_route(square_problem, (1, 2, 99, 4, 1))
    assert not result.valid
    assert not result.renderable
    assert result.unknown_nodes == (99,)


def test_invalid_route_has_no_comparable_objective(square_problem):
    evaluation = evaluate_route(square_problem, (1, 2, 4, 1))
    assert evaluation.distance is None
    assert evaluation.gap_percent is None
