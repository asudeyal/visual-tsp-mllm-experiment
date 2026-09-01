from src.search import (
    canonicalize_routes,
    edge_similarity,
    is_exact_two_opt_transition,
)


def test_canonical_routes_ignore_vehicle_order():
    routes_a = ((1, 2, 3, 1), (1, 4, 5, 1))
    routes_b = ((1, 5, 4, 1), (1, 3, 2, 1))

    assert canonicalize_routes(routes_a, 1) == canonicalize_routes(routes_b, 1)


def test_exact_two_opt_accepts_single_intra_route_reversal():
    old = ((1, 2, 3, 4, 5, 1), (1, 6, 7, 1))
    new = ((1, 2, 4, 3, 5, 1), (1, 6, 7, 1))

    assert is_exact_two_opt_transition(old, new) is True


def test_two_opt_rejects_inter_route_customer_exchange():
    old = ((1, 2, 3, 4, 1), (1, 5, 6, 7, 1))
    new = ((1, 2, 5, 4, 1), (1, 3, 6, 7, 1))

    assert is_exact_two_opt_transition(old, new) is False


def test_two_opt_rejects_multiple_route_changes():
    old = ((1, 2, 3, 4, 1), (1, 5, 6, 7, 1))
    new = ((1, 3, 2, 4, 1), (1, 6, 5, 7, 1))

    assert is_exact_two_opt_transition(old, new) is False


def test_edge_similarity_is_one_for_same_structure():
    routes_a = ((1, 2, 3, 1), (1, 4, 5, 1))
    routes_b = ((1, 5, 4, 1), (1, 3, 2, 1))

    assert edge_similarity(routes_a, routes_b) == 1.0
