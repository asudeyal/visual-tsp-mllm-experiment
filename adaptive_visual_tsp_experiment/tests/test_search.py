from src.search import canonicalize_route, detect_structural_stagnation, edge_similarity, is_exact_two_opt_transition


def test_reverse_cycle_has_same_canonical_form():
    a = (1, 2, 3, 4, 1)
    b = (1, 4, 3, 2, 1)
    assert canonicalize_route(a, 1) == canonicalize_route(b, 1)


def test_edge_similarity_ignores_direction():
    assert edge_similarity((1, 2, 3, 4, 1), (1, 4, 3, 2, 1)) == 1.0


def test_structural_stagnation_from_repetition():
    route = (1, 2, 3, 4, 1)
    result = detect_structural_stagnation(
        [route] * 5,
        depot=1,
        window=5,
        similarity_threshold=0.90,
        max_unique_routes=2,
    )
    assert result.stagnated
    assert result.exact_repeat_signal


def test_exact_two_opt_audit():
    old = (1, 2, 3, 4, 5, 1)
    # reverse segment 3..4 after cuts (2,3) and (4,5)
    new = (1, 2, 4, 3, 5, 1)
    assert is_exact_two_opt_transition(old, new)
