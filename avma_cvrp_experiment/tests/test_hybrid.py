from src.controller.orchestrator import AdaptiveVisualCVRPOrchestrator
from src.schemas import ProblemInstance


def make_problem():
    return ProblemInstance(
        name="hybrid-test",
        dimension=7,
        node_ids=(1, 2, 3, 4, 5, 6, 7),
        coordinates={
            1: (0.0, 0.0),
            2: (1.0, 0.0),
            3: (2.0, 0.0),
            4: (3.0, 0.0),
            5: (4.0, 0.0),
            6: (5.0, 0.0),
            7: (6.0, 0.0),
        },
        depot=1,
        capacity=100,
        demands={1: 0, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1},
        max_vehicles=2,
    )


def make_orchestrator():
    # _hybrid_audit is a pure observer-side method; constructing via __new__
    # avoids needing provider/config setup for these unit tests.
    orchestrator = AdaptiveVisualCVRPOrchestrator.__new__(AdaptiveVisualCVRPOrchestrator)
    orchestrator.problem = make_problem()
    return orchestrator


def test_hybrid_audit_accepts_exact_selected_intra_route_two_opt():
    orchestrator = make_orchestrator()
    old = ((1, 2, 3, 4, 5, 1), (1, 6, 7, 1))
    new = ((1, 2, 4, 3, 5, 1), (1, 6, 7, 1))

    audit = orchestrator._hybrid_audit(old, new, ((2, 3), (4, 5)))

    assert audit["selected_edges_exist_in_input_route"] is True
    assert audit["selected_edges_same_input_route"] is True
    assert audit["selected_edges_non_adjacent"] is True
    assert audit["selected_edges_are_exactly_removed"] is True
    assert audit["exact_single_two_opt_transition"] is True


def test_hybrid_audit_rejects_wrong_selected_edges():
    orchestrator = make_orchestrator()
    old = ((1, 2, 3, 4, 5, 1), (1, 6, 7, 1))
    new = ((1, 2, 4, 3, 5, 1), (1, 6, 7, 1))

    # The actual removed edges are (2,3) and (4,5), not (1,2) and (3,4).
    audit = orchestrator._hybrid_audit(old, new, ((1, 2), (3, 4)))

    assert audit["transition_is_exact_single_two_opt"] is True
    assert audit["selected_edges_are_exactly_removed"] is False
    assert audit["exact_single_two_opt_transition"] is False


def test_hybrid_audit_rejects_inter_route_exchange():
    orchestrator = make_orchestrator()
    old = ((1, 2, 3, 4, 1), (1, 5, 6, 7, 1))
    new = ((1, 2, 5, 4, 1), (1, 3, 6, 7, 1))

    audit = orchestrator._hybrid_audit(old, new, ((2, 3), (1, 5)))

    assert audit["exact_single_two_opt_transition"] is False


def test_hybrid_audit_rejects_adjacent_selected_edges():
    orchestrator = make_orchestrator()
    old = ((1, 2, 3, 4, 5, 1), (1, 6, 7, 1))
    new = ((1, 3, 2, 4, 5, 1), (1, 6, 7, 1))

    # This is a degenerate/adjacent edge selection, not a valid 2-opt cut.
    audit = orchestrator._hybrid_audit(old, new, ((1, 2), (2, 3)))

    assert audit["selected_edges_non_adjacent"] is False
    assert audit["exact_single_two_opt_transition"] is False
