from pathlib import Path
from tempfile import TemporaryDirectory

import run_gemini_multi_agent1 as ma1
from src.core import evaluate_route, parse_tsplib, parse_tsplib_tour


ROOT = Path(__file__).resolve().parents[1]


def test_single_valid_candidate_is_selected_without_scorer_api() -> None:
    instance = parse_tsplib(ROOT / "data/eil51.tsp")
    valid_route = parse_tsplib_tour(ROOT / "data/eil51.opt.tour")
    invalid_route = valid_route[:-2] + [1]
    valid = {
        "candidate_id": 5,
        **evaluate_route(instance, valid_route),
        "route_image": "unused-valid.png",
    }
    invalid = {
        "candidate_id": 2,
        **evaluate_route(instance, invalid_route),
        "route_image": "unused-invalid.png",
    }
    pending = {
        "iteration": 4,
        "critic": {"candidates": [invalid, valid]},
        "scorer_attempts": [],
    }
    original = ma1.request_scorer
    ma1.request_scorer = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("Tek geçerli aday için scorer API çağrılmamalı")
    )
    try:
        with TemporaryDirectory() as temp:
            completed, error = ma1._finish_scorer(
                pending,
                instance=instance,
                model="unused",
                output=Path(temp),
                fallback_route=invalid_route,
                fallback_image=Path("previous.png"),
            )
    finally:
        ma1.request_scorer = original

    assert error is None
    assert completed is not None
    assert completed["scorer"]["best_candidate_id"] == 5
    assert completed["scorer"]["eligible_candidate_ids"] == [5]
    assert completed["scorer"]["excluded_invalid_candidate_ids"] == [2]
    assert completed["scorer"]["api_call"] is None
    assert completed["selected_solution"]["validation"]["is_valid"] is True


def test_invalid_route_has_no_optimum_gap() -> None:
    instance = parse_tsplib(ROOT / "data/eil51.tsp")
    result = evaluate_route(instance, [1, 2, 3, 1])
    assert result["distance"] is not None
    assert result["validation"]["is_valid"] is False
    assert result["gap_to_known_optimum_percent"] is None
