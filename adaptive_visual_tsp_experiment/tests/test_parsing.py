import pytest

from src.agents.base import ModelOutputError, parse_hybrid, parse_route, parse_scorer


def test_route_parser_accepts_json_fence():
    assert parse_route('```json\n{"route":[1,2,3,1]}\n```') == (1, 2, 3, 1)


def test_route_parser_rejects_prose_without_json():
    with pytest.raises(ModelOutputError):
        parse_route("I would use 1-2-3-1")


def test_scorer_requires_all_candidate_ids():
    ranking, best = parse_scorer('{"ranking":[3,1,2],"best_id":3}', {1, 2, 3})
    assert ranking == [3, 1, 2]
    assert best == 3


def test_hybrid_parser():
    route, edges = parse_hybrid('{"selected_edges":[[1,2],[4,5]],"route":[1,2,4,3,5,1]}')
    assert route == (1, 2, 4, 3, 5, 1)
    assert edges == ((1, 2), (4, 5))
