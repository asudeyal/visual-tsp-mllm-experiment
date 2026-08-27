import pytest

from src.agents.base import ModelOutputError, parse_hybrid, parse_route, parse_scorer


def test_route_accepts_digit_only_strings() -> None:
    assert parse_route('{"route":["1","22",28,"31"]}') == (1, 22, 28, 31)


@pytest.mark.parametrize("bad", [" 44 ", "44.0", "+44", "-44", "node 44", ""])
def test_route_rejects_non_strict_numeric_strings(bad: str) -> None:
    with pytest.raises(ModelOutputError):
        parse_route('{"route":[1,' + json_quote(bad) + ',2]}')


def json_quote(value: str) -> str:
    import json
    return json.dumps(value)


def test_scorer_accepts_digit_only_strings() -> None:
    ranking, best_id = parse_scorer(
        '{"ranking":["2","1","3"],"best_id":"2"}',
        {1, 2, 3},
    )
    assert ranking == [2, 1, 3]
    assert best_id == 2


def test_hybrid_accepts_digit_only_strings() -> None:
    route, edges = parse_hybrid(
        '{"selected_edges":[["1","2"],["4","5"]],"route":["1","4","3","2","5","1"]}'
    )
    assert route == (1, 4, 3, 2, 5, 1)
    assert edges == ((1, 2), (4, 5))
