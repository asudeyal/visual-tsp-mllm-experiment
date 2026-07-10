from src.tsp_core import (
    generate_locations,
    percentage_gap,
    route_distance,
    solve_exact_tsp,
    solve_ortools_tsp,
    validate_tsp_route,
)
from src.llm_routes import (
    RouteParseError,
    critic_prompt,
    image_mime_type,
    initializer_prompt,
    parse_single_salesman_route,
)


def test_validate_tsp_route_accepts_complete_closed_route() -> None:
    validation = validate_tsp_route([0, 2, 1, 3, 0], num_locations=4)
    assert validation.is_valid
    assert validation.missing_nodes == []
    assert validation.repeated_nodes == []


def test_validate_tsp_route_reports_missing_and_repeated_nodes() -> None:
    validation = validate_tsp_route([0, 1, 1, 3, 0], num_locations=4)
    assert not validation.is_valid
    assert validation.missing_nodes == [2]
    assert validation.repeated_nodes == [1]


def test_route_distance_on_unit_square() -> None:
    locations = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    assert route_distance(locations, [0, 1, 2, 3, 0]) == 4.0


def test_percentage_gap() -> None:
    assert percentage_gap(11.0, 10.0) == 10.0
    assert percentage_gap(9.0, 10.0) == -10.0


def test_ortools_returns_valid_route_and_matches_exact_for_seed_42() -> None:
    locations = generate_locations(num_locations=10, seed=42)
    or_tools = solve_ortools_tsp(locations, time_limit_seconds=1)
    exact = solve_exact_tsp(locations)

    assert or_tools.validation.is_valid
    assert exact.validation.is_valid
    assert or_tools.distance >= exact.distance - 1e-9
    assert percentage_gap(or_tools.distance, exact.distance) < 0.01


def test_parse_paper_style_zero_shot_route() -> None:
    response = """<<start>>
Salesman1: Depot-8-7-4-2-5-6-3-1-9-Depot
<<end>>"""
    assert parse_single_salesman_route(response) == [0, 8, 7, 4, 2, 5, 6, 3, 1, 9, 0]


def test_parse_route_accepts_node_prefix() -> None:
    response = "<<start>>\nSalesman 1: Node0-Node2-Node1-Depot\n<<end>>"
    assert parse_single_salesman_route(response) == [0, 2, 1, 0]


def test_parse_route_rejects_unstructured_text() -> None:
    try:
        parse_single_salesman_route("I suggest visiting every node clockwise.")
    except RouteParseError as exc:
        assert "<<start>>" in str(exc)
    else:
        raise AssertionError("Biçimsiz cevap RouteParseError oluşturmalıydı.")


def test_image_mime_type() -> None:
    from pathlib import Path

    assert image_mime_type(Path("points.png")) == "image/png"
    assert image_mime_type(Path("points.jpg")) == "image/jpeg"


def test_initializer_and_critic_prompts_have_distinct_roles() -> None:
    assert "current route" not in initializer_prompt().lower()
    assert "current route" in critic_prompt().lower()
    assert "improve" in critic_prompt().lower()
