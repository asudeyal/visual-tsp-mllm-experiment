from src.gemini import (
    critic_prompt,
    initializer_prompt,
    parse_route,
    parse_scorer_response,
    scorer_prompt,
)
from src.problem_instance import (
    ProblemInstance,
    ProblemSource,
)
from src.problem_loader import generate_random_problem


def test_route_parser_accepts_depot_one_format() -> None:
    text = (
        "<<start>>\n"
        "Salesman1: Depot-2-3-4-Depot\n"
        "<<end>>"
    )
    assert parse_route(
        text,
        depot_id=1,
    ) == [1, 2, 3, 4, 1]


def test_route_parser_uses_dynamic_depot_zero() -> None:
    text = (
        "<<start>>\n"
        "Salesman1: Depot-2-1-3-Depot\n"
        "<<end>>"
    )
    assert parse_route(
        text,
        depot_id=0,
    ) == [0, 2, 1, 3, 0]


def test_initializer_prompt_uses_problem_metadata() -> None:
    problem = generate_random_problem(10, seed=42)

    prompt = initializer_prompt(problem)

    assert "random_n10_seed42" in prompt
    assert "exactly 10 nodes" in prompt
    assert "labelled 0 through 9" in prompt
    assert "Node 0" in prompt
    assert "Depot means node 0" in prompt


def test_prompts_support_nonconsecutive_node_ids() -> None:
    problem = ProblemInstance(
        name="nonconsecutive",
        source_type=ProblemSource.TSPLIB,
        dimension=4,
        depot_id=10,
        edge_weight_type="EUC_2D",
        coordinates={
            10: (0.0, 0.0),
            20: (1.0, 0.0),
            30: (1.0, 1.0),
            50: (0.0, 1.0),
        },
    )

    initializer = initializer_prompt(problem)
    critic = critic_prompt(problem)
    scorer = scorer_prompt(problem, [1, 2])

    assert "10, 20, 30, 50" in initializer
    assert "20, 30, 50" in critic
    assert "depot node 10" in scorer


def test_scorer_parser_reads_all_ids_and_best() -> None:
    text = (
        "<<image1: 10, image2: 7, image3: 9>>\n"
        "<<the best route: 1>>"
    )
    scores, best = parse_scorer_response(
        text,
        expected_image_ids=[1, 2, 3],
    )
    assert scores == {
        1: 10.0,
        2: 7.0,
        3: 9.0,
    }
    assert best == 1


def test_scorer_parser_uses_highest_score_when_best_line_missing() -> None:
    text = "<<image1: 8, image2: 9, image3: 7>>"

    scores, best = parse_scorer_response(
        text,
        expected_image_ids=[1, 2, 3],
    )

    assert scores[2] == 9.0
    assert best == 2
