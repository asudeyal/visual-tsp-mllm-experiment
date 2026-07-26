from src.gemini import parse_route, parse_scorer_response


def test_route_parser_accepts_depot_format() -> None:
    text = "<<start>>\nSalesman1: Depot-2-3-4-Depot\n<<end>>"
    assert parse_route(text) == [1, 2, 3, 4, 1]


def test_scorer_parser_reads_all_ids_and_best() -> None:
    text = "<<image1: 10, image2: 7, image3: 9>>\n<<the best route: 1>>"
    scores, best = parse_scorer_response(text, expected_image_ids=[1, 2, 3])
    assert scores == {1: 10.0, 2: 7.0, 3: 9.0}
    assert best == 1
