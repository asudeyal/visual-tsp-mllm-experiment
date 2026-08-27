from src.config import RenderConfig
from src.rendering import render_problem, render_route
from src.schemas import ProblemInstance


def test_model_renders_are_created(tmp_path, square_problem):
    cfg = RenderConfig()
    problem_image = render_problem(square_problem, tmp_path / "problem.png", cfg)
    route_image = render_route(square_problem, (1, 2, 3, 4, 1), tmp_path / "route.png", cfg)
    assert problem_image.exists() and problem_image.stat().st_size > 0
    assert route_image.exists() and route_image.stat().st_size > 0


def test_dense_problem_render_does_not_fail(tmp_path):
    problem = ProblemInstance(
        name="dense_cluster",
        dimension=6,
        node_ids=(1, 2, 3, 4, 5, 6),
        coordinates={
            1: (37.0, 52.0),
            2: (32.0, 39.0),
            3: (30.0, 40.0),
            4: (38.0, 46.0),
            5: (42.0, 41.0),
            6: (31.0, 32.0),
        },
        depot=1,
        edge_weight_type="EUC_2D",
        source_path=None,
        source_sha256="test",
        reference_optimum=None,
    )

    cfg = RenderConfig()
    problem_image = render_problem(problem, tmp_path / "dense_problem.png", cfg)
    route_image = render_route(problem, (1, 2, 3, 4, 5, 6, 1), tmp_path / "dense_route.png", cfg)

    assert problem_image.exists() and problem_image.stat().st_size > 0
    assert route_image.exists() and route_image.stat().st_size > 0
