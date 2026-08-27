from src.problem import load_tsplib, route_length


def test_load_euc_2d_tsplib(tmp_path):
    path = tmp_path / "tiny.tsp"
    path.write_text(
        """NAME: tiny\nTYPE: TSP\nDIMENSION: 4\nEDGE_WEIGHT_TYPE: EUC_2D\nNODE_COORD_SECTION\n1 0 0\n2 3 0\n3 3 4\n4 0 4\nEOF\n""",
        encoding="utf-8",
    )
    problem = load_tsplib(path)
    assert problem.depot == 1
    assert problem.dimension == 4
    assert route_length(problem, (1, 2, 3, 4, 1)) == 14.0
