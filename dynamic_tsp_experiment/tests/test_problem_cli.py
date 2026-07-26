import argparse
from pathlib import Path

import pytest

from src.problem_cli import (
    add_problem_arguments,
    load_problem_from_args,
)
from src.problem_instance import ProblemSource


ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_problem_arguments(parser)
    return parser


def test_random_cli_problem() -> None:
    args = _parser().parse_args(
        [
            "--num-nodes",
            "20",
            "--seed",
            "43",
        ]
    )

    problem, request = load_problem_from_args(args)

    assert problem.dimension == 20
    assert problem.seed == 43
    assert problem.depot_id == 0
    assert problem.source_type is ProblemSource.RANDOM
    assert request["mode"] == "random"


def test_tsplib_cli_problem() -> None:
    args = _parser().parse_args(
        [
            "--tsplib-file",
            str(ROOT / "data" / "eil51.tsp"),
            "--optimal-tour-file",
            str(ROOT / "data" / "eil51.opt.tour"),
        ]
    )

    problem, request = load_problem_from_args(args)

    assert problem.name == "eil51"
    assert problem.dimension == 51
    assert problem.reference is not None
    assert problem.reference.distance == 426
    assert request["mode"] == "tsplib"


def test_source_arguments_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        _parser().parse_args(
            [
                "--num-nodes",
                "10",
                "--tsplib-file",
                "example.tsp",
            ]
        )


def test_optimal_tour_is_rejected_for_random_problem() -> None:
    args = _parser().parse_args(
        [
            "--num-nodes",
            "10",
            "--optimal-tour-file",
            "example.opt.tour",
        ]
    )

    with pytest.raises(
        ValueError,
        match="yalnız --tsplib-file",
    ):
        load_problem_from_args(args)
