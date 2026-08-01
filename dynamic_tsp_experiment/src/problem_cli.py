"""Dinamik TSP problem kaynağı için ortak komut satırı seçenekleri."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.problem_instance import ProblemInstance
from src.problem_loader import (
    generate_random_problem,
    load_tsplib_problem,
)


def add_problem_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    """Rastgele ve TSPLIB girişlerini birbirini dışlayacak biçimde ekler."""

    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--num-nodes",
        type=int,
        help="Seed ile rastgele üretilecek toplam düğüm sayısı.",
    )
    source.add_argument(
        "--tsplib-file",
        "--instance",
        dest="tsplib_file",
        type=Path,
        help=(
            "NODE_COORD_SECTION içeren EUC_2D veya GEO TSPLIB "
            "problem dosyası."
        ),
    )
    parser.add_argument(
        "--optimal-tour-file",
        "--optimal-tour",
        dest="optimal_tour_file",
        type=Path,
        help="Varsa TSPLIB bilinen optimum TOUR_SECTION dosyası.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Yalnız rastgele problem üretiminde kullanılan seed.",
    )
    parser.add_argument(
        "--depot-id",
        type=int,
        help="Depo düğümü; rastgele problemde varsayılan 0'dır.",
    )


def _discover_optimal_tour(
    instance_file: Path,
) -> Path | None:
    candidate = instance_file.with_suffix(".opt.tour")
    return candidate if candidate.is_file() else None


def load_problem_from_args(
    args: argparse.Namespace,
    *,
    default_tsplib_file: Path | None = None,
    default_optimal_tour_file: Path | None = None,
) -> tuple[ProblemInstance, dict[str, Any]]:
    """Komut satırı seçimine göre problem ve tekrar üretim bilgisini döndürür."""

    num_nodes = getattr(args, "num_nodes", None)
    tsplib_file = getattr(args, "tsplib_file", None)
    optimal_tour_file = getattr(args, "optimal_tour_file", None)
    depot_id = getattr(args, "depot_id", None)
    seed = int(getattr(args, "seed", 42))

    if num_nodes is not None:
        if optimal_tour_file is not None:
            raise ValueError(
                "--optimal-tour-file yalnız --tsplib-file ile kullanılabilir."
            )
        selected_depot = 0 if depot_id is None else int(depot_id)
        problem = generate_random_problem(
            int(num_nodes),
            seed=seed,
            depot_id=selected_depot,
        )
        return problem, {
            "mode": "random",
            "num_nodes": problem.dimension,
            "seed": seed,
            "depot_id": selected_depot,
        }

    selected_instance = (
        Path(tsplib_file)
        if tsplib_file is not None
        else (
            Path(default_tsplib_file)
            if default_tsplib_file is not None
            else None
        )
    )
    if selected_instance is None:
        raise ValueError(
            "Problem kaynağı belirtilmedi. --num-nodes veya "
            "--tsplib-file kullanılmalıdır."
        )

    if optimal_tour_file is not None:
        selected_tour = Path(optimal_tour_file)
    elif (
        tsplib_file is None
        and default_optimal_tour_file is not None
    ):
        selected_tour = Path(default_optimal_tour_file)
    else:
        selected_tour = _discover_optimal_tour(selected_instance)

    problem = load_tsplib_problem(
        selected_instance,
        optimal_tour_file=selected_tour,
        depot_id=depot_id,
    )
    return problem, {
        "mode": "tsplib",
        "depot_id": problem.depot_id,
        "instance_file": str(selected_instance),
        "optimal_tour_file": (
            str(selected_tour)
            if selected_tour is not None
            else None
        ),
    }
