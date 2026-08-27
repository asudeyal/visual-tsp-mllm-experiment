from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation import evaluate_route
from src.problem import euc_2d_distance, load_tsplib


PROJECT_ROOT = Path(__file__).resolve().parent


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OR-Tools baseline for EUC_2D TSP")
    parser.add_argument("--instance", required=True)
    parser.add_argument("--reference-optimum", type=float, default=None)
    parser.add_argument("--time-limit", type=int, default=2)
    parser.add_argument("--output", default="output/baseline_result.json")
    return parser.parse_args()


def solve(problem, time_limit: int) -> tuple[int, ...]:
    try:
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    except ImportError as exc:
        raise RuntimeError("OR-Tools kurulu değil. requirements.txt yükleyin.") from exc

    nodes = list(problem.node_ids)
    depot_index = nodes.index(problem.depot)
    manager = pywrapcp.RoutingIndexManager(len(nodes), 1, depot_index)
    routing = pywrapcp.RoutingModel(manager)

    def callback(from_index: int, to_index: int) -> int:
        a = nodes[manager.IndexToNode(from_index)]
        b = nodes[manager.IndexToNode(to_index)]
        return euc_2d_distance(problem, a, b)

    transit = routing.RegisterTransitCallback(callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit)
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.seconds = time_limit
    solution = routing.SolveWithParameters(params)
    if solution is None:
        raise RuntimeError("OR-Tools çözüm üretemedi")

    route = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        route.append(nodes[manager.IndexToNode(index)])
        index = solution.Value(routing.NextVar(index))
    route.append(problem.depot)
    return tuple(route)


def main() -> None:
    args = parse_args()
    problem = load_tsplib(resolve(args.instance), reference_optimum=args.reference_optimum, strict_euc_2d=True)
    route = solve(problem, args.time_limit)
    evaluation = evaluate_route(problem, route)
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"method": "ortools", "route": list(route), "evaluation": evaluation.to_dict()}, indent=2),
        encoding="utf-8",
    )
    print(f"Baseline distance: {evaluation.distance}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
