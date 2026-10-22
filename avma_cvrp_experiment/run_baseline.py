"""OR-Tools baseline solver for Capacitated Vehicle Routing Problem (CVRP)."""

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import ortools
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from src.problem import euc_2d_distance, load_cvrplib


FIRST_SOLUTION_STRATEGY = "PATH_CHEAPEST_ARC"
LOCAL_SEARCH_METAHEURISTIC = "GUIDED_LOCAL_SEARCH"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OR-Tools CVRP Baseline Solver"
    )

    parser.add_argument(
        "--instance",
        required=True,
        help="Path to CVRP .vrp file",
    )

    parser.add_argument(
        "--max-vehicles",
        type=int,
        default=None,
        help="Maximum number of vehicles.",
    )

    parser.add_argument(
        "--reference-optimum",
        type=float,
        default=None,
        help="Reference BKS/optimum used only for post-run gap calculation.",
    )

    parser.add_argument(
        "--time-limit",
        type=int,
        default=30,
        help="Search time limit in seconds",
    )

    parser.add_argument(
        "--output",
        default="baseline_results.json",
    )

    return parser.parse_args()


def solve_cvrp(
    problem,
    time_limit_seconds: int,
) -> dict[str, object]:

    num_vehicles = problem.max_vehicles

    if num_vehicles is None:
        raise ValueError(
            "CVRP baseline için max_vehicles belirtilmelidir. "
            "Komut satırında --max-vehicles N kullanın."
        )

    node_list = list(problem.node_ids)
    depot_index = node_list.index(problem.depot)

    manager = pywrapcp.RoutingIndexManager(
        len(node_list),
        num_vehicles,
        depot_index,
    )

    routing = pywrapcp.RoutingModel(manager)

    # ---------------------------------------------------------
    # Distance matrix
    # ---------------------------------------------------------
    distance_matrix = [
        [
            int(
                euc_2d_distance(
                    problem,
                    from_node,
                    to_node,
                )
            )
            for to_node in node_list
        ]
        for from_node in node_list
    ]

    transit_callback_index = routing.RegisterTransitMatrix(
        distance_matrix
    )

    routing.SetArcCostEvaluatorOfAllVehicles(
        transit_callback_index
    )

    # ---------------------------------------------------------
    # Demand vector
    # ---------------------------------------------------------
    demand_vector = [
        int(problem.demands.get(node, 0))
        for node in node_list
    ]

    demand_callback_index = routing.RegisterUnaryTransitVector(
        demand_vector
    )

    routing.AddDimension(
        demand_callback_index,
        0,
        int(problem.capacity),
        True,
        "Capacity",
    )

    # ---------------------------------------------------------
    # Search parameters
    # ---------------------------------------------------------
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()

    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )

    search_parameters.time_limit.FromSeconds(
        int(time_limit_seconds)
    )

    print(
        f"Solving {problem.name} with OR-Tools "
        f"(Time limit: {time_limit_seconds}s, "
        f"Vehicles: {num_vehicles})..."
    )

    start_time = time.perf_counter()

    solution = routing.SolveWithParameters(
        search_parameters
    )

    elapsed_seconds = time.perf_counter() - start_time

    base_metadata = {
        "solver": "Google OR-Tools Routing Solver",
        "ortools_version": ortools.__version__,
        "python_version": platform.python_version(),
        "first_solution_strategy": FIRST_SOLUTION_STRATEGY,
        "local_search_metaheuristic": LOCAL_SEARCH_METAHEURISTIC,
        "time_limit_seconds": int(time_limit_seconds),
        "elapsed_seconds": round(elapsed_seconds, 6),
        "run_timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    if not solution:
        return {
            "status": "failed",
            "instance": problem.name,
            "dimension": problem.dimension,
            "capacity": problem.capacity,
            "max_vehicles": num_vehicles,
            "reference_optimum": problem.reference_optimum,
            **base_metadata,
            "message": "No solution found within time limit.",
        }

    routes = []
    total_distance = 0

    # ---------------------------------------------------------
    # Extract routes
    # ---------------------------------------------------------
    for vehicle_id in range(num_vehicles):

        index = routing.Start(vehicle_id)

        route = []
        route_distance = 0

        while not routing.IsEnd(index):

            node_index = manager.IndexToNode(index)
            node = node_list[node_index]

            route.append(node)

            previous_index = index

            index = solution.Value(
                routing.NextVar(index)
            )

            route_distance += int(
                routing.GetArcCostForVehicle(
                    previous_index,
                    index,
                    vehicle_id,
                )
            )

        node_index = manager.IndexToNode(index)
        node = node_list[node_index]

        route.append(node)

        # Ignore unused vehicles whose route is simply depot -> depot.
        if len(route) > 2:
            routes.append(route)
            total_distance += route_distance

    reference_optimum = problem.reference_optimum

    if reference_optimum is not None:
        gap_percent = (
            (float(total_distance) - float(reference_optimum))
            / float(reference_optimum)
            * 100.0
        )
    else:
        gap_percent = None

    return {
        "status": "success",
        "instance": problem.name,
        "dimension": problem.dimension,
        "capacity": problem.capacity,
        "max_vehicles": num_vehicles,
        "vehicles_used": len(routes),
        "total_distance": int(total_distance),
        "reference_optimum": reference_optimum,
        "gap_percent": (
            round(gap_percent, 6)
            if gap_percent is not None
            else None
        ),
        **base_metadata,
        "routes": routes,
    }


def main() -> None:

    args = parse_args()

    problem = load_cvrplib(
        args.instance,
        max_vehicles=args.max_vehicles,
        reference_optimum=args.reference_optimum,
    )

    result = solve_cvrp(
        problem,
        args.time_limit,
    )

    output_path = Path(args.output)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    if result["status"] == "success":

        print(
            f"Success! Total Distance: "
            f"{result['total_distance']}"
        )

        print(
            f"Vehicles Used: "
            f"{result['vehicles_used']}"
        )

        print(
            f"Vehicle Limit: "
            f"{result['max_vehicles']}"
        )

        if result["reference_optimum"] is not None:
            print(
                f"Reference BKS: "
                f"{result['reference_optimum']}"
            )

            print(
                f"Gap to BKS: "
                f"{result['gap_percent']:.4f}%"
            )

        print(
            f"Elapsed Time: "
            f"{result['elapsed_seconds']:.3f}s"
        )

        print(
            f"Results saved to {output_path}"
        )

    else:

        print(
            f"Failed to find a solution for "
            f"{result.get('instance', 'unknown instance')}."
        )

        print(
            f"Results saved to {output_path}"
        )


if __name__ == "__main__":
    main()