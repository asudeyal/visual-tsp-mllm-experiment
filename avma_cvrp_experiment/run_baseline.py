"""OR-Tools baseline solver for Capacitated Vehicle Routing Problem (CVRP)."""

import argparse
import json
from pathlib import Path

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from src.problem import euc_2d_distance, load_cvrplib


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
        help=(
            "Maximum number of vehicles. "
            "If omitted, the value supplied by the instance/configuration "
            "is used when available."
        ),
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

    # Vehicle count must be explicitly constrained for benchmark instances
    # such as E-n51-k5. load_cvrplib() stores this in problem.max_vehicles.
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

    def distance_callback(
        from_index: int,
        to_index: int,
    ) -> int:

        from_node = node_list[
            manager.IndexToNode(from_index)
        ]

        to_node = node_list[
            manager.IndexToNode(to_index)
        ]

        return euc_2d_distance(
            problem,
            from_node,
            to_node,
        )

    transit_callback_index = routing.RegisterTransitCallback(
        distance_callback
    )

    routing.SetArcCostEvaluatorOfAllVehicles(
        transit_callback_index
    )

    def demand_callback(from_index: int) -> int:

        from_node = node_list[
            manager.IndexToNode(from_index)
        ]

        return problem.demands.get(
            from_node,
            0,
        )

    demand_callback_index = routing.RegisterUnaryTransitCallback(
        demand_callback
    )

    routing.AddDimension(
        demand_callback_index,
        0,
        problem.capacity,
        True,
        "Capacity",
    )

    search_parameters = (
        pywrapcp.DefaultRoutingSearchParameters()
    )

    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )

    search_parameters.time_limit.FromSeconds(
        time_limit_seconds
    )

    print(
        f"Solving {problem.name} with OR-Tools "
        f"(Time limit: {time_limit_seconds}s, "
        f"Vehicles: {num_vehicles})..."
    )

    solution = routing.SolveWithParameters(
        search_parameters
    )

    if not solution:
        return {
            "status": "failed",
            "message": "No solution found within time limit.",
        }

    routes = []
    total_distance = 0

    for vehicle_id in range(num_vehicles):

        index = routing.Start(vehicle_id)

        route = []
        route_distance = 0

        while not routing.IsEnd(index):

            node = node_list[
                manager.IndexToNode(index)
            ]

            route.append(node)

            previous_index = index

            index = solution.Value(
                routing.NextVar(index)
            )

            route_distance += (
                routing.GetArcCostForVehicle(
                    previous_index,
                    index,
                    vehicle_id,
                )
            )

        node = node_list[
            manager.IndexToNode(index)
        ]

        route.append(node)

        # Ignore unused vehicles whose route is simply depot -> depot.
        if len(route) > 2:

            routes.append(route)
            total_distance += route_distance

    return {
        "status": "success",
        "instance": problem.name,
        "dimension": problem.dimension,
        "capacity": problem.capacity,
        "max_vehicles": num_vehicles,
        "vehicles_used": len(routes),
        "total_distance": total_distance,
        "routes": routes,
    }


def main() -> None:

    args = parse_args()

    problem = load_cvrplib(
        args.instance,
        max_vehicles=args.max_vehicles,
    )

    result = solve_cvrp(
        problem,
        args.time_limit,
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

        print(
            f"Results saved to {output_path}"
        )

    else:

        print(
            "Failed to find a solution."
        )


if __name__ == "__main__":
    main()