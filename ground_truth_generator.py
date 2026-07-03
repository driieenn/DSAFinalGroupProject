"""
ground_truth_generator.py
independent brute-force checker. no imports or calls for pareto search code. requires tensor_builder.py


to call:
from ground_truth_generator import generate_ground_truth

gt = generate_ground_truth("A", "J") # any valid source/destination node pair
gt["all_candidates"]   # every simple path found, with time/cost
gt["best_by_time"]     # expected Task 3 answer (minimize time)
gt["best_by_cost"]     # expected Task 3 answer (minimize cost)
gt["pareto_front"]     # expected Task 4 answer (non-dominated routes)
"""

from tensor_builder import NODES, EDGES


def _build_adjacency_list(edges: list[tuple]) -> dict[str, list[tuple[str, float, float]]]:
    adjacency: dict[str, list[tuple[str, float, float]]] = {n: [] for n in NODES}
    for source, destination, time_value, cost_value in edges:
        adjacency[source].append((destination, float(time_value), float(cost_value)))
    return adjacency


_ADJACENCY = _build_adjacency_list(EDGES)


# section 1 - path enumeration
def enumerate_all_simple_paths(source: str, destination: str) -> list[dict]:
    """Finds every simple path from source to destination via backtracking."""
    if source not in _ADJACENCY:
        raise ValueError(f"Unknown source node: {source}")
    if destination not in _ADJACENCY:
        raise ValueError(f"Unknown destination node: {destination}")

    found_routes: list[dict] = []
    visited = {source}
    current_path = [source]

    def backtrack(node: str, running_time: float, running_cost: float) -> None:
        if node == destination:
            found_routes.append({
                "path": list(current_path),
                "time": running_time,
                "cost": running_cost,
            })
            return

        for neighbor, edge_time, edge_cost in _ADJACENCY[node]:
            if neighbor in visited:
                continue
            visited.add(neighbor)
            current_path.append(neighbor)
            backtrack(neighbor, running_time + edge_time, running_cost + edge_cost)
            current_path.pop()
            visited.remove(neighbor)

    backtrack(source, 0.0, 0.0)
    return found_routes


# section 2 - single-objective ground truth
def brute_force_best_by_objective(routes: list[dict], objective: str) -> dict:
    """Returns the route with the minimum time or cost."""
    if objective not in ("time", "cost"):
        raise ValueError("objective must be 'time' or 'cost'")
    if not routes:
        return {}

    best_route = routes[0]
    for route in routes[1:]:
        if route[objective] < best_route[objective]:
            best_route = route
    return best_route


# section 3 - pareto front (dominance filtering)
def _route_is_dominated_by(candidate: dict, other: dict) -> bool:
    """True if `other` is no worse in all objectives and better in at least one."""
    no_worse_in_all = other["time"] <= candidate["time"] and other["cost"] <= candidate["cost"]
    better_in_at_least_one = other["time"] < candidate["time"] or other["cost"] < candidate["cost"]
    return no_worse_in_all and better_in_at_least_one


def brute_force_pareto_front(routes: list[dict]) -> list[dict]:
    """Returns non-dominated routes, sorted by (time, cost)."""
    front = []
    for i, candidate in enumerate(routes):
        dominated = any(
            i != j and _route_is_dominated_by(candidate, other)
            for j, other in enumerate(routes)
        )
        if not dominated:
            front.append(candidate)

    front.sort(key=lambda r: (r["time"], r["cost"]))
    return front


# section 4 - main entry point
def generate_ground_truth(source: str, destination: str) -> dict:
    """Main entry point: returns all candidates, best-by-time, best-by-cost, and the Pareto front."""
    all_candidates = enumerate_all_simple_paths(source, destination)

    return {
        "source": source,
        "destination": destination,
        "all_candidates": all_candidates,
        "best_by_time": brute_force_best_by_objective(all_candidates, "time"),
        "best_by_cost": brute_force_best_by_objective(all_candidates, "cost"),
        "pareto_front": brute_force_pareto_front(all_candidates),
    }


# section 5 - comparison helpers
def paths_match(path_a: list[str], path_b: list[str]) -> bool:
    return list(path_a) == list(path_b)


def route_sets_match(routes_a: list[dict], routes_b: list[dict], tolerance: float = 1e-9) -> bool:
    """Compares two route lists as unordered sets, order-independent."""
    def as_key(route: dict):
        return (
            tuple(route["path"]),
            round(route["time"] / tolerance) * tolerance,
            round(route["cost"] / tolerance) * tolerance,
        )

    return {as_key(r) for r in routes_a} == {as_key(r) for r in routes_b}


# section 6 - standalone demo (tests a to j)
if __name__ == "__main__":
    SOURCE, DESTINATION = "A", "J"
    ground_truth = generate_ground_truth(SOURCE, DESTINATION)

    print("Ground Truth Generator")
    print(f"Route: {SOURCE} -> {DESTINATION}")
    print(f"Total simple paths found: {len(ground_truth['all_candidates'])}")

    best_time = ground_truth["best_by_time"]
    print(f"\nBest by time: {' -> '.join(best_time['path'])} | time={best_time['time']:.2f}h | cost=${best_time['cost']:.2f}")

    best_cost = ground_truth["best_by_cost"]
    print(f"Best by cost: {' -> '.join(best_cost['path'])} | time={best_cost['time']:.2f}h | cost=${best_cost['cost']:.2f}")

    print(f"\nPareto front ({len(ground_truth['pareto_front'])} routes):")
    for r in ground_truth["pareto_front"]:
        print(f"  {' -> '.join(r['path'])} | time={r['time']:.2f}h | cost=${r['cost']:.2f}")