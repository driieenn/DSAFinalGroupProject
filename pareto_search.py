"""
pareto_search.py
primary multi-objective Pareto-front search. requires tensor_builder.py

to call:
from pareto_search import pareto_search, time_matrix, cost_matrix, node_index

result = pareto_search(time_matrix, cost_matrix, "A", "J", node_index)  # any valid source/destination node pair
result["candidates"]       # every simple path found, with time/cost
result["pareto_front"]     # Task 4 answer (non-dominated routes)
result["fastest_route"]    # fastest route on the Pareto front
result["cheapest_route"]   # cheapest route on the Pareto front

time_matrix, cost_matrix, node_index are already built from tensor_builder.NODES / EDGES at import time
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import heapq
from itertools import combinations
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from tensor_builder import NODES, EDGES, INF, build_adjacency_matrices

# section 1 - graph setup
time_matrix, cost_matrix, node_index = build_adjacency_matrices(NODES, EDGES)


# section 2 - route data structure
class Route:
    """
    Represents a single candidate route as a [time, cost] vector.

    This is the core data structure for multi-objective Pareto search.
    Each route stores:
      - path       : ordered list of node labels from source to destination
      - objectives : numpy array [total_time, total_cost] -- the cost vector
    """

    def __init__(self, path: list[str], total_time: float, total_cost: float):
        self.path = path
        self.objectives = np.array([total_time, total_cost], dtype=float)

    @property
    def time(self) -> float:
        return self.objectives[0]

    @property
    def cost(self) -> float:
        return self.objectives[1]

    @property
    def path_str(self) -> str:
        return " -> ".join(self.path)

    def __repr__(self) -> str:
        return (
            f"Route({self.path_str} | "
            f"time={self.time:.2f}h, cost=${self.cost:.2f})"
        )

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "time": self.time,
            "cost": self.cost,
            "objectives": self.objectives.tolist(),
        }


@dataclass(frozen=True)
class SearchState:
    """Internal search label used while pruning dominated partial routes."""

    node_idx: int
    path: tuple[str, ...]
    time: float
    cost: float
    visited: frozenset[int]

    def to_route(self) -> Route:
        return Route(list(self.path), self.time, self.cost)


# section 3 - candidate route generation
def generate_candidate_routes(
    time_matrix: np.ndarray,
    cost_matrix: np.ndarray,
    source: str,
    destination: str,
    node_index: dict[str, int],
    nodes: list[str] = NODES,
) -> list[Route]:
    """
    Generates candidate routes using dominance-pruned label-setting search.

    The traversal keeps a non-dominated label set for each node. A new partial
    route is discarded as soon as another partial route to the same node is no
    worse in both objectives and strictly better in at least one, with a
    visited-set subset check to keep the pruning safe for simple-path search.

    Returns:
        List of Route objects discovered for the destination node.
    """
    if source not in node_index:
        raise ValueError(f"Source node '{source}' not found.")
    if destination not in node_index:
        raise ValueError(f"Destination node '{destination}' not found.")
    if source == destination:
        raise ValueError("Source and destination must be different nodes.")

    index_to_node = {v: k for k, v in node_index.items()}
    src_idx = node_index[source]
    dst_idx = node_index[destination]
    n = len(nodes)

    def dominates_state(label_a: SearchState, label_b: SearchState) -> bool:
        """True if label_a safely dominates label_b for the same node."""
        return (
            label_a.node_idx == label_b.node_idx
            and label_a.visited.issubset(label_b.visited)
            and label_a.time <= label_b.time
            and label_a.cost <= label_b.cost
            and (label_a.time < label_b.time or label_a.cost < label_b.cost)
        )

    def is_dominated_by_existing(
        candidate: SearchState,
        labels: list[SearchState],
    ) -> bool:
        return any(dominates_state(existing, candidate) for existing in labels)

    labels_by_node: dict[int, list[SearchState]] = {i: [] for i in range(n)}
    completed_routes: list[Route] = []

    initial_state = SearchState(
        node_idx=src_idx,
        path=(source,),
        time=0.0,
        cost=0.0,
        visited=frozenset({src_idx}),
    )
    labels_by_node[src_idx].append(initial_state)

    stack = [initial_state]

    while stack:
        state = stack.pop()

        # Skip labels that were already removed by a later dominating label.
        if state not in labels_by_node[state.node_idx]:
            continue

        if state.node_idx == dst_idx:
            completed_routes.append(state.to_route())
            continue

        for neighbor_idx in range(n):
            t = time_matrix[state.node_idx][neighbor_idx]
            c = cost_matrix[state.node_idx][neighbor_idx]

            # if no edge, self-loop, or already visited
            if t == INF or t == 0:
                continue
            if neighbor_idx in state.visited:
                continue

            next_state = SearchState(
                node_idx=neighbor_idx,
                path=state.path + (index_to_node[neighbor_idx],),
                time=state.time + float(t),
                cost=state.cost + float(c),
                visited=state.visited | frozenset({neighbor_idx}),
            )

            labels = labels_by_node[neighbor_idx]
            if is_dominated_by_existing(next_state, labels):
                continue

            labels_by_node[neighbor_idx] = [
                label for label in labels if not dominates_state(next_state, label)
            ]
            labels_by_node[neighbor_idx].append(next_state)
            stack.append(next_state)

    return completed_routes


# section 4 - pareto dominance
def dominates(route_a: Route, route_b: Route) -> bool:
    """
    Returns True if route_a dominates route_b under Pareto dominance.

    Definition:
        route_a dominates route_b if and only if:
          (1) route_a is no worse than route_b in ALL objectives
          (2) route_a is strictly better than route_b in AT LEAST ONE objective

    For this problem, lower is better for both time and cost.

    This uses numpy vector comparison for clean generalization
    across any number of objectives.
    """
    no_worse_in_all = np.all(route_a.objectives <= route_b.objectives)
    better_in_at_least_one = np.any(route_a.objectives < route_b.objectives)
    return bool(no_worse_in_all and better_in_at_least_one)


def dominance_relation(route_a: Route, route_b: Route) -> str:
    """
    Returns a human-readable dominance relationship between two routes.
    Useful for debugging and explanation.
    """
    if dominates(route_a, route_b):
        return "A dominates B"
    elif dominates(route_b, route_a):
        return "B dominates A"
    elif np.array_equal(route_a.objectives, route_b.objectives):
        return "A and B are identical in objectives"
    else:
        return "A and B are non-dominated (incomparable)"


def pareto_filter(routes: list[Route]) -> list[Route]:
    """
    Filters a list of Route objects and returns only the Pareto-optimal front.

    Algorithm:
      For each candidate route, check if any OTHER route dominates it.
      If no other route dominates it, it survives into the Pareto front.

    Time complexity: O(n^2 * m) where n = number of routes, m = number of objectives.
    For small graphs with few objectives this is entirely acceptable.

    Returns:
        Sorted list of non-dominated Route objects (sorted by time ascending).
    """
    if not routes:
        return []

    pareto_front = []

    for i, candidate in enumerate(routes):
        is_dominated = False
        for j, other in enumerate(routes):
            if i == j:
                continue
            if dominates(other, candidate):
                is_dominated = True
                break
        if not is_dominated:
            pareto_front.append(candidate)

    # sort by ascending time, then by ascending cost
    pareto_front.sort(key=lambda r: (r.time, r.cost))
    return pareto_front


def identify_dominated_routes(routes: list[Route]) -> list[tuple[Route, Route]]:
    """
    Returns a list of (dominated_route, dominating_route) pairs.
    Useful for explaining WHY certain routes were eliminated.
    """
    dominated_pairs = []
    for i, candidate in enumerate(routes):
        for j, other in enumerate(routes):
            if i == j:
                continue
            if dominates(other, candidate):
                dominated_pairs.append((candidate, other))
                break
    return dominated_pairs


# section 5 - full pipeline
def pareto_search(
    time_matrix: np.ndarray,
    cost_matrix: np.ndarray,
    source: str,
    destination: str,
    node_index: dict[str, int],
    nodes: list[str] = NODES,
) -> dict:
    """
    Full multi-objective Pareto-front search.

    Pipeline:
      Step 1 - Graph traversal (DFS) to generate all simple candidate routes
      Step 2 - Represent each route as a [time, cost] objective vector
      Step 3 - Apply Pareto dominance filtering
      Step 4 - Return the non-dominated set as the final algorithm output

    This is the PRIMARY multi-objective answer -- not a validation step.

    Returns:
        dict with keys:
          source           : source node label
          destination      : destination node label
          candidates       : all Route objects generated
          pareto_front     : non-dominated Route objects (the answer)
          dominated_pairs  : list of (eliminated_route, eliminator) pairs
          dominated_count  : number of routes eliminated
          fastest_route    : Route with lowest time on the Pareto front
          cheapest_route   : Route with lowest cost on the Pareto front
    """
    candidates = generate_candidate_routes(
        time_matrix, cost_matrix, source, destination, node_index, nodes
    )

    if not candidates:
        return {
            "source": source,
            "destination": destination,
            "candidates": [],
            "pareto_front": [],
            "dominated_pairs": [],
            "dominated_count": 0,
            "fastest_route": None,
            "cheapest_route": None,
        }

    front = pareto_filter(candidates)
    dominated_pairs = identify_dominated_routes(candidates)

    fastest = min(front, key=lambda r: r.time)
    cheapest = min(front, key=lambda r: r.cost)

    return {
        "source": source,
        "destination": destination,
        "candidates": candidates,
        "pareto_front": front,
        "dominated_pairs": dominated_pairs,
        "dominated_count": len(dominated_pairs),
        "fastest_route": fastest,
        "cheapest_route": cheapest,
    }


# section 6 - reporting and display
def display_pareto_results(result: dict, show_dominated: bool = True) -> None:
    """
    Prints a full formatted report of the Pareto search result.

    Parameters:
        result          : output dict from pareto_search()
        show_dominated  : if True, also prints eliminated routes with reasons
    """
    src = result["source"]
    dst = result["destination"]
    front = result["pareto_front"]
    total = len(result["candidates"])

    print(f"\n{'='*60}")
    print(f"  PARETO-FRONT SEARCH  |  {src}  ->  {dst}")
    print(f"{'='*60}")
    print(f"  Candidates generated : {total}")
    print(f"  Dominated removed    : {result['dominated_count']}")
    print(f"  Pareto-optimal found : {len(front)}")
    print(f"{'-'*60}")

    if not front:
        print("  No routes found between these nodes.")
        print(f"{'='*60}")
        return

    # pareto front table
    print(f"\n  PARETO-OPTIMAL ROUTES (the multi-objective answer):\n")
    print(f"  {'#':<4} {'Path':<35} {'Time (h)':>9} {'Cost ($)':>10}  Note")
    print(f"  {'-'*4} {'-'*35} {'-'*9} {'-'*10}  {'-'*15}")

    fastest_id = id(result["fastest_route"])
    cheapest_id = id(result["cheapest_route"])

    for i, r in enumerate(front, 1):
        note = ""
        if id(r) == fastest_id and id(r) == cheapest_id:
            note = "< fastest & cheapest"
        elif id(r) == fastest_id:
            note = "< fastest"
        elif id(r) == cheapest_id:
            note = "< cheapest"
        print(f"  {i:<4} {r.path_str:<35} {r.time:>9.2f} {r.cost:>10.2f}  {note}")

    # vector summary
    print(f"\n  Objective vectors [time, cost]:")
    for r in front:
        print(f"    {r.path_str:<35} -> {r.objectives}")

    # tradeoffs
    if len(front) > 1:
        print(f"\n  Trade-off analysis:")
        fastest = result["fastest_route"]
        cheapest = result["cheapest_route"]
        time_diff = cheapest.time - fastest.time
        cost_diff = fastest.cost - cheapest.cost
        print(f"    Taking the fastest route costs ${cost_diff:.2f} more than the cheapest.")
        print(f"    Taking the cheapest route takes {time_diff:.2f}h longer than the fastest.")

    # eliminated routes
    if show_dominated and result["dominated_pairs"]:
        print(f"\n  ELIMINATED ROUTES (dominated):\n")
        for dominated, dominator in result["dominated_pairs"]:
            print(f"    X {dominated.path_str}")
            print(f"        [time={dominated.time:.2f}h, cost=${dominated.cost:.2f}]")
            print(f"      dominated by -> {dominator.path_str}")
            print(f"        [time={dominator.time:.2f}h, cost=${dominator.cost:.2f}]")
            print()


def plot_objective_space(result: dict) -> None:
    """
    Plots all candidate routes in [time, cost] objective space.

    - Grey points  : dominated routes (eliminated)
    - Red points   : Pareto-optimal routes (the answer)
    - Red line     : the Pareto front boundary
    - Annotations  : abbreviated path labels on Pareto-optimal points
    """
    candidates = result["candidates"]
    front = result["pareto_front"]
    src = result["source"]
    dst = result["destination"]

    dominated_routes = {
        id(pair[0]) for pair in result["dominated_pairs"]
    }

    dom_times = [r.time for r in candidates if id(r) in dominated_routes]
    dom_costs = [r.cost for r in candidates if id(r) in dominated_routes]
    front_times = [r.time for r in front]
    front_costs = [r.cost for r in front]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Dominated routes
    ax.scatter(dom_times, dom_costs, color="lightgrey", s=80,
               zorder=2, label=f"Dominated routes ({len(dom_times)})")

    # Pareto front points
    ax.scatter(front_times, front_costs, color="crimson", s=120,
               zorder=4, label=f"Pareto-optimal routes ({len(front)})")

    # Pareto front boundary line (step function -- staircase shape)
    sorted_front = sorted(front, key=lambda r: r.time)
    step_times = [r.time for r in sorted_front]
    step_costs = [r.cost for r in sorted_front]
    ax.step(step_times, step_costs, where="post", color="crimson",
            linewidth=1.5, linestyle="--", zorder=3, alpha=0.7)

    # Annotate Pareto-optimal routes
    for r in front:
        label = r.path_str if len(r.path) <= 4 else (
            r.path[0] + "->...->" + r.path[-1]
        )
        ax.annotate(
            label,
            xy=(r.time, r.cost),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=7.5,
            color="darkred",
        )

    ax.set_xlabel("Total Time (hours)", fontsize=12)
    ax.set_ylabel("Total Cost (USD $)", fontsize=12)
    ax.set_title(
        f"Objective Space: {src} -> {dst}\n"
        f"Pareto Front ({len(front)} non-dominated routes out of {len(candidates)} candidates)",
        fontsize=13,
    )
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


# section 7 - standalone demo
if __name__ == "__main__":
    # Demo Route object
    demo_route = Route(["A", "B", "D", "G", "J"], 4.0 + 2.0 + 1.5 + 1.2, 150 + 70 + 40 + 35)
    print("Demo Route:")
    print(f"  Path       : {demo_route.path_str}")
    print(f"  Objectives : {demo_route.objectives}")
    print(f"  Time       : {demo_route.time:.2f}h")
    print(f"  Cost       : ${demo_route.cost:.2f}")

    # Candidate route generation demo
    candidates = generate_candidate_routes(time_matrix, cost_matrix, "A", "J", node_index)
    print(f"\nTotal candidate routes found (A -> J): {len(candidates)}\n")
    print(f"  {'#':<4} {'Path':<35} {'Time (h)':>9} {'Cost ($)':>10}")
    print(f"  {'-'*4} {'-'*35} {'-'*9} {'-'*10}")
    for i, r in enumerate(candidates, 1):
        print(f"  {i:<4} {r.path_str:<35} {r.time:>9.2f} {r.cost:>10.2f}")

    # Dominance relation demo
    print("\ndominance relationship demo")
    test_pairs = [
        (Route(["A", "B", "J"], 5.0, 100), Route(["A", "C", "J"], 6.0, 120), "faster AND cheaper"),
        (Route(["A", "B", "J"], 5.0, 100), Route(["A", "C", "J"], 4.0, 120), "faster but more expensive"),
        (Route(["A", "B", "J"], 5.0, 100), Route(["A", "C", "J"], 5.0, 100), "identical objectives"),
        (Route(["A", "B", "J"], 5.0, 130), Route(["A", "C", "J"], 6.0, 100), "trade-off: neither dominates"),
    ]
    for a, b, desc in test_pairs:
        rel = dominance_relation(a, b)
        print(f"  Scenario : {desc}")
        print(f"  Route A  : time={a.time:.1f}h  cost=${a.cost:.0f}")
        print(f"  Route B  : time={b.time:.1f}h  cost=${b.cost:.0f}")
        print(f"  Result   : {rel}\n")

    # Pareto filtering demo
    pareto_front = pareto_filter(candidates)
    dominated_pairs = identify_dominated_routes(candidates)

    print(f"Pareto Filtering Results (A -> J)")
    print("=" * 55)
    print(f"  Candidates total  : {len(candidates)}")
    print(f"  Dominated removed : {len(dominated_pairs)}")
    print(f"  Pareto front size : {len(pareto_front)}\n")

    print("Eliminated Routes (with reason):")
    print(f"  {'-'*52}")
    for dominated, dominator in dominated_pairs:
        print(f"  DOMINATED : {dominated.path_str}")
        print(f"             [time={dominated.time:.2f}h, cost=${dominated.cost:.2f}]")
        print(f"  BY        : {dominator.path_str}")
        print(f"             [time={dominator.time:.2f}h, cost=${dominator.cost:.2f}]")
        print()

    # Full pipeline demo
    result_AJ = pareto_search(time_matrix, cost_matrix, "A", "J", node_index)
    display_pareto_results(result_AJ, show_dominated=True)

    # Plot demo
    plot_objective_space(result_AJ)