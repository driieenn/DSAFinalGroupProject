
"""
dijkstra_search.py

Single-objective optimizer for the DSA Final Group Project.

This module implements a Dijkstra-style shortest path algorithm that can
optimize one objective at a time, either time or cost, depending on which
adjacency matrix is passed into the function.
"""

from __future__ import annotations

import heapq
import numpy as np

from tensor_builder import (
    NODES,
    build_adjacency_matrices,
    get_neighbors,
)


def dijkstra(
    matrix: np.ndarray,
    source: str,
    target: str,
    node_index: dict[str, int],
    nodes: list[str] = NODES,
) -> tuple[list[str] | None, float]:
    """
    Finds the shortest path from source to target using Dijkstra's Algorithm.

    Args:
        matrix:
            The selected adjacency matrix. This can be the time matrix
            or the cost matrix.
        source:
            Starting node label.
        target:
            Destination node label.
        node_index:
            Dictionary mapping node labels to matrix indices.
        nodes:
            Ordered list of node labels.

    Returns:
        A tuple containing:
        - path: list of node labels, or None if no path exists
        - total_cost: total weight of the selected objective
    """
    if source not in node_index:
        raise ValueError(f"Unknown source node: {source}")

    if target not in node_index:
        raise ValueError(f"Unknown target node: {target}")

    if source == target:
        return [source], 0.0

    dist = {node: float("inf") for node in nodes}
    parent = {node: None for node in nodes}
    visited = set()

    dist[source] = 0.0
    priority_queue = [(0.0, source)]

    while priority_queue:
        current_cost, current_node = heapq.heappop(priority_queue)

        if current_node in visited:
            continue

        visited.add(current_node)

        if current_node == target:
            break

        for neighbor, weight in get_neighbors(matrix, current_node, node_index, nodes):
            if neighbor in visited:
                continue

            new_cost = current_cost + float(weight)

            if new_cost < dist[neighbor]:
                dist[neighbor] = new_cost
                parent[neighbor] = current_node
                heapq.heappush(priority_queue, (new_cost, neighbor))

    if np.isinf(dist[target]):
        return None, float("inf")

    path = []
    current = target

    while current is not None:
        path.append(current)
        current = parent[current]

    path.reverse()

    return path, dist[target]


def display_dijkstra_result(
    matrix: np.ndarray,
    source: str,
    target: str,
    node_index: dict[str, int],
    objective_name: str,
    nodes: list[str] = NODES,
) -> tuple[list[str] | None, float]:
    """
    Runs Dijkstra and prints a formatted result.
    """
    path, total = dijkstra(matrix, source, target, node_index, nodes)

    print(f"\n{objective_name.upper()} ROUTE")
    print("-" * 50)

    if path is None:
        print(f"No path found from {source} to {target}.")
    else:
        print(f"Path: {' -> '.join(path)}")
        print(f"Total {objective_name}: {total:.2f}")

    return path, total


if __name__ == "__main__":
    time_matrix, cost_matrix, node_index = build_adjacency_matrices()

    SOURCE = "A"
    TARGET = "J"

    print("Dijkstra Search Demo")
    print(f"Source: {SOURCE}")
    print(f"Target: {TARGET}")

    display_dijkstra_result(
        time_matrix,
        SOURCE,
        TARGET,
        node_index,
        "time",
    )

    display_dijkstra_result(
        cost_matrix,
        SOURCE,
        TARGET,
        node_index,
        "cost",
    )
