"""
tensor_builder.py

Task 2: Data Structure Implementation - Adjacency Matrices & Tensor

This module builds:
1. a time adjacency matrix
2. a cost adjacency matrix
3. a stacked tensor containing both matrices

The graph is directed, so an edge from A to B does not automatically mean
there is also an edge from B to A.
"""

from __future__ import annotations

import numpy as np


# Infinity means there is no direct edge between two nodes.
INF = float("inf")


# Fixed node order.
# This order determines the row and column positions in the matrices.
NODES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


# Edge list based on the finalized graph/network design.
# Format: (source_node, destination_node, time_value, cost_value)
EDGES = [
    ("A", "B", 4.0, 100),
    ("A", "C", 5.0, 40),
    ("B", "D", 2.0, 90),
    ("B", "E", 3.0, 180),
    ("C", "E", 2.5, 120),
    ("C", "F", 4.0, 60),
    ("D", "E", 1.0, 80),
    ("D", "G", 1.5, 70),
    ("E", "H", 1.0, 90),
    ("E", "I", 2.0, 20),
    ("F", "H", 1.0, 55),
    ("F", "I", 2.0, 80),
    ("H", "I", 1.5, 40),
    ("G", "J", 1.2, 35),
    ("H", "J", 1.8, 45),
    ("I", "J", 1.5, 40),
]


def build_node_index(nodes: list[str]) -> dict[str, int]:
    """
    Creates a dictionary that maps each node label to its matrix index.

    Example:
        A -> 0
        B -> 1
        C -> 2
    """
    return {node: index for index, node in enumerate(nodes)}


def validate_edges(
    nodes: list[str],
    edges: list[tuple[str, str, float, float]],
) -> None:
    """
    Checks if all edges are valid before building the matrices.
    """
    node_set = set(nodes)

    for source, destination, time_value, cost_value in edges:
        if source not in node_set:
            raise ValueError(f"Invalid source node: {source}")

        if destination not in node_set:
            raise ValueError(f"Invalid destination node: {destination}")

        if time_value < 0:
            raise ValueError(
                f"Time cannot be negative for edge {source} -> {destination}"
            )

        if cost_value < 0:
            raise ValueError(
                f"Cost cannot be negative for edge {source} -> {destination}"
            )


def build_adjacency_matrices(
    nodes: list[str] = NODES,
    edges: list[tuple[str, str, float, float]] = EDGES,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """
    Builds the time and cost adjacency matrices.

    Returns:
        time_matrix: adjacency matrix using time as edge weight
        cost_matrix: adjacency matrix using cost as edge weight
        node_index: dictionary mapping node labels to matrix indices
    """
    validate_edges(nodes, edges)

    size = len(nodes)
    node_index = build_node_index(nodes)

    # Start with infinity to show that most node pairs have no direct edge.
    time_matrix = np.full((size, size), INF)
    cost_matrix = np.full((size, size), INF)

    # Cost from a node to itself is zero.
    np.fill_diagonal(time_matrix, 0)
    np.fill_diagonal(cost_matrix, 0)

    # Fill in the directed edges.
    for source, destination, time_value, cost_value in edges:
        row = node_index[source]
        col = node_index[destination]

        time_matrix[row][col] = time_value
        cost_matrix[row][col] = cost_value

    return time_matrix, cost_matrix, node_index


def build_tensor(
    time_matrix: np.ndarray,
    cost_matrix: np.ndarray,
) -> np.ndarray:
    """
    Stacks the time and cost matrices into one tensor.

    Tensor shape:
        (2, number_of_nodes, number_of_nodes)

    tensor[0] = time matrix
    tensor[1] = cost matrix
    """
    return np.stack([time_matrix, cost_matrix])


def get_neighbors(
    matrix: np.ndarray,
    node: str,
    node_index: dict[str, int],
    nodes: list[str] = NODES,
) -> list[tuple[str, float]]:
    """
    Returns the outgoing neighbors of a given node based on a selected matrix.

    This helper function can be used by the Dijkstra and Pareto teams.
    """
    row = node_index[node]
    neighbors = []

    for col, weight in enumerate(matrix[row]):
        if weight != INF and weight != 0:
            neighbors.append((nodes[col], weight))

    return neighbors


def print_matrix(matrix: np.ndarray, title: str) -> None:
    """
    Prints a readable version of a matrix.
    """
    print(f"\n{title}")
    print("-" * len(title))

    for row in matrix:
        print(row)


if __name__ == "__main__":
    time_matrix, cost_matrix, node_index = build_adjacency_matrices()
    tensor = build_tensor(time_matrix, cost_matrix)

    print("Node Index:")
    print(node_index)

    print_matrix(time_matrix, "Time Matrix")
    print_matrix(cost_matrix, "Cost Matrix")

    print("\nTensor Shape:")
    print(tensor.shape)

    print("\nTensor Meaning:")
    print("tensor[0] = time matrix")
    print("tensor[1] = cost matrix")

    print("\nNeighbors of A using time matrix:")
    print(get_neighbors(time_matrix, "A", node_index))