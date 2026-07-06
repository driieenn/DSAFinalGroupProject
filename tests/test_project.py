"""
tests/test_project.py

Automated tests for the DSA Final Group Project.

These tests verify:
- matrix and tensor construction
- Dijkstra output against brute-force ground truth
- Pareto-front output against brute-force ground truth
- route reconstruction
- full integration output
"""

from __future__ import annotations

import contextlib
import io
import math

import numpy as np

import ground_truth_generator as gtg
from dijkstra_search import dijkstra
from pareto_search import display_pareto_results, pareto_search
from tensor_builder import NODES, EDGES, build_adjacency_matrices, build_tensor


def route_key(route: dict) -> tuple:
    """
    Converts a route dictionary into a comparable key.
    """
    return (
        tuple(route["path"]),
        round(float(route["time"]), 9),
        round(float(route["cost"]), 9),
    )


def test_matrix_and_tensor_construction() -> None:
    """
    Tests whether the time matrix, cost matrix, and tensor are built correctly.
    """
    time_matrix, cost_matrix, node_index = build_adjacency_matrices()
    tensor = build_tensor(time_matrix, cost_matrix)

    size = len(NODES)

    assert time_matrix.shape == (size, size)
    assert cost_matrix.shape == (size, size)
    assert tensor.shape == (2, size, size)

    assert np.array_equal(tensor[0], time_matrix)
    assert np.array_equal(tensor[1], cost_matrix)

    for node in NODES:
        idx = node_index[node]
        assert time_matrix[idx][idx] == 0
        assert cost_matrix[idx][idx] == 0

    for source, destination, time_value, cost_value in EDGES:
        row = node_index[source]
        col = node_index[destination]

        assert time_matrix[row][col] == float(time_value)
        assert cost_matrix[row][col] == float(cost_value)


def test_dijkstra_fastest_route_vs_ground_truth() -> None:
    """
    Tests Dijkstra using the time matrix against brute-force best-by-time.
    """
    time_matrix, cost_matrix, node_index = build_adjacency_matrices()

    source = "A"
    destination = "J"

    path, total_time = dijkstra(time_matrix, source, destination, node_index)
    ground_truth = gtg.generate_ground_truth(source, destination)

    assert path == ground_truth["best_by_time"]["path"]
    assert math.isclose(total_time, ground_truth["best_by_time"]["time"])


def test_dijkstra_cheapest_route_vs_ground_truth() -> None:
    """
    Tests Dijkstra using the cost matrix against brute-force best-by-cost.
    """
    time_matrix, cost_matrix, node_index = build_adjacency_matrices()

    source = "A"
    destination = "J"

    path, total_cost = dijkstra(cost_matrix, source, destination, node_index)
    ground_truth = gtg.generate_ground_truth(source, destination)

    assert path == ground_truth["best_by_cost"]["path"]
    assert math.isclose(total_cost, ground_truth["best_by_cost"]["cost"])


def test_dijkstra_unreachable_pair() -> None:
    """
    Tests that Dijkstra returns no path for an unreachable directed pair.
    """
    time_matrix, cost_matrix, node_index = build_adjacency_matrices()

    path, total = dijkstra(time_matrix, "J", "A", node_index)

    assert path is None
    assert total == float("inf")


def test_route_reconstruction() -> None:
    """
    Tests whether Dijkstra reconstructs a valid path from source to destination.
    """
    time_matrix, cost_matrix, node_index = build_adjacency_matrices()

    source = "A"
    destination = "J"

    path, total = dijkstra(time_matrix, source, destination, node_index)

    assert path is not None
    assert path[0] == source
    assert path[-1] == destination
    assert total >= 0

    for current_node, next_node in zip(path, path[1:]):
        row = node_index[current_node]
        col = node_index[next_node]
        assert not np.isinf(time_matrix[row][col])


def test_pareto_front_vs_ground_truth() -> None:
    """
    Tests whether Pareto-front search matches brute-force Pareto-front output.
    """
    time_matrix, cost_matrix, node_index = build_adjacency_matrices()

    source = "A"
    destination = "J"

    result = pareto_search(time_matrix, cost_matrix, source, destination, node_index)
    ground_truth = gtg.generate_ground_truth(source, destination)

    pareto_front_dicts = [route.to_dict() for route in result["pareto_front"]]

    expected = {route_key(route) for route in ground_truth["pareto_front"]}
    actual = {route_key(route) for route in pareto_front_dicts}

    assert actual == expected


def test_pareto_candidates_vs_ground_truth() -> None:
    """
    Tests whether Pareto search generates the same candidate routes as brute force.
    """
    time_matrix, cost_matrix, node_index = build_adjacency_matrices()

    source = "A"
    destination = "J"

    result = pareto_search(time_matrix, cost_matrix, source, destination, node_index)
    ground_truth = gtg.generate_ground_truth(source, destination)

    candidate_dicts = [route.to_dict() for route in result["candidates"]]

    expected = {route_key(route) for route in ground_truth["all_candidates"]}
    actual = {route_key(route) for route in candidate_dicts}

    assert actual == expected


def test_no_pareto_route_dominates_another() -> None:
    """
    Tests that no route in the returned Pareto front dominates another route.
    """
    time_matrix, cost_matrix, node_index = build_adjacency_matrices()

    result = pareto_search(time_matrix, cost_matrix, "A", "J", node_index)
    front = result["pareto_front"]

    for i, route_a in enumerate(front):
        for j, route_b in enumerate(front):
            if i == j:
                continue

            no_worse = route_a.time <= route_b.time and route_a.cost <= route_b.cost
            strictly_better = route_a.time < route_b.time or route_a.cost < route_b.cost

            assert not (no_worse and strictly_better)


def test_full_integration_output() -> None:
    """
    Tests whether the Pareto display function prints the expected report.
    """
    time_matrix, cost_matrix, node_index = build_adjacency_matrices()

    source = "A"
    destination = "J"

    result = pareto_search(time_matrix, cost_matrix, source, destination, node_index)

    buffer = io.StringIO()

    with contextlib.redirect_stdout(buffer):
        display_pareto_results(result, show_dominated=False)

    output = buffer.getvalue()

    assert "PARETO-FRONT SEARCH" in output
    assert "Candidates generated" in output
    assert "Pareto-optimal found" in output

    for route in result["pareto_front"]:
        assert route.path_str in output
