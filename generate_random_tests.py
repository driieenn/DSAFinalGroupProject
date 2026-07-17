"""
generate_random_tests.py

Random-graph benchmark harness for the project algorithms.

The script generates nonnegative directed graphs in the same tuple format
used by tensor_builder.py, then times the project's Dijkstra,
Pareto-front, and brute-force ground-truth logic on those graphs.

The graph generator is tuned so the average edge counts are roughly:
small = 15, medium = 100, large = 300.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from random import Random
from statistics import mean
from time import perf_counter
from typing import Iterator

import ground_truth_generator as gtg
from dijkstra_search import dijkstra
from pareto_search import pareto_search
from tensor_builder import build_adjacency_matrices


BRUTE_FORCE_NODE_LIMIT = 30
PARETO_NODE_LIMIT = 100


@dataclass
class GraphCase:
	category: str
	index: int
	nodes: list[str]
	edges: list[tuple[str, str, float, float]]


def build_adjacency_list(
	nodes: list[str],
	edges: list[tuple[str, str, float, float]],
) -> dict[str, list[tuple[str, float, float]]]:
	adjacency: dict[str, list[tuple[str, float, float]]] = {node: [] for node in nodes}
	for source, destination, time_value, cost_value in edges:
		adjacency[source].append((destination, float(time_value), float(cost_value)))
	return adjacency


@contextmanager
def patched_ground_truth_adjacency(
	adjacency: dict[str, list[tuple[str, float, float]]],
) -> Iterator[None]:
	original_adjacency = gtg._ADJACENCY
	gtg._ADJACENCY = adjacency
	try:
		yield
	finally:
		gtg._ADJACENCY = original_adjacency


def _random_weight_pair(rng: Random) -> tuple[float, float]:
	return round(rng.uniform(1.0, 25.0), 2), round(rng.uniform(1.0, 250.0), 2)


def generate_directed_graph_with_target_edges(
	node_count: int,
	target_edge_count: int,
	rng: Random,
) -> tuple[list[str], list[tuple[str, str, float, float]]]:
	nodes = [f"N{i}" for i in range(node_count)]
	edge_map: dict[tuple[str, str], tuple[float, float]] = {}

	# Guarantee a path from the first node to the last node.
	for index in range(node_count - 1):
		source = nodes[index]
		destination = nodes[index + 1]
		edge_map[(source, destination)] = _random_weight_pair(rng)

	all_forward_pairs = [
		(nodes[source_index], nodes[destination_index])
		for source_index in range(node_count)
		for destination_index in range(source_index + 1, node_count)
		if (nodes[source_index], nodes[destination_index]) not in edge_map
	]
	rng.shuffle(all_forward_pairs)

	# Keep the graph sparse enough to match the requested average edge counts,
	# but never below the guaranteed spine path.
	target_edge_count = max(target_edge_count, len(edge_map))
	target_edge_count = min(target_edge_count, len(edge_map) + len(all_forward_pairs))
	additional_edges_needed = target_edge_count - len(edge_map)

	for source, destination in all_forward_pairs[:additional_edges_needed]:
		edge_map[(source, destination)] = _random_weight_pair(rng)

	edges = [
		(source, destination, time_value, cost_value)
		for (source, destination), (time_value, cost_value) in edge_map.items()
	]
	rng.shuffle(edges)
	return nodes, edges


def create_graph_suite(seed: int, small_count: int, medium_count: int, large_count: int) -> list[GraphCase]:
	rng = Random(seed)
	graph_cases: list[GraphCase] = []

	for index in range(small_count):
		node_count = rng.randint(6, 12)
		target_edge_count = rng.randint(13, 17)
		nodes, edges = generate_directed_graph_with_target_edges(node_count, target_edge_count, rng)
		graph_cases.append(GraphCase("small", index + 1, nodes, edges))

	for index in range(medium_count):
		node_count = rng.randint(15, 49)
		target_edge_count = rng.randint(90, 110)
		nodes, edges = generate_directed_graph_with_target_edges(node_count, target_edge_count, rng)
		graph_cases.append(GraphCase("medium", index + 1, nodes, edges))

	for index in range(large_count):
		node_count = 100
		target_edge_count = rng.randint(270, 330)
		nodes, edges = generate_directed_graph_with_target_edges(node_count, target_edge_count, rng)
		graph_cases.append(GraphCase("large", index + 1, nodes, edges))

	return graph_cases


def time_call(function, *args, **kwargs):
	start = perf_counter()
	result = function(*args, **kwargs)
	elapsed = perf_counter() - start
	return elapsed, result


def format_seconds(value: float | None) -> str:
	if value is None:
		return "skipped"
	return f"{value * 1000.0:.3f} ms"


def benchmark_graph(graph_case: GraphCase) -> dict:
	time_matrix, cost_matrix, node_index = build_adjacency_matrices(graph_case.nodes, graph_case.edges)
	source = graph_case.nodes[0]
	destination = graph_case.nodes[-1]

	dijkstra_time_time, time_result = time_call(
		dijkstra,
		time_matrix,
		source,
		destination,
		node_index,
		graph_case.nodes,
	)
	dijkstra_cost_time, cost_result = time_call(
		dijkstra,
		cost_matrix,
		source,
		destination,
		node_index,
		graph_case.nodes,
	)

	result = {
		"category": graph_case.category,
		"index": graph_case.index,
		"nodes": len(graph_case.nodes),
		"edges": len(graph_case.edges),
		"dijkstra_time": dijkstra_time_time,
		"dijkstra_cost": dijkstra_cost_time,
		"pareto_time": None,
		"brute_force_time": None,
		"pareto_routes": None,
		"brute_force_routes": None,
		"brute_force_status": "skipped",
		"pareto_status": "skipped",
	}

	if len(graph_case.nodes) <= PARETO_NODE_LIMIT:
		pareto_time, pareto_result = time_call(
			pareto_search,
			time_matrix,
			cost_matrix,
			source,
			destination,
			node_index,
			graph_case.nodes,
		)
		result["pareto_status"] = "ok"
		result["pareto_time"] = pareto_time
		result["pareto_routes"] = len(pareto_result["pareto_front"])

	if len(graph_case.nodes) <= BRUTE_FORCE_NODE_LIMIT:
		adjacency = build_adjacency_list(graph_case.nodes, graph_case.edges)

		with patched_ground_truth_adjacency(adjacency):
			brute_force_time, ground_truth = time_call(gtg.generate_ground_truth, source, destination)

		time_path, time_total = time_result
		cost_path, cost_total = cost_result

		assert time_path == ground_truth["best_by_time"]["path"]
		assert cost_path == ground_truth["best_by_cost"]["path"]
		result.update(
			{
				"brute_force_time": brute_force_time,
				"brute_force_routes": len(ground_truth["pareto_front"]),
				"brute_force_status": "ok",
				"dijkstra_time_total": time_total,
				"dijkstra_cost_total": cost_total,
			}
		)

	return result


def summarize_category(rows: list[dict]) -> dict:
	pareto_rows = [row for row in rows if row["pareto_status"] == "ok"]
	brute_force_rows = [row for row in rows if row["brute_force_status"] == "ok"]

	return {
		"graphs": len(rows),
		"avg_nodes": mean(row["nodes"] for row in rows),
		"avg_edges": mean(row["edges"] for row in rows),
		"avg_dijkstra": mean(row["dijkstra_time"] + row["dijkstra_cost"] for row in rows) / 2.0,
		"pareto_graphs": len(pareto_rows),
		"brute_force_graphs": len(brute_force_rows),
		"avg_pareto": mean(row["pareto_time"] for row in pareto_rows) if pareto_rows else None,
		"avg_brute_force": mean(row["brute_force_time"] for row in brute_force_rows) if brute_force_rows else None,
	}


def print_summary(results: dict[str, list[dict]]) -> None:
	print("Random Graph Benchmark")
	print(f"Pareto node limit: {PARETO_NODE_LIMIT} nodes")
	print(f"Brute force node limit: {BRUTE_FORCE_NODE_LIMIT} nodes")
	print("-" * 90)
	print(
		f"{'Category':<10} {'Graphs':>6} {'Avg nodes':>10} {'Avg edges':>10} "
		f"{'Dijkstra':>12} {'Pareto':>12} {'Brute force':>14} {'Pareto ok':>9} {'BF ok':>7}"
	)
	print("-" * 90)

	for category in ("small", "medium", "large"):
		summary = summarize_category(results[category])
		print(
			f"{category:<10} {summary['graphs']:>6} {summary['avg_nodes']:>10.1f} {summary['avg_edges']:>10.1f} "
			f"{format_seconds(summary['avg_dijkstra']):>12} {format_seconds(summary['avg_pareto']):>12} "
			f"{format_seconds(summary['avg_brute_force']):>14} {summary['pareto_graphs']:>9} {summary['brute_force_graphs']:>7}"
		)

	print("-" * 90)
	print(
		"Dijkstra is timed on every generated graph. Pareto runs on graphs up to the Pareto "
		"node limit, while brute force is only attempted up to the brute-force node limit."
	)


def run_benchmarks(seed: int, small_count: int, medium_count: int, large_count: int) -> None:
	graph_suite = create_graph_suite(seed, small_count, medium_count, large_count)
	categorized_results: dict[str, list[dict]] = {"small": [], "medium": [], "large": []}

	for graph_case in graph_suite:
		result = benchmark_graph(graph_case)
		categorized_results[graph_case.category].append(result)

	print_summary(categorized_results)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Generate random directed weighted graphs and benchmark the project algorithms.")
	parser.add_argument("--seed", type=int, default=2301, help="Random seed used for graph generation.")
	parser.add_argument("--small-count", type=int, default=100, help="Number of small graphs to generate.")
	parser.add_argument("--medium-count", type=int, default=50, help="Number of medium graphs to generate.")
	parser.add_argument("--large-count", type=int, default=10, help="Number of large graphs to generate.")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	run_benchmarks(args.seed, args.small_count, args.medium_count, args.large_count)


if __name__ == "__main__":
	main()
