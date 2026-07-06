"""
main.py

Official CLI demo runner for the DSA Final Group Project.

This file connects:
- tensor_builder.py
- dijkstra_search.py
- pareto_search.py

Run using:
    python main.py
"""

from tensor_builder import build_adjacency_matrices, build_tensor, print_matrix, NODES
from dijkstra_search import dijkstra
from pareto_search import pareto_search, display_pareto_results


def show_available_nodes():
    print("\nAvailable nodes:")
    print(", ".join(NODES))


def format_path(path):
    if path is None:
        return "No path found"
    return " -> ".join(path)


def run_dijkstra_demo(matrix, source, destination, node_index, objective_name):
    """
    Runs Dijkstra for one objective and prints the result.

    Expected dijkstra output:
        path, total_value
    """
    path, total_value = dijkstra(matrix, source, destination, node_index)

    print(f"\n{objective_name.upper()} ROUTE")
    print("-" * 40)

    if path is None:
        print(f"No route found from {source} to {destination}.")
        return

    print(f"Route: {format_path(path)}")
    print(f"Total {objective_name}: {total_value:.2f}")


def run_pareto_demo(time_matrix, cost_matrix, source, destination, node_index):
    """
    Runs Pareto-front search and prints the result.
    """
    result = pareto_search(time_matrix, cost_matrix, source, destination, node_index)
    display_pareto_results(result, show_dominated=True)


def main():
    print("=" * 60)
    print("DSA FINAL GROUP PROJECT")
    print("Multi-Objective Network Optimization")
    print("=" * 60)

    time_matrix, cost_matrix, node_index = build_adjacency_matrices()
    tensor = build_tensor(time_matrix, cost_matrix)

    print("\nTensor successfully built.")
    print(f"Tensor shape: {tensor.shape}")
    print("tensor[0] = time matrix")
    print("tensor[1] = cost matrix")

    show_matrices = input("\nShow time and cost matrices first? (y/n): ").strip().lower()

    if show_matrices == "y":
        print_matrix(time_matrix, "Time Matrix")
        print_matrix(cost_matrix, "Cost Matrix")

    show_available_nodes()

    source = input("\nEnter start node: ").strip().upper()
    destination = input("Enter destination node: ").strip().upper()

    if source not in node_index:
        print(f"Invalid start node: {source}")
        return

    if destination not in node_index:
        print(f"Invalid destination node: {destination}")
        return

    if source == destination:
        print("Start and destination nodes must be different.")
        return

    print("\nChoose optimization mode:")
    print("1 - Fastest route using time")
    print("2 - Cheapest route using cost")
    print("3 - Pareto-front routes using time and cost")
    print("4 - Run all and compare")

    choice = input("\nEnter choice: ").strip()

    if choice == "1":
        run_dijkstra_demo(time_matrix, source, destination, node_index, "time")

    elif choice == "2":
        run_dijkstra_demo(cost_matrix, source, destination, node_index, "cost")

    elif choice == "3":
        run_pareto_demo(time_matrix, cost_matrix, source, destination, node_index)

    elif choice == "4":
        run_dijkstra_demo(time_matrix, source, destination, node_index, "time")
        run_dijkstra_demo(cost_matrix, source, destination, node_index, "cost")
        run_pareto_demo(time_matrix, cost_matrix, source, destination, node_index)

    else:
        print("Invalid choice. Please choose 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()
