# Code Documentation

This document describes every module in the project, its role, and its major
functions/classes. It's meant to be read alongside the source. Every
function below also carries a docstring and inline comments in the code
itself; this file adds the cross-module context. 

**Module role summary:**

| Module | Role |
|---|---|
| `tensor_builder.py` | Data structures -  builds the graph's matrices/tensor |
| `dijkstra_search.py` | **Main algorithm** - single-objective shortest path |
| `pareto_search.py` | **Main algorithm** -  multi-objective Pareto-front search |
| `ground_truth_generator.py` | **Testing utility only** -  brute-force oracle, not used in production output |
| `main.py` | Demo runner -  wires the above together, no algorithm logic |
| `tests/test_project.py` | Automated tests |

---

## Documentation Conventions

**Module docstrings.** Every `.py` file opens with a docstring naming the
file, its role, what task it satisfies, and whether it's a
main algorithm or a testing utility. For example:

```python
"""
pareto_search.py

Multi-Objective Pareto-Front Search
primary multi-objective Pareto-front search. requires tensor_builder.py
...
"""
```

versus the brute-force module, which is explicitly flagged as non-production:

```python
"""
ground_truth_generator.py

*** TESTING UTILITY -- NOT A MAIN ALGORITHM MODULE ***
...
"""
```

**Function docstrings.** Every major function has a docstring covering a
short summary, its `Args`/parameters, and its `Returns` value, e.g.:

```python
def dijkstra(matrix, source, target, node_index, nodes=NODES):
    """
    Dijkstra-style single-objective shortest path.

    Args:
        matrix     : either time_matrix or cost_matrix (NxN ndarray)
        source     : start node label, e.g. "A"
        target     : end node label, e.g. "J"
        node_index : dict from build_node_index() / build_adjacency_matrices()
        nodes      : ordered list of node labels

    Returns:
        path       : list of node labels from source to target, or None
                     if no path exists
        total_cost : float, total weight along that path (inf if unreachable)
    """
```

**Inline comments** explain *why* a step happens, not *what* the line
literally does — e.g. `# Early exit once target is settled` rather than
`# break out of the loop`.

The rest of this document is organized module-by-module, matching the
project structure in `README.md`.

---

## `tensor_builder.py` - Graph & Tensor Representation

Builds the directed, dual-weighted network into matrix form. Pure data
structure code; contains no pathfinding logic.

### Module-level data
- **`INF`** - `float("inf")`. Used to represent "no direct edge" in a matrix
  cell, as opposed to `0`, which is reserved for a node's distance to itself.
- **`NODES`** - the fixed, ordered list `["A", ..., "J"]`. The list's index
  order determines the row/column order of every matrix built from it.
- **`EDGES`** - the fixed edge list as `(source, destination, time, cost)`
  tuples, taken directly from the finalized network design.

#### `build_node_index(nodes) -> dict[str, int]`
Maps each node label to its integer row/column index (e.g. `{"A": 0, "B": 1,
...}`). Every other function that needs to go from a label to a matrix
position uses this mapping.

#### `validate_edges(nodes, edges) -> None`
Defensive check run before matrix construction. Raises `ValueError` if an
edge references a node that isn't in `nodes`, or if a time/cost value is
negative (negative weights would break the non-negative-weight assumption
that Dijkstra's algorithm depends on).

#### `build_adjacency_matrices(nodes=NODES, edges=EDGES) -> (time_matrix, cost_matrix, node_index)`
The core builder. For an N-node graph, returns two N×N `numpy` arrays:
- `time_matrix[i][j]` = time weight of the edge `i -> j`, or `INF` if no such
  edge exists.
- `cost_matrix[i][j]` = the same, for cost.
- The diagonal of both matrices is `0` (a node's "distance" to itself).

Because the graph is **directed**, `matrix[i][j]` and `matrix[j][i]` are
independent because an edge one way does not imply an edge the other way.

#### `build_tensor(time_matrix, cost_matrix) -> np.ndarray`
Stacks the two matrices with `np.stack` into one `(2, N, N)` array, so
`tensor[0]` is the time matrix and `tensor[1]` is the cost matrix. This is
the "vector of objective values per edge" representation described
in the project proposal - indexing `tensor[:, i, j]` gives the full
`[time, cost]` objective vector for edge `i -> j`.

#### `get_neighbors(matrix, node, node_index, nodes=NODES) -> list[tuple[str, float]]`
Given one objective matrix (time *or* cost) and a node, returns that node's
outgoing neighbors and edge weights, skipping `INF` (no edge) and `0`
(self-loop) entries. Shared helper used by both `dijkstra_search.py` and
`pareto_search.py`'s route generation.

#### `print_matrix(matrix, title) -> None`
Pretty-prints a matrix with a title/underline. Display-only, no return value.

---

## `dijkstra_search.py` - Single-Objective Optimizer (MAIN algorithm)

Answers "what's the best route for *one* objective at a time?" The objective is determined entirely by which matrix you pass in.

### Module-level data
`time_matrix`, `cost_matrix`, `node_index` are pre-built at import time from
`tensor_builder.NODES`/`EDGES`, so other files can `from dijkstra_search
import time_matrix, cost_matrix, node_index` without rebuilding the graph.

#### `dijkstra(matrix, source, target, node_index, nodes=NODES) -> (path, total_cost)`
The primary algorithm. A manually implemented Dijkstra's algorithm using
exactly the four components specified by the project plan:

1. **`dist`** - a dict tracking the best known cost-so-far to reach each
   node, initialized to `inf` except the source (`0.0`).
2. **`parent`** - a dict tracking which node we arrived from, used to
   reconstruct the winning path at the end by walking backwards from the
   target.
3. **`visited`** - a set of nodes whose shortest distance is finalized. Once
   a node is popped from the queue and added here, it's never re-relaxed.
4. **`pq`** - a binary min-heap (`heapq`) of `(cost_so_far, node)` pairs,
   always expanding the currently-cheapest frontier node next.

**Algorithm flow:** pop the cheapest node off `pq`; if already visited, skip
it; otherwise mark it visited, check for early exit if it's the target, then "relax" each unvisited neighbor (if the path through the current node is cheaper than that
neighbor's currently-known best, update `dist`/`parent` and push it onto the
heap).

**Return value:** `(path, total_cost)` where `path` is the ordered list of
node labels from `source` to `target`, or `(None, inf)` if `target` is
unreachable from `source`.

**Complexity:** O((V + E) log V) with a binary heap, standard for Dijkstra.

#### `scipy_shortest_path(matrix, source, target, node_index, nodes=NODES) -> (path, total_cost)`
**Verification helper, not part of the core algorithm.** Runs
`scipy.sparse.csgraph.dijkstra` on the same matrix and reconstructs the path
from SciPy's predecessor array. Used to cross-check that the hand-rolled
`dijkstra()` above produces the same answer as a well-tested external
implementation. The sparse conversion maps `INF` → `0` and re-zeroes the diagonal so self-loops are not misread as edges.

---

## `pareto_search.py` - Multi-Objective Pareto Search (MAIN algorithm)

**This is the primary multi-objective answer, not a validation step.**
Answers "what are *all* the routes worth considering when both time and cost
matter, and no single route wins on both?"

#### `class Route`
The core data structure for multi-objective search. Wraps a path with its
`[time, cost]` **objective vector** (a 2-element `numpy` array), so
dominance comparisons can be done as vector operations rather than
field-by-field comparisons.
- `.time`, `.cost` - convenience accessors into `objectives[0]`/`objectives[1]`.
- `.path_str` - the path formatted as `"A -> B -> ... -> J"`.
- `.to_dict()` - plain-dict form (`path`, `time`, `cost`, `objectives`), used
  when comparing against `ground_truth_generator`'s plain-dict routes in
  tests.

#### `generate_candidate_routes(time_matrix, cost_matrix, source, destination, node_index, nodes=NODES) -> list[Route]`
**Step 1 of the pipeline.** Iterative DFS (explicit stack, not recursion)
over every *simple* path (no repeated nodes) from `source` to `destination`.
Each completed path becomes a `Route`. Validates that `source`/`destination`
exist and are different nodes before searching.

#### `dominates(route_a, route_b) -> bool`
**The dominance rule.** `route_a` dominates `route_b` iff `route_a` is no
worse in *every* objective (`<=` on both time and cost) **and** strictly
better in *at least one*. Implemented with `numpy` array comparisons
(`np.all`, `np.any`) so it generalizes cleanly if a third objective (e.g.
reliability) were added later.

#### `dominance_relation(route_a, route_b) -> str`
Human-readable explanation of the relationship between two routes (`"A
dominates B"`, `"B dominates A"`, `"identical"`, or `"non-dominated
(incomparable)"`). Debugging/explanation aid, not used in the core pipeline.

### `pareto_filter(routes) -> list[Route]`
**Step 2 of the pipeline.** For each candidate, checks whether *any* other
candidate dominates it; if none does, it survives onto the Pareto front.
Returns the front sorted by `(time, cost)` ascending.\
**Complexity:** O(n² · m) where n = number of candidate routes, m = number
of objectives (2 here). Acceptable for the small graphs this project
targets, and explicitly noted as such in the code.

#### `identify_dominated_routes(routes) -> list[tuple[Route, Route]]`
Companion to `pareto_filter` that also records *why* each eliminated route
was cut, as `(dominated_route, route_that_dominated_it)` pairs - used for
the "ELIMINATED ROUTES" section of the report and the grey points on the
objective-space plot.

#### `pareto_search(time_matrix, cost_matrix, source, destination, node_index, nodes=NODES) -> dict`
**The full pipeline / main entry point**, matching the plan exactly:
1. DFS to generate all simple candidate routes (`generate_candidate_routes`).
2. Represent each as a `[time, cost]` vector (`Route`).
3. Filter by Pareto dominance (`pareto_filter`).
4. Return the non-dominated set as the final answer.

Returns a dict with `source`, `destination`, `candidates` (all routes
found), `pareto_front` (the answer), `dominated_pairs`, `dominated_count`,
`fastest_route` (lowest-time route *on the front*), and `cheapest_route`
(lowest-cost route *on the front*). Handles the "no path exists" case by
returning empty/`None` fields instead of raising.

#### `display_pareto_results(result, show_dominated=True) -> None`
Formats and prints the full report: summary counts, a table of Pareto-front
routes annotated with `< fastest`/`< cheapest` markers, the raw objective
vectors, a trade-off summary (how much more the fastest route costs vs. the
cheapest, and vice versa), and optionally the eliminated routes with their
dominating route. Purely a presentation function, it doesn't compute
anything new.

#### `plot_objective_space(result) -> None`
Draws a matplotlib scatter/step plot of every candidate route in
`(time, cost)` space: grey points for dominated routes, red points for the
Pareto front, a dashed staircase line connecting the front, and path labels
on each front point. Visual counterpart to `display_pareto_results`.

---

## `ground_truth_generator.py` - Testing Utility (NOT a main algorithm module)

> **This module is deliberately kept separate from, and is never imported
> by, `dijkstra_search.py` or `pareto_search.py`.** Its only job is to be an
> independent, "obviously correct" oracle that the real algorithms are
> checked against. If it shared code with the algorithms it's meant to
> validate, a shared bug could pass both silently - so it re-implements
> everything from scratch, deliberately trading performance for legibility.

#### `enumerate_all_simple_paths(source, destination) -> list[dict]`
Brute-force backtracking DFS (recursive, with explicit `visited`/
`current_path` state) over every simple path between two nodes. Each result
is a plain dict: `{"path": [...], "time": ..., "cost": ...}`. This is the
brute-force equivalent of `pareto_search.generate_candidate_routes`, written
independently.

#### `brute_force_best_by_objective(routes, objective) -> dict`
Linear scan for the route with the minimum `"time"` or `"cost"` value - the
simplest possible "best route" check, used as the ground truth for
`dijkstra_search.dijkstra`'s output.

#### `_route_is_dominated_by(candidate, other) -> bool` / `brute_force_pareto_front(routes) -> list[dict]`
A from-scratch re-implementation of the same dominance rule used in
`pareto_search.dominates`/`pareto_filter`, but written with plain
comparison logic instead of `numpy` vectors, and with an independent
O(n²) all-pairs scan. Used as the ground truth for
`pareto_search.pareto_filter`'s output.

#### `generate_ground_truth(source, destination) -> dict`
Convenience entry point that runs all of the above and returns
`all_candidates`, `best_by_time`, `best_by_cost`, and `pareto_front` in one
dict. This is what the test suite calls to get "the right answer" for a
given source/destination pair.

#### `paths_match(path_a, path_b) -> bool` / `route_sets_match(routes_a, routes_b, tolerance=1e-9) -> bool`
Comparison helpers built specifically for testing: `route_sets_match`
treats two route lists as **unordered sets** (since DFS traversal order
is not guaranteed to match between two independent implementations) and
compares them via a rounded `(path, time, cost)` key, so floating-point
rounding differences within `tolerance` don't cause false test failures.

---

## `main.py` - Demo Runner

Not an algorithm module. It only imports and calls the real modules above
and handles user I/O. Every function has a short docstring; the notes below
add the "why" behind each one

#### `show_available_nodes() -> None`
Prints the full `NODES` list (`A–J`) so the user knows what's valid before
being asked to type a start/destination node. Display-only, no return value.

#### `format_path(path: list[str] | None) -> str`
Converts a path list like `["A", "B", "J"]` into the readable string
`"A -> B -> J"`. Returns the literal string `"No path found"` if `path` is
`None`, so callers can print a result unconditionally without a separate
`if path is None` branch at every call site.

#### `run_dijkstra_demo(matrix, source, destination, node_index, objective_name) -> None`
Runs `dijkstra_search.dijkstra` once for whichever matrix (time or cost) is
passed in, then prints a labeled block (`"TIME ROUTE"` / `"COST ROUTE"`)
containing the route (via `format_path`) and the total value formatted to
two decimal places. `objective_name` is a plain string (`"time"` or
`"cost"`) used only for the printed labels - the actual objective being
optimized is determined by which `matrix` argument is passed in, exactly
like `dijkstra()` itself. Handles the unreachable case by printing
`"No route found from {source} to {destination}."` instead of trying to
format `None`/`inf`.

#### run_pareto_demo(time_matrix, cost_matrix, source, destination, node_index) -> None
Calls `pareto_search.pareto_search` with both matrices and prints the full
result via `pareto_search.display_pareto_results(result, show_dominated=True)`
- i.e. it always shows the eliminated/dominated routes as well as the
Pareto front.

#### main() -> None
The CLI entry point and orchestration logic, in order:
1. Prints the project banner.
2. Unconditionally builds the time matrix, cost matrix, and tensor (`build_adjacency_matrices`, `build_tensor`) and prints the tensor's shape
3. Asks `"Show time and cost matrices first? (y/n)"`; if `y`, prints both matrices via `tensor_builder.print_matrix`.
4. Calls `show_available_nodes()`.
5. Reads a start node and a destination node from the user.
**Unlike a typical input-validation loop, this does not re-prompt.** If
either node isn't in `node_index`, or the two nodes are the same, `main()`
prints an error message and returns immediately. The user has to re-run
`python main.py` to try again. This is a deliberate simplicity trade-off
for a course demo, not an oversight; it's called out explicitly (and
in the README) so it isn't mistaken for a bug.
6. Prints the four-option mode menu and reads the user's choice.
7. Dispatches based on that choice:
    - 1 → `run_dijkstra_demo(time_matrix, ..., "time")`
    - 2 → `run_dijkstra_demo(cost_matrix, ..., "cost")`
    - 3 → `run_pareto_demo(...)`
    - 4 → all three of the above, in that order
    - anything else → prints `"Invalid choice. Please choose 1, 2, 3, or 4."`
and returns.

`main.py` contains no pathfinding logic itself, every actual computation is delegated to `tensor_builder`, `dijkstra_search`, or `pareto_search`. 

---

## `tests/test_project.py` - Automated Tests
Uses `pytest`. Every test either checks a module's output directly (e.g.
matrix shape, diagonal values) or cross-checks a main algorithm's output
against `ground_truth_generator`'s brute-force answer, on both the fixed
project graph and several randomized graphs (via the local
`randomize_test_graph` helper, adapted from the project notebook) so
correctness isn't only demonstrated on one hand-picked example.

- `test_matrix_shapes_and_diagonal`, `test_tensor_stacks_time_then_cost`,
  `test_known_edge_values_appear_in_matrices`
- `test_dijkstra_fastest_route_on_fixed_graph`,
  `test_dijkstra_cheapest_route_on_fixed_graph`,
  `test_dijkstra_returns_none_when_unreachable`,
  `test_dijkstra_matches_reference_on_random_graphs` - the
  last one against a second, independently written reference Dijkstra
  (`_reference_dijkstra`, local to the test file - not imported from
  `dijkstra_search.py`).
- `test_pareto_front_matches_ground_truth_on_fixed_graph`,
  `test_pareto_front_matches_ground_truth_on_random_graphs`,
  `test_no_route_dominates_another_within_the_front` - the
  last one a direct sanity check on the Pareto-front definition itself
  (no surviving route may dominate another surviving route).

Run all of them with `pytest -v` from the project root.
