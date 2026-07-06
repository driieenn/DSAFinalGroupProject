# Code Documentation
## Multi-Objective Network Optimization — LT6 Final Project

This document describes, module by module and function by function, how the codebase
implements the design described in the project proposal: graph/tensor representation,
single-objective Dijkstra-style optimization, Pareto-based multi-objective search, and
brute-force ground truth testing.

---

## 1. `tensor_builder.py`

**Purpose:** Build the graph's data structures — two weighted adjacency matrices (time,
cost) and a combined tensor — from a node list and an edge list. This is the foundation
every other module (Dijkstra, Pareto search, ground truth) is built on top of.

### Constants
- `INF = float("inf")` — sentinel meaning "no direct edge between these two nodes."
- `NODES` — the fixed, ordered list of node labels (`["A", "B", ..., "J"]`). The position of
  a node in this list determines its row/column index in every matrix.
- `EDGES` — the list of `(source, destination, time_value, cost_value)` tuples that defines
  the base graph. The graph is **directed**: an edge `A -> B` does not imply `B -> A`.

### `build_node_index(nodes: list) -> dict`
Maps each node label to its integer index (`"A" -> 0, "B" -> 1, ...`). Used everywhere a
node label needs to be converted into a matrix row/column.

### `validate_edges(nodes: list, edges: list) -> None`
Defensive check run before building any matrix. Raises `ValueError` if:
- an edge references a node that isn't in `nodes`, or
- an edge has a negative time or cost value (Dijkstra requires non-negative weights).

### `build_adjacency_matrices(nodes=NODES, edges=EDGES) -> (time_matrix, cost_matrix, node_index)`
1. Validates the edge list.
2. Creates two `N × N` NumPy arrays filled with `INF` (meaning "no edge").
3. Sets the diagonal to `0` (a node's distance to itself).
4. For every edge `(src, dst, t, c)`, sets `time_matrix[src][dst] = t` and
   `cost_matrix[src][dst] = c`.

Returns the two matrices plus the `node_index` dictionary so callers can translate labels
to indices.

### `build_tensor(time_matrix, cost_matrix) -> np.ndarray`
Stacks the two matrices into a single tensor of shape `(2, N, N)` using `np.stack`.
- `tensor[0]` = time matrix (objective 0)
- `tensor[1]` = cost matrix (objective 1)

This is the literal implementation of the proposal's idea that "each edge is no longer a
single weight, but a vector of objective values" — indexing `tensor[:, i, j]` gives the full
`[time, cost]` vector for the edge `i -> j`.

### `get_neighbors(matrix, node, node_index, nodes=NODES) -> list[(neighbor, weight)]`
Given **any one** objective matrix (time or cost), returns the outgoing neighbors of `node`
and the edge weight to each. Skips entries equal to `INF` (no edge) or `0` (self). This is
the shared building block used by both the Dijkstra optimizer and the Pareto route
generator — whichever matrix you pass in decides which objective is being explored.

### `print_matrix(matrix, title) -> None`
Debug/demo helper that prints a labeled matrix row by row.

---

## 2. Single-Objective Optimizer — `dijkstra()` (developed in the notebook)

### `dijkstra(matrix, source, target, node_index, nodes=NODES) -> (path, total_cost)`
A manual, from-scratch implementation of Dijkstra's algorithm using the exact components
called for in the proposal:

| Component | Variable |
|---|---|
| Distance/cost tracker | `dist: dict[node -> float]`, initialized to `inf` except the source (`0.0`) |
| Parent tracker (for path reconstruction) | `parent: dict[node -> node or None]` |
| Visited set | `visited: set()` |
| Priority queue | `pq`, a binary heap of `(cost_so_far, node)` via `heapq` |

**Algorithm:**
1. Pop the lowest-cost unvisited node from the heap.
2. Mark it visited; if it's the target, stop early.
3. "Relax" every unvisited neighbor (from `get_neighbors()` on the chosen matrix): if going
   through the current node produces a lower cost than previously known, update `dist` and
   `parent`, and push the neighbor back onto the heap.
4. When the loop ends, reconstruct the path by walking `parent` pointers backward from
   `target` to `source`, then reverse it.

**Key design point:** the function is objective-agnostic — pass `time_matrix` to get the
fastest route, or `cost_matrix` to get the cheapest route. This directly implements the
proposal's requirement that "the optimizer handles one goal at a time" using whichever
matrix is selected.

**Complexity:** `O((V + E) log V)` with a binary heap, standard for Dijkstra.

### `scipy_shortest_path()` (verification helper)
Wraps `scipy.sparse.csgraph.dijkstra` on the same matrix to independently confirm the
hand-written `dijkstra()` produces the same path and total cost. Used purely as a sanity
check, not as part of the "real" algorithm the project is graded on.

---

## 3. `pareto_search.py`

**Purpose:** Implement the multi-objective route search described in the proposal — return
the **set** of non-dominated (Pareto-optimal) routes between two nodes, instead of a single
best answer.

```python
from pareto_search import pareto_search, time_matrix, cost_matrix, node_index

result = pareto_search(time_matrix, cost_matrix, "A", "J", node_index)
result["candidates"]      # every simple path found, with time/cost
result["pareto_front"]    # non-dominated routes (the multi-objective answer)
result["fastest_route"]   # fastest route on the Pareto front
result["cheapest_route"]  # cheapest route on the Pareto front
```

### Module-level setup
At import time, `pareto_search.py` imports `NODES`, `EDGES`, `INF`, and
`build_adjacency_matrices` from `tensor_builder.py` and immediately calls
`time_matrix, cost_matrix, node_index = build_adjacency_matrices(NODES, EDGES)`. This means
`time_matrix`/`cost_matrix`/`node_index` for the **base** graph are already built and
importable directly from `pareto_search` — callers don't have to rebuild them unless they're
working with a different (e.g. randomized) graph, in which case they call
`build_adjacency_matrices()` themselves with their own node/edge lists and pass the results
into `pareto_search()` explicitly.

### `class Route`
Represents one candidate path as a `[time, cost]` objective vector.

- `__init__(path, total_time, total_cost)` — stores `path` (list of node labels) and
  `objectives = np.array([total_time, total_cost])`.
- `.time` / `.cost` — convenience properties reading `objectives[0]` / `objectives[1]`.
- `.path_str` — `"A -> B -> D -> G -> J"` style string for display.
- `.to_dict()` — serializes the route (used to compare against ground truth dictionaries).

Using a small class instead of raw tuples keeps the objective vector generalizable to more
than two objectives later (e.g. adding reliability) without changing the dominance logic.

### `generate_candidate_routes(time_matrix, cost_matrix, source, destination, node_index, nodes=NODES) -> list[Route]`
Enumerates **every simple path** (no repeated nodes) from `source` to `destination` using an
**iterative DFS** (an explicit stack, not recursion, to avoid recursion-depth issues):

- Stack entries are `(current_node_index, visited_set, cumulative_time, cumulative_cost, path_so_far)`.
- At each step, every unvisited neighbor (from *both* matrices simultaneously, since every
  edge has both a time and a cost value) is pushed onto the stack with updated running totals.
- When the destination is reached, the accumulated path/time/cost is packaged into a `Route`
  object and added to the result list.

This is Step 1 of the pipeline described in the proposal: "starting from the source node,
generate candidate routes through graph traversal."

**Complexity:** exponential in the worst case (number of simple paths can grow factorially),
which is why this is explicitly restricted to small graphs in both the proposal and the
limitations section.

### `dominates(route_a: Route, route_b: Route) -> bool`
Implements the exact Pareto dominance definition from the proposal:

> route_a dominates route_b if and only if route_a is no worse than route_b in **all**
> objectives, and strictly better in **at least one**.

Implemented with two vectorized NumPy comparisons:
```python
no_worse_in_all       = np.all(route_a.objectives <= route_b.objectives)
better_in_at_least_one = np.any(route_a.objectives <  route_b.objectives)
```
Both lower time and lower cost are "better," so no transformation is needed for this pair of
objectives.

### `dominance_relation(route_a, route_b) -> str`
Human-readable helper ("A dominates B" / "B dominates A" / "identical" / "non-dominated
(incomparable)") — mainly used for explanation/debugging in the notebook.

### `pareto_filter(routes: list[Route]) -> list[Route]`
Given all candidate routes, keeps only the ones **not dominated by any other route** in the
list (an `O(n² · m)` pairwise comparison, where `n` = number of routes and `m` = number of
objectives — acceptable for the small graphs used here). Results are sorted by ascending
time, then ascending cost, for readability.

### `identify_dominated_routes(routes: list[Route]) -> list[(dominated, dominator)]`
Same pairwise scan as `pareto_filter`, but instead of just keeping survivors, it records
**why** each eliminated route was eliminated (and by which specific route), so the report
can explain the filtering decision rather than just presenting a final list.

### `pareto_search(time_matrix, cost_matrix, source, destination, node_index, nodes=NODES) -> dict`
The main entry point — orchestrates the full multi-objective pipeline in one call:

1. `generate_candidate_routes(...)`
2. `pareto_filter(candidates)`
3. `identify_dominated_routes(candidates)`
4. Picks `fastest_route` and `cheapest_route` **from the Pareto front** (not from all
   candidates), since those two extremes of the front are usually the most interesting to a
   user.

Returns a dictionary with `source`, `destination`, `candidates`, `pareto_front`,
`dominated_pairs`, `dominated_count`, `fastest_route`, `cheapest_route`. As emphasized in the
project's requirements-coverage notes, **the Pareto front returned here is the actual answer
to the multi-objective problem — not a validation step.** (Ground-truth brute force,
described below, plays that validating role instead.)

### `display_pareto_results(result: dict, show_dominated=True) -> None`
Formats the `pareto_search()` output into a readable report:
- Summary counts (candidates generated, dominated removed, Pareto-optimal found).
- A table of the Pareto-optimal routes with time/cost and a "fastest" / "cheapest" /
  "fastest & cheapest" annotation.
- The raw `[time, cost]` objective vectors for each surviving route.
- A trade-off summary comparing the fastest and cheapest routes on the front (extra cost to
  go fast vs. extra time to go cheap).
- Optionally, the list of eliminated routes and which route dominated each one.

### `plot_objective_space(result: dict) -> None`
Visualizes every candidate route as a point in (time, cost) space:
- Grey points = dominated (eliminated) routes.
- Red points = the Pareto-optimal front.
- A dashed red step line connecting the front (the classic "staircase" shape of a 2-D Pareto
  frontier).
- Each Pareto-optimal point is annotated with its path.

This is the direct visual counterpart to the textual report — it shows *why* the surviving
routes are optimal (nothing grey sits below/left of the red staircase).

---

## 4. `ground_truth_generator.py`

**Purpose:** An **independent** brute-force reference implementation used only to verify the
Dijkstra and Pareto outputs. It deliberately does **not** import or call anything from
`pareto_search.py` — it only depends on `tensor_builder.py` (`NODES`, `EDGES`) — so a match
between this module's output and `pareto_search`'s output is real evidence of correctness,
not the algorithm agreeing with itself.

```python
from ground_truth_generator import generate_ground_truth

gt = generate_ground_truth("A", "J")
gt["all_candidates"]   # every simple path found, with time/cost
gt["best_by_time"]     # expected single-objective answer (minimize time)
gt["best_by_cost"]     # expected single-objective answer (minimize cost)
gt["pareto_front"]     # expected multi-objective answer (non-dominated routes)
```

### Module-level setup
- `_build_adjacency_list(edges) -> dict[str, list[(neighbor, time, cost)]]` — converts the
  flat `EDGES` list into an adjacency list keyed by source node, for backtracking traversal.
- `_ADJACENCY = _build_adjacency_list(EDGES)` — built once at import time from
  `tensor_builder`'s `NODES`/`EDGES`. (Tests that need a different/randomized graph
  temporarily reassign `gtg.NODES`, `gtg.EDGES`, and rebuild `gtg._ADJACENCY` before calling
  the generator, then restore the originals afterward.)

### Section 1 — Path enumeration
`enumerate_all_simple_paths(source, destination) -> list[dict]`
Finds **every simple path** from `source` to `destination` using classic recursive
backtracking: a `visited` set and a `current_path` list are maintained across recursive
calls; whenever `destination` is reached, a snapshot `{"path", "time", "cost"}` is appended
to `found_routes`; the last node is then popped/unvisited on the way back up so every other
branch can still be explored. Raises `ValueError` if `source`/`destination` aren't valid
nodes.

### Section 2 — Single-objective ground truth
`brute_force_best_by_objective(routes, objective) -> dict`
Given the list of all enumerated routes, does a simple linear scan to find the one with the
minimum `"time"` or minimum `"cost"` (whichever `objective` string is passed in). This is the
expected answer that the manual `dijkstra()` implementation should reproduce exactly.

### Section 3 — Pareto front (brute-force dominance filtering)
- `_route_is_dominated_by(candidate, other) -> bool` — dictionary-based equivalent of
  `pareto_search.dominates()`: `other` dominates `candidate` if `other` is no worse in both
  time and cost, and strictly better in at least one.
- `brute_force_pareto_front(routes) -> list[dict]` — for every candidate, checks whether
  *any* other route dominates it; keeps only the non-dominated ones and sorts the survivors
  by `(time, cost)`. This is the independent, from-scratch equivalent of
  `pareto_search.pareto_filter()`.

### Section 4 — Main entry point
`generate_ground_truth(source, destination) -> dict`
Runs sections 1–3 in sequence and returns `source`, `destination`, `all_candidates`,
`best_by_time`, `best_by_cost`, and `pareto_front` — everything needed to check both the
single-objective and multi-objective outputs of the "real" algorithm modules.

### Section 5 — Comparison helpers
- `paths_match(path_a, path_b) -> bool` — plain list equality between two node-label
  sequences.
- `route_sets_match(routes_a, routes_b, tolerance=1e-9) -> bool` — compares two lists of
  route dictionaries as **unordered sets**, using a rounded `(path, time, cost)` tuple as the
  comparison key so floating-point rounding doesn't cause a false mismatch. This is what the
  test suite uses to check "does `pareto_search()`'s candidate/Pareto-front list exactly
  equal the brute-force one?" regardless of the order either list was produced in.

### Section 6 — Standalone demo
Running `python ground_truth_generator.py` directly prints the brute-force ground truth for
`A -> J` on the base graph: total simple paths found, the best-by-time and best-by-cost
routes, and the full Pareto front.

**Why brute force is acceptable here (and only here):** the number of simple paths grows
very quickly with graph size, so this approach is not efficient for large graphs — but for
the small graphs used in this project it is entirely feasible, and its only job is to
generate correctness labels for testing, not to solve real instances.

---

## 5. Test Suite (notebook "Testing" section)

The automated test suite (in the notebook, easily extractable into a `pytest` file) ties
everything together. On each run it:

1. Builds (or reuses) a **randomized** test graph via `randomize_test_graph()` — up to 22
   nodes, randomized time (1–10h) and cost ($30–$200) ranges, guaranteed single start (`A`)
   and end node, and no unintended dead ends.
2. Builds the adjacency matrices/tensor for that random graph
   (`tensor_builder.build_adjacency_matrices`, `build_tensor`).
3. Temporarily points `ground_truth_generator` at the same random graph and generates ground
   truth (`generate_ground_truth`).
4. Runs `pareto_search()` on the same random graph.
5. Executes five checks:

| Test | What it verifies |
|---|---|
| `test_matrix_and_tensor_construction` | Matrix/tensor shapes are correct; tensor slices match the source matrices; `INF` pattern and values match a matrix rebuilt independently from the edge list. |
| `test_dijkstra_output_vs_ground_truth` | A reference Dijkstra run (time and cost) matches the brute-force `best_by_time` / `best_by_cost` paths and totals **exactly**. |
| `test_route_reconstruction` | Reconstructed paths are non-empty, start at the source, and end at the destination. |
| `test_pareto_front_vs_ground_truth` | The candidate set and Pareto front produced by `pareto_search()` exactly match the brute-force enumeration (`route_sets_match`), and the fastest/cheapest routes on the front match ground truth. |
| `test_full_integration_output` | `display_pareto_results()`'s printed report actually contains the correct header, counts, and every expected route string. |

Each test is wrapped so a failure is recorded and reported by name rather than stopping the
whole run immediately, and the suite raises an `AssertionError` summarizing all failures (if
any) at the end — otherwise it prints `"All automated tests passed."`

Running this suite repeatedly (each run regenerates a new random graph) is what lets the
team claim the algorithms are correct in general, not just on the one hand-built example
graph.

---

## 6. How the Pieces Fit Together

```
                 ┌────────────────────┐
                 │  tensor_builder.py │
                 │  (matrices/tensor) │
                 └─────────┬──────────┘
                           │ time_matrix, cost_matrix, node_index
              ┌────────────┴─────────────┐
              ▼                          ▼
     dijkstra(matrix, ...)      pareto_search.py
   (single-objective route)   (multi-objective routes)
              │                          │
              ▼                          ▼
        one best path         Pareto-optimal route set
              │                          │
              └────────────┬─────────────┘
                            ▼
               ground_truth_generator.py
             (independent brute-force check)
                            │
                            ▼
                    automated test suite
                 (pass/fail correctness report)
```

This mirrors the pipeline described in the proposal: represent the network -> optimize for
one objective -> optimize for multiple objectives via Pareto dominance -> verify everything
against independently generated ground truth.
