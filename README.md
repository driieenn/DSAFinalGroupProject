# DSA Final Group Project
**Course:** PROG 2301 Data Structures and Algorithms\
**Instructor:** Prof. Jose Miguel Bautista\
**LT6**: Chloe Ganaden, Elijah Haduca, Adrien Maniquiz, Uriel Orpilla, Hillary So, Ethen Soriano, Hannah Trajano

---

## Overview
This project implements algorithms for finding optimal routes through a directed, dual-weighted network, where every edge carries two independent objective values, **time** and **cost**, instead of a single weight. The network is represented using adjacency matrices and a stacked tensor rather than a plain weighted graph, so that each edge stores a full objective vector.

On top of that representation, the project implements a **single-objective optimizer** (a hand-written Dijkstra-style shortest path algorithm that can minimize either time or cost) and a **multi-objective Pareto-front search** (which returns the full set of non-dominated time/cost trade-off routes instead of forcing everything into one score). Both optimized algorithms are compared against an independent **brute-force ground-truth generator**, which enumerates every possible route and is used *solely* for correctness testing, it is not part of the production algorithm.

The official entry point for running the project is **main.py** (an interactive CLI demo). The original exploratory work - network design, visualization, and early testing - now lives in two notebooks under notebooks: **visual_demo.ipynb** (a visual walkthrough of the network and both algorithms) and **networkx_heatmap_benchmark.ipynb** (a NetworkX-based benchmark of the project's algorithms plus a heatmap visualization). The finalized, documented, and tested versions of that same logic are what live in the .py modules described below, and those .py modules are what should actually be run and graded. Superseded/earlier drafts of notebooks and modules are kept in **archive/** for reference only and are not part of the graded project.


---

## Features

- Directed network model with 10 nodes and 16 edges, each carrying a `[time, cost]` weight pair.
- Adjacency-matrix and tensor (`(2, N, N)` NumPy array) representation of the network.
- Single-objective optimizer (Dijkstra-style) that finds the fastest or cheapest route on demand.
- Multi-objective Pareto-front search that returns every non-dominated time/cost trade-off route, with a printed report and an objective-space plot.
- Independent brute-force ground-truth generator for automated correctness testing, kept fully separate from the real algorithms.
- Interactive CLI demo (`main.py`) that ties everything together.
- Automated `pytest` suite that cross-checks both algorithms against the brute-force oracle on the fixed graph and on several randomized graphs.

---

## Project Structure

```text
.
├── README.md                   # this file
├── CODE_DOCUMENTATION.md       # detailed module/function reference
├── tensor_builder.py           # graph data, adjacency matrices, tensor
├── dijkstra_search.py          # single-objective optimizer (MAIN algorithm)
├── pareto_search.py            # multi-objective Pareto-front search (MAIN algorithm)
├── ground_truth_generator.py   # TESTING UTILITY ONLY: independent brute-force checker
├── main.py                     # Official entry point — interactive demo runner (CLI)
├── notebooks/
│   ├── visual_demo.ipynb                    # Visual demo of the network and both algorithms
│   └── networkx_heatmap_benchmark.ipynb     # NetworkX benchmark + heatmap visualization
├── tests/
│   └── test_project.py         # pytest suite (unit tests + ground-truth cross-checks)
└── archive/                    # Old/superseded files only 
```

| File | Purpose |
|---|---|
| `tensor_builder.py` | Builds the network's time matrix, cost matrix, and stacked tensor. Pure data structure code, no routing logic. |
| `dijkstra_search.py` | **Main algorithm.** Hand-implemented Dijkstra's algorithm that finds the best route for one objective (time or cost) at a time. |
| `pareto_search.py` | **Main algorithm.** Generates all candidate routes and filters them down to the Pareto-optimal (non-dominated) set. |
| `ground_truth_generator.py` | **Testing utility, not part of the main algorithm.** Brute-force enumerates every simple path and computes the "obviously correct" answer, used only to check the two modules above. |
| `main.py` | The official entry point. Interactive CLI demo that wires the modules together contains no algorithm logic of its own. |
| `notebooks/visual_demo.ipynb` | Visual demo notebook, walks through the network and both algorithms with plots. | 
| `notebooks/networkx_heatmap_benchmark.ipynb` | Benchmarks the project's algorithms against NetworkX and produces a heatmap visualization. | 
| `tests/test_project.py` | Automated `pytest` suite. |
| ` archive/` | Old files only, kept for reference and not graded. |



---

## Requirements

- Python 3.10+
- NumPy
- SciPy
- Matplotlib
- pandas
- networkx
- pytest

(pandas and networkx are used for the network visualization and benchmarking in notebooks/visual_demo.ipynb and notebooks/networkx_heatmap_benchmark.ipynb; they are included here so those notebooks still run if you open them.)


---

## Installation

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd <your-repo-folder>

# 2. (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

If you'd rather install manually without `requirements.txt`:

```bash
pip install numpy scipy matplotlib pandas networkx pytest
```

---

## Running the Program

The official entry point is `main.py`. Run it from the project root:

```bash
python main.py
```

On startup, `main.py` always builds the time matrix, cost matrix, and the stacked tensor, and prints the tensor's shape. From there, you'll be walked through:


1. **Show matrices?(y/n)** - optionally print the full time and cost matrices before continuing.
2. **Available nodes** - the list of all valid node labels `(A–J)` is printed for reference.
3. **Start node and destination node** - enter any two different labels from that list. If a node isn't recognized, or the start and destination are the same, the program prints an error and exits (it does not re-prompt. Re-run `python main.py` and try again).
4. **Optimization mode:**
    - 1 - fastest route (minimize time)
    - 2 - cheapest route (minimize cost
    - 3 - Pareto-front routes (time and cost together
    - 4 - run all three and compare

Example:
```bash
$ python main.py
============================================================
DSA FINAL GROUP PROJECT
Multi-Objective Network Optimization
============================================================

Tensor successfully built.
Tensor shape: (2, 10, 10)
tensor[0] = time matrix
tensor[1] = cost matrix

Show time and cost matrices first? (y/n): n

Available nodes:
A, B, C, D, E, F, G, H, I, J

Enter start node: A
Enter destination node: J

Choose optimization mode:
1 - Fastest route using time
2 - Cheapest route using cost
3 - Pareto-front routes using time and cost
4 - Run all and compare

Enter choice: 4
```

Every module can also be run on its own to see a smaller, standalone demo of just that piece:

```bash
python tensor_builder.py           # prints the matrices and tensor shape
python dijkstra_search.py          # prints fastest/cheapest A -> J routes
python pareto_search.py            # prints full Pareto analysis + opens a plot
python ground_truth_generator.py   # prints the brute-force A -> J answer
```

`pareto_search.py`'s demo opens a matplotlib window. If you're running headless (e.g. over SSH with no display), set `MPLBACKEND=Agg` first:

```bash
MPLBACKEND=Agg python pareto_search.py
```

---

## Running Tests

The brute-force implementation in `ground_truth_generator.py` is **not part of the main algorithm**. It exists only to independently verify that the optimized algorithms (`dijkstra_search.py`, `pareto_search.py`) produce correct results, by enumerating every possible route and comparing.

Run the automated suite with `pytest`:

```bash
pytest -v
```

This checks:
- **Matrix/tensor construction** - correct shape, correct diagonal, correct edge placement, directedness.
- **`dijkstra_search.dijkstra`** - matches `ground_truth_generator`'s brute-force best-by-time / best-by-cost on the fixed graph, returns `(None, inf)` for unreachable pairs, and matches a second independent reference implementation across 5 randomized graphs.
- **`pareto_search.pareto_search`** - its candidate set and Pareto front exactly match the brute-force enumeration, on both the fixed graph and 3 randomized graphs, and no route in the returned front dominates another route in the same front.

You can also run the brute-force module by itself to see its standalone output for the fixed `A -> J` route:

```bash
python ground_truth_generator.py
```

---

## Demo

To reproduce what the project demonstrates:

1. **Build the tensors.** Run `python tensor_builder.py` to see the time matrix, cost matrix, and confirm the tensor shape is `(2, 10, 10)`.
2. **Run the single-objective optimizer.** Run `python dijkstra_search.py` to see the fastest and cheapest `A -> J` routes.
3. **Run the Pareto search.** Run `python pareto_search.py` to see the full multi-objective analysis (candidates, eliminated/dominated routes, and the final Pareto front), plus a plot of the objective space.
4. **Compare with brute force.** Run `python ground_truth_generator.py` and check that its `best_by_time`, `best_by_cost`, and `pareto_front` match steps 2 and 3.
5. **Observe it all together.** Run python `main.py`, choose mode 4 ("run all and compare"), and enter a valid start/destination pair from the printed node list (e.g. A and J). You'll see the fastest route, the cheapest route, and the Pareto front side by side. (Note: `main.py` exits immediately on an invalid node or matching start/destination rather than re-prompting, so double-check your entries.)
6. **Verify automatically instead of by eye.** Run `pytest -v` and confirm all tests pass.

---

## Modules

Short summaries below; see **`CODE_DOCUMENTATION.md`** for full function-by-function documentation.

- **`tensor_builder.py`** - Data structures for the network. Builds `NODES`/`EDGES` into a time matrix, cost matrix, and a `(2, N, N)` tensor.
- **`dijkstra_search.py`** - Main single-objective algorithm. A hand-implemented Dijkstra (distance tracker, parent tracker, visited set, priority queue) that works on either the time or cost matrix.
- **`pareto_search.py`** - Main multi-objective algorithm. Generates candidate routes, represents each as a `[time, cost]` vector via the `Route` class, and filters them by Pareto dominance.
- **`ground_truth_generator.py`** - Testing utility only. Independent brute-force enumeration and comparison, used as an oracle in `tests/test_project.py`. Never imported by the two modules above.
- **`main.py`** -  The official entry point. Interactive CLI demo built from `show_available_nodes`, `format_path`, `run_dijkstra_demo`, and `run_pareto_demo`, wiring the algorithm modules together with no algorithm logic of its own. Validates node input and exits with an error message rather than re-prompting.
- **`notebooks/visual_demo.ipynb`** - Visual demo notebook that walks through the network and both algorithms with plots and narrative explanation.
- **`notebooks/networkx_heatmap_benchmark.ipynb`** - Benchmarks the project's hand-written algorithms against NetworkX's built-in implementations and produces a heatmap visualization of the results.
- **`tests/test_project.py`** - Automated `pytest` suite.
- **`archive/`** - Old files only (earlier drafts of notebooks and modules). Not run or graded as part of the project.

---

## Contributors

| Name | Responsibility |
|---|---|
| Chloe Ganaden | Adjacency matrices & tensor representation |
| Elijah Haduca | Ground-truth generation & testing / Pareto search |
| Adrien Maniquis | Single-objective optimizer / Pareto search |
| Uriel Orpilla | Single-objective optimizer / Testing Suite |
| Hillary So | Network & graph design / Demo Runner |
| Ethan Soriano | Adjacency matrices & tensor representation |
| Hannah Trajano | Documentation / Report / Presentation |


---

## Notes and Limitations

- The network is a small, hand-designed simulation (10 nodes, 16 directed edges) chosen to create meaningful time/cost trade-offs. It is not sourced from real logistics data.
- Route enumeration in `ground_truth_generator.py` (and the candidate generation step in `pareto_search.py`) is brute-force over all simple paths. This is intentional for a small graph but does not scale to large networks.
- The Pareto-front implementation is a simplified version of ideas from NAMOA*/BOA*-style multi-objective search literature; it does not claim to reproduce those algorithms at scale.
