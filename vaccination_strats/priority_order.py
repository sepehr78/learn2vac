"""
Implementation of the priority order approach of
"Suppressing Epidemics in Networks Using Priority Planning" (Scaman et al. 2016)
"""

import contextlib
import io
import itertools
import os

import networkx as nx
import pulp
import time


def compute_cutwidth_ilp(graph, return_cutwidth=False):
    """Compute the cutwidth via a fast ILP formulation.

    Args:
        graph (nx.Graph): Undirected graph to analyze.
        return_cutwidth (bool): If True, also return cut_positions list.

    Returns:
        Tuple[List[int], int, List[int]]:
            priority_order: Node order achieving minimal cutwidth.
            cutwidth: Minimal cutwidth value.
            cut_positions: List of cut indices (0-based) where minimal cut occurs.
    """
    # Number of nodes
    N = len(graph.nodes)
    V = list(graph.nodes)
    E = list(graph.edges)

    # Create the ILP problem
    prob = pulp.LpProblem('Cutwidth_New_ILP', pulp.LpMinimize)

    # -------------------------------
    # Variables
    # -------------------------------

    # Binary variables x^k_v: 1 if node v is placed at position <= k
    x = pulp.LpVariable.dicts(
        'x',
        ((v, k) for v in V for k in range(N)),
        cat='Binary'
    )

    # Binary variables y^k_uv: 1 if edge uv crosses cut k
    y = pulp.LpVariable.dicts(
        'y',
        ((u, v, k) for u, v in E for k in range(N)),
        cat='Binary'
    )

    # Objective variable z
    z = pulp.LpVariable('z', lowBound=0, upBound=len(E), cat='Integer')

    # -------------------------------
    # Constraints
    # -------------------------------

    # 1. Cumulative Assignment Constraints
    for v in V:
        for k in range(1, N):
            # sum_{i=0}^{k-1} x^{i}_v <= k * x^{k}_v
            prob += (
                pulp.lpSum(x[(v, i)] for i in range(k)) <= k * x[(v, k)],
                f"CumulativeAssignment_{v}_{k}"
            )

    # 2. Initial Assignment Constraint
    # sum_{v in V} x^{0}_v = 1
    prob += (
        pulp.lpSum(x[(v, 0)] for v in V) == 1,
        "InitialAssignment"
    )

    # 3. Cumulative Node Count Constraint
    for k in range(N):
        # sum_{v in V} x^{k}_v = k + 1
        prob += (
            pulp.lpSum(x[(v, k)] for v in V) == k + 1,
            f"CumulativeNodeCount_{k}"
        )

    # 4 & 5. Edge Crossing Constraints
    for u, v in E:
        for k in range(N):
            # x^{k}_u - x^{k}_v <= y^{k}_{u,v}
            prob += (
                x[(u, k)] - x[(v, k)] <= y[(u, v, k)],
                f"EdgeCrossing1_{u}_{v}_{k}"
            )
            # x^{k}_v - x^{k}_u <= y^{k}_{u,v}
            prob += (
                x[(v, k)] - x[(u, k)] <= y[(u, v, k)],
                f"EdgeCrossing2_{u}_{v}_{k}"
            )

    # 6. Cutwidth Constraint
    for k in range(N):
        prob += (
            pulp.lpSum(y[(u, v, k)] for u, v in E) <= z,
            f"CutwidthConstraint_{k}"
        )


    # -------------------------------
    # Objective Function
    # -------------------------------
    prob += z, "Minimize_z"

    # -------------------------------
    # Solver Configuration
    # -------------------------------
    if "OMP_NUM_THREADS" in os.environ and not globals().get('use_all_threads_gurobi', False):
        solver = pulp.GUROBI(msg=False, Threads=int(os.environ["OMP_NUM_THREADS"]))
    else:
        solver = pulp.GUROBI(msg=False)

    # -------------------------------
    # Solve the ILP
    # -------------------------------
    with contextlib.redirect_stdout(io.StringIO()):  # Suppress solver output
        prob.solve(solver)

    # -------------------------------
    # Extract the Solution
    # -------------------------------

    # Build the priority order from x^{k}_v variables
    # Since x^{k}_v indicates whether v is among the first k+1 nodes,
    # we can determine the position of each node.

    node_positions = {}
    for v in V:
        for k in range(N):
            if pulp.value(x[(v, k)]) == 1:
                node_positions[v] = k
                break  # Found the earliest position where x^{k}_v = 1

    # Sort nodes by their positions to get the priority order
    priority_order = sorted(node_positions.items(), key=lambda item: item[1])
    priority_order = [v for v, k in priority_order]

    # Extract the cutwidth value
    cutwidth = int(pulp.value(z))

    # Find the cut position c where the cutwidth occurs
    cut_position = None
    for c in range(1, N + 1):
        cut_value = sum(pulp.value(y[(e[0], e[1], c)]) for e in E)
        if int(cut_value) == cutwidth:
            cut_position = c
            break  # Stop at the first occurrence

    if return_cutwidth:
        return priority_order, cutwidth, cut_position

    return priority_order

