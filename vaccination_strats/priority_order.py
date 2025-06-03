import contextlib
import io
import itertools
import os

import networkx as nx
import pulp
import time


def compute_cutwidth_ilp(graph: nx.Graph):
    """
    Computes the cutwidth of an undirected graph using an ILP formulation.

    Parameters:
        graph (networkx.Graph): An undirected graph for which to compute the cutwidth.

    Returns:
        priority_order (list): A list of nodes representing the priority order that achieves the minimal cutwidth.
        cutwidth (int): The minimal cutwidth of the graph.
    """
    # Number of nodes
    n = len(graph.nodes)
    nodes = list(graph.nodes)
    edges = list(graph.edges)

    # Create the ILP problem
    prob = pulp.LpProblem('Cutwidth', pulp.LpMinimize)

    # -------------------------------
    # Variables
    # -------------------------------

    # Binary variables x_{v,c}: 1 if node v is assigned to position c
    x = pulp.LpVariable.dicts(
        'x',
        ((v, c) for v in nodes for c in range(1, n + 1)),
        cat='Binary'
    )

    # Binary variables y_{e,c}: 1 if edge e crosses cut c
    y = pulp.LpVariable.dicts(
        'y',
        ((e, c) for e in edges for c in range(1, n + 1)),
        cat='Binary'
    )

    # Auxiliary variables b_{v,c}: indicates if node v is before cut c
    b = pulp.LpVariable.dicts(
        'b',
        ((v, c) for v in nodes for c in range(1, n + 1)),
        cat='Binary'
    )

    # Integer variable W: the maximum number of edge crossings across all cuts
    W = pulp.LpVariable('W', lowBound=0, cat='Integer')

    # -------------------------------
    # Constraints
    # -------------------------------

    # 1. Unique Assignment Constraints

    # Each node is assigned to exactly one position
    for v in nodes:
        prob += (
            pulp.lpSum(x[(v, c)] for c in range(1, n + 1)) == 1,
            f"UniquePosition_{v}"
        )

    # Each position is occupied by exactly one node
    for c in range(1, n + 1):
        prob += (
            pulp.lpSum(x[(v, c)] for v in nodes) == 1,
            f"UniqueNode_{c}"
        )

    # 2. Define b_{v,c} as the sum of x_{v,k} for k < c
    for v in nodes:
        for c in range(1, n + 1):
            prob += (
                b[(v, c)] == pulp.lpSum(x[(v, k)] for k in range(1, c)),
                f"b_def_{v}_{c}"
            )

    # 3. Edge Crossing Logic Constraints
    for e in edges:
        u, v = e
        for c in range(1, n + 1):
            # y_{e,c} <= b_{u,c} + b_{v,c}
            prob += (
                y[(e, c)] <= b[(u, c)] + b[(v, c)],
                f"y_leq_bu_bv_{e}_{c}"
            )

            # y_{e,c} <= 2 - (b_{u,c} + b_{v,c})
            prob += (
                y[(e, c)] <= 2 - (b[(u, c)] + b[(v, c)]),
                f"y_leq_2_bu_bv_{e}_{c}"
            )

            # y_{e,c} >= b_{u,c} - b_{v,c}
            prob += (
                y[(e, c)] >= b[(u, c)] - b[(v, c)],
                f"y_geq_bu_bv_{e}_{c}"
            )

            # y_{e,c} >= b_{v,c} - b_{u,c}
            prob += (
                y[(e, c)] >= b[(v, c)] - b[(u, c)],
                f"y_geq_bv_bu_{e}_{c}"
            )

    # 4. Max Cut Constraints
    for c in range(1, n + 1):
        prob += (
            pulp.lpSum(y[(e, c)] for e in edges) <= W,
            f"MaxCut_{c}"
        )

    # -------------------------------
    # Objective Function
    # -------------------------------
    prob += W, "Minimize_W"

    # -------------------------------
    # Solver Configuration
    # -------------------------------
    # Note: Ensure that GUROBI is installed and properly licensed.
    # The variable 'use_all_threads_gurobi' should be defined in your environment.
    # If not defined, you can set it to True or False as needed.

    # Example:
    # use_all_threads_gurobi = True

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

    # Extract the priority order based on the x variables
    priority_order = sorted(
        [(v, c) for v in nodes for c in range(1, n + 1) if pulp.value(x[(v, c)]) == 1],
        key=lambda item: item[1]
    )
    priority_order = [v for v, c in priority_order]

    # Extract the cutwidth value
    cutwidth = int(pulp.value(W))

    # Find the cut position c where the cutwidth occurs
    cut_position = None
    for c in range(1, n + 1):
        cut_value = sum(pulp.value(y[(e, c)]) for e in edges)
        if int(cut_value) == cutwidth:
            cut_position = c - 2  # Adjust for 0-based indexing
            break  # Stop at the first occurrence

    return priority_order, cutwidth, cut_position



def compute_cutwidth_exhaustive(graph):
    """
    Computes the cutwidth of an undirected graph using exhaustive enumeration.

    Parameters:
        graph (networkx.Graph): An undirected graph for which to compute the cutwidth.

    Returns:
        min_priority_order (list): A list of nodes representing the priority order that achieves the minimal cutwidth.
        min_cutwidth (int): The minimal cutwidth of the graph.
    """
    V = list(graph.nodes)
    E = list(graph.edges)
    N = len(V)

    min_cutwidth = float('inf')
    min_priority_order = None
    min_cut_position = None

    # Generate all possible permutations of the nodes
    for perm in itertools.permutations(V):
        max_cut = 0
        cut_position = None

        # For each position c, compute the cut C_c
        for c in range(1, N + 1):
            cut = 0
            # Nodes before cut c
            nodes_before_cut = set(perm[:c-1])
            # Nodes after cut c
            nodes_after_cut = set(perm[c-1:])
            # Count edges crossing the cut at position c
            for u, v in E:
                if (u in nodes_before_cut and v in nodes_after_cut) or (v in nodes_before_cut and u in nodes_after_cut):
                    cut += 1

            if cut > max_cut:
                max_cut = cut
                cut_position = c  # Update cut position where max_cut occurs

            # Early stopping if current max_cut exceeds known min_cutwidth
            if max_cut >= min_cutwidth:
                break
        if max_cut < min_cutwidth:
            min_cutwidth = max_cut
            min_priority_order = perm
            min_cut_position = cut_position - 2  # Adjust for 0-based indexing

    return list(min_priority_order), min_cutwidth, min_cut_position

def compute_cutwidth_ilp_new(graph, return_cutwidth=False):
    """
    Computes the cutwidth of an undirected graph using the new ILP formulation.

    Parameters:
        graph (networkx.Graph): An undirected graph for which to compute the cutwidth.

    Returns:
        priority_order (list): A list of nodes representing the priority order that achieves the minimal cutwidth.
        cutwidth (int): The minimal cutwidth of the graph.
        cut_positions (list): List of cut positions where the cutwidth occurs.
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



def run_single_solver(n, rep, G, graph_id, solver):
    """
    Runs a single solver on a given graph, measures the time, and records the cutwidth.

    Parameters:
        n (int): Number of nodes in the graph.
        rep (int): Repetition index.
        G (networkx.Graph): The graph to process.
        graph_id (int): Unique identifier for the graph.
        solver (dict): Solver information with 'name' and 'function'.

    Returns:
        dict: Result containing graph_id, n, rep, algorithm, time, and cutwidth.
    """
    algorithm_name = solver['name']
    compute_function = solver['function']

    start_time = time.process_time()
    priority_order, cutwidth, cut_positions = compute_function(G)
    end_time = time.process_time()

    time_taken = end_time - start_time

    return {
        'graph_id': graph_id,
        'n': n,
        'rep': rep,
        'algorithm': algorithm_name,
        'time': time_taken,
        'cutwidth': cutwidth
    }

