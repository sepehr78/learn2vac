from collections import defaultdict

import networkx as nx
import numpy as np

from spectral_analysis import spectral_radius
from ..dp_alg import NodeType, TreeNode


def min_spec_rad_with_nt(G: nx.Graph, root: TreeNode, K: int, epsilon: float = 0.1, high=None,
                         verbose=False):
    # Compute initial spectral radius bounds
    overall_rho = spectral_radius(G)
    left = 1
    right = overall_rho if high is None else high
    best_lambda = overall_rho
    best_removals = []

    if verbose:
        print(f"Initial spectral radius: {overall_rho}")


    # Binary search to find the minimal feasible spectral radius
    while right - left > epsilon:
        mid = (left + right) / 2
        if verbose:
            print(f"Checking feasibility for lambda = {mid}")
        removals = spec_rad_feasibility_with_td_smart(G, root, K, mid)
        if removals is not None:
            best_lambda = mid
            best_removals = removals
            right = mid  # Try to find a smaller lambda
        else:
            left = mid  # Need to increase lambda

    if verbose:
        print(f"Best spectral radius found: {best_lambda}")
    return best_removals

def spec_rad_feasibility_with_td_smart(G: nx.Graph, root: TreeNode, K: int, lambda_threshold: float, efficient_mode=True):
    """
    Determines whether there exists a set of at most K node removals such that
    the spectral radius of the remaining graph is at most lambda_threshold.

    :param G: The input undirected NetworkX graph
    :param root: The root node of the nice tree decomposition
    :param K: Maximum number of node removals allowed
    :param lambda_threshold: Threshold for the spectral radius
    :return: True if such a removal set exists, False otherwise
    """

    num_join_nodes_left = -np.inf if efficient_mode else 0

    # Step 1: Precompute V_t for each node
    def compute_Vt(node):
        nonlocal num_join_nodes_left
        """
        Recursively computes V_t for each node and stores it in node.Vt
        """
        for child in node.children:
            compute_Vt(child)
        if not node.children:
            node.Vt = set(node.bag)
        elif node.type == 'join':
            num_join_nodes_left += 1
            node.Vt = set(node.bag)
            for child in node.children:
                node.Vt.update(child.Vt)
        else:
            child = node.children[0]
            node.Vt = set(node.bag).union(child.Vt)

    compute_Vt(root)

    # Step 2: Initialize DP table
    # DP[node][S][c] = set of removal sets R (as frozensets)
    DP = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))

    # Step 3: Post-order traversal to fill DP
    def post_order(node):
        nonlocal num_join_nodes_left
        for child in node.children:
            post_order(child)

        if node.type == NodeType.LEAF:
            # Leaf node: X_t = empty set
            DP[node][frozenset()][0].add(frozenset())

        elif node.type == NodeType.INTRODUCE:
            # Introduce node: X_t = X_child ∪ {v}
            v = node.vertex
            child = node.children[0]
            for S_child in DP[child]:
                for c in DP[child][S_child]:
                    # Case 1: v is not in S (i.e., can be removed)
                    # S remains the same
                    S_new = S_child
                    if c + 1 <= K:
                        for R_child in DP[child][S_child][c]:
                            R_new = set(R_child)
                            R_new.add(v)
                            DP[node][S_new][c + 1].add(frozenset(R_new))
                            if num_join_nodes_left <= 0:  # if no join nodes left, no need to add all possible removals
                                break

                    # Case 2: v is in S (i.e., cannot be removed)
                    S_new_with_v = set(S_child)
                    S_new_with_v.add(v)
                    S_new_with_v = frozenset(S_new_with_v)
                    for R_child in DP[child][S_child][c]:
                        # Need to check spectral radius
                        subgraph = G.subgraph(node.Vt - set(R_child))
                        rho = spectral_radius(subgraph)
                        if rho <= lambda_threshold:
                            DP[node][S_new_with_v][c].add(R_child)
                            if num_join_nodes_left <= 0:  # if no join nodes left, no need to add all possible removals
                                break

        elif node.type == NodeType.FORGET:
            # Forget node: X_t = X_child \ {w}
            w = node.vertex
            child = node.children[0]
            for S_child in DP[child]:
                if w in S_child:
                    S_with_w = S_child
                    S_without_w = frozenset(S_child - {w})
                else:
                    S_with_w = frozenset(S_child.union({w}))
                    S_without_w = S_child
                for c in DP[child][S_child]:
                    # union DP[Child][S][c] with DP[Child][S u {w}][c]
                    if S_without_w in DP[child] and c in DP[child][S_without_w]:
                        if False and num_join_nodes_left <= 0:
                            DP[node][S_without_w][c].add(next(iter(DP[child][S_without_w][c])))
                        else:
                            DP[node][S_without_w][c].update(DP[child][S_without_w][c])
                    if S_with_w in DP[child] and c in DP[child][S_with_w]:
                        if False and num_join_nodes_left <= 0:
                            DP[node][S_without_w][c].add(next(iter(DP[child][S_with_w][c])))
                        else:
                            DP[node][S_without_w][c].update(DP[child][S_with_w][c])

        elif node.type == NodeType.JOIN:
            num_join_nodes_left -= 1
            # Join node: X_t = X_t1 = X_t2
            child1, child2 = node.children
            common_S = set(DP[child1].keys()).intersection(DP[child2].keys())
            for S in common_S:
                for c1 in DP[child1][S]:
                    for c2 in DP[child2][S]:
                        if c1 > K or c2 > K:  # should never happen!
                            print("Unexpected stage reached!")
                            continue
                        found = False
                        for R1 in DP[child1][S][c1]:
                            for R2 in DP[child2][S][c2]:
                                R = set(R1).union(R2)
                                if len(R) > K:
                                    continue
                                # Ensure spectral radius condition
                                subgraph = G.subgraph(node.Vt - R)
                                rho = spectral_radius(subgraph)
                                if rho <= lambda_threshold:
                                    DP[node][S][len(R)].add(frozenset(R))
                                    if num_join_nodes_left <= 0:
                                        found = True
                                        break
                            if found:
                                break

        else:
            raise ValueError(f"Unknown node type: {node.type}")

    post_order(root)

    # Step 4: Check the DP at the root
    # At the root, S should be empty and c <= K
    for c in range(K + 1):
        if frozenset() in DP[root] and c in DP[root][frozenset()]:
            return DP[root][frozenset()][c].pop()
    return None

