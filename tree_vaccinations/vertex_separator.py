from typing import List, Tuple

import networkx as nx
import numpy as np

from spectral_analysis import spectral_radius


def minimize_spectral_radius(G: nx.Graph, k: int, epsilon: float = 1e-4) -> Tuple[float, List[int]]:
    """
    Minimizes the largest spectral radius among connected components in the tree G by removing up to k nodes.
    Returns the minimal spectral radius achievable and the list of nodes to remove to achieve it.

    Parameters:
    - G: A NetworkX undirected tree graph. Nodes should be labeled with consecutive integers starting from 0.
    - k: The maximum number of nodes to remove.
    - epsilon: Precision for the binary search on spectral radius.

    Returns:
    - A tuple containing:
        - The minimal largest spectral radius achievable.
        - A list of node labels to remove to achieve this spectral radius.
    """
    n = G.number_of_nodes()
    adj = {node: list(G.neighbors(node)) for node in G.nodes()}
    # root = next(iter(G.nodes()))  # Choose an arbitrary root

    root = np.random.choice(G.nodes())


    # Initialize removals array
    removals = np.zeros(n, dtype=int)

    def can_partition(lambda_val: float) -> bool:
        """
        Determines if it's possible to remove up to k nodes such that every connected component
        in the remaining tree has a spectral radius <= lambda_val.

        Parameters:
        - lambda_val: The spectral radius threshold to test.

        Returns:
        - True if feasible with <= k cuts, False otherwise.
        """
        nonlocal k
        total_removals = 0
        removals[:] = 0  # Reset removals

        def dfs(u: int, parent: int) -> List[int]:
            """
            Post-order DFS traversal to determine if the current subtree can be kept without exceeding lambda_val.
            If not, mark the current node for removal.

            Parameters:
            - u: Current node.
            - parent: Parent node.

            Returns:
            - A list of nodes in the current subtree if spectral radius <= lambda_val.
            - An empty list if the current node is removed.
            """
            nonlocal total_removals
            current_subtree = [u]
            for v in adj[u]:
                if v != parent:
                    child_subtree = dfs(v, u)
                    if child_subtree:  # If the child subtree is kept
                        current_subtree.extend(child_subtree)
            # Compute spectral radius of the current subtree
            rho = spectral_radius(G.subgraph(current_subtree))
            if rho <= lambda_val:
                return current_subtree  # Keep the subtree
            else:
                # Must remove the current node
                total_removals += 1
                removals[u] = 1
                return []  # Subtree is removed

        dfs(root, -1)
        return total_removals <= k

    # Compute initial spectral radius bounds
    overall_rho = spectral_radius(G)
    left, right = 0.0, overall_rho
    best_lambda = overall_rho
    best_removals = []

    # Binary search to find the minimal feasible spectral radius
    while right - left > epsilon:
        mid = (left + right) / 2
        if can_partition(mid):
            best_lambda = mid
            best_removals = removals.copy()
            right = mid  # Try to find a smaller lambda
        else:
            left = mid  # Need to increase lambda

    # After binary search, collect the nodes to remove
    nodes_to_remove = [node for node, removed in zip(G.nodes(), best_removals) if removed]

    return best_lambda, nodes_to_remove
