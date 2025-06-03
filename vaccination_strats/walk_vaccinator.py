import warnings
from typing import List
import networkx as nx
import numpy as np
from vaccination_strats.vaccinator import Vaccinator
warnings.simplefilter("error", RuntimeWarning)

class WalkVaccinator(Vaccinator):
    def __init__(self, budget: int, num_rounds: int, num_nodes: int, learner_class, k=None):
        """
        Initializes the ClosedWalkVaccinator.

        Args:
            learner: The learner object that provides the inferred network.
            budget: The vaccination budget.
            k (int): The length of closed walks to consider.
        """
        super().__init__(budget, num_rounds, num_nodes, learner_class)
        self.k = k if k is not None else np.ceil(np.log(num_nodes) / np.log(1 + 0.1)).astype(int)

    def get_vaccination_list(self, current_infected) -> List[int]:
        """
        Selects the node to vaccinate based on the number of closed walks of length k.

        Args:
            current_infected: A list of currently infected nodes.

        Returns:
            A list containing the index of the node to vaccinate.
        """
        if self.is_over_budget():
            return []

        # Retrieve the inferred network
        graph = self.learner.get_inferred_network()
        adj_mat = nx.to_numpy_array(graph)
        num_nodes = adj_mat.shape[0]

        # set columns and rows of vaccinated nodes to 0
        for node in self.vaccinated:
            adj_mat[node, :] = adj_mat[:, node] = 0

        adj_mat = adj_mat.astype(np.float64)

        # Compute A^k
        A_power_k = np.linalg.matrix_power(adj_mat, self.k)

        # Compute the total number of closed walks of length k
        trace_A_k = np.trace(A_power_k)

        walk_counts = np.zeros(num_nodes, dtype=np.float64)

        # Precompute a boolean mask array for each node removal
        # This avoids creating a new mask in every iteration
        indices = np.arange(num_nodes)

        for i in range(num_nodes):
            mask = indices != i
            A_minus_i = adj_mat[mask][:, mask]

            if A_minus_i.size == 0:
                # If the graph is empty after removal, there are no closed walks
                trace_A_minus_i_k = 0
            else:
                A_minus_i_power_k = np.linalg.matrix_power(A_minus_i, self.k)
                trace_A_minus_i_k = np.trace(A_minus_i_power_k)

            # Number of closed walks of length k that include node i
            walk_counts[i] = trace_A_k - trace_A_minus_i_k

        # Select the node with the maximum number of closed walks
        max_nodes = np.argsort(walk_counts)[-self.budget:]
        for node in max_nodes:
            self.vaccinated[node] = None
        return max_nodes