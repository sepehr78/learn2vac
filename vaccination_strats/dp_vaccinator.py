"""
Our DP algorithm:
Dynamic programming vaccination strategy for bounded-treewidth graphs.
"""

from typing import List

from dp_alg import nice_tree_decomp, conv_nx_nt_to_treenode, compute_treewidth_from_PACE
from dp_alg.tree_decomp_vacc import min_spec_rad_with_nt
from vaccination_strats.vaccinator import Vaccinator


"""Code for the optimal treewidth vaccinator using bottom-up dynamic programming."""

class TWVaccinator(Vaccinator):
    """Vaccinator using exact dynamic programming on tree decompositions.

    For graphs with bounded treewidth, computes the optimal set of nodes
    whose removal minimizes the spectral radius in polynomial time.
    """
    def __init__(self, budget: int, num_rounds: int, num_nodes: int,
                 learner_class, tw_lower: int = None, tw_upper: int = None):
        """Initialize a TWVaccinator.

        Args:
            budget (int): Total number of vaccinations allowed.
            num_rounds (int): Number of vaccination rounds.
            num_nodes (int): Number of nodes in the network.
            learner_class (Type): Learner class for inferring the network.
            tw_lower (int, optional): Lower bound on treewidth. Defaults to None.
            tw_upper (int, optional): Upper bound on treewidth. Defaults to None.
        """
        super().__init__(budget, num_rounds, num_nodes, learner_class)
        self.tw_lower = tw_lower
        self.tw_upper = tw_upper

    def get_vaccination_list(self, current_infected) -> List[int]:
        """Select nodes by exact spectral radius minimization on graphs.

        Args:
            current_infected (Sequence[int]): Nodes currently infected.

        Returns:
            List[int]: Nodes selected for vaccination to optimally reduce spectral radius.
        """
        if self.is_over_budget():
            return []

        # vaccinate the k nodes to reduce spectral radius the most (assuming tree)
        graph = self.learner.get_inferred_network()

        if self.tw_lower is None or self.tw_upper is None:
            tw = compute_treewidth_from_PACE(graph)
            self.tw_upper = self.tw_lower = tw

        # construct the nice tree decomposition
        nice_td = nice_tree_decomp(graph, self.tw_lower, self.tw_upper)
        root_node = conv_nx_nt_to_treenode(nice_td)

        # run the dynamic programming algorithm on the nice tree decomposition
        nodes_to_remove = list(min_spec_rad_with_nt(graph, root_node, self.budget))

        for node in nodes_to_remove:
            self.vaccinated[node] = None

        return nodes_to_remove
