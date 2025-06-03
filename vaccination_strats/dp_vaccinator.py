from typing import List

from dp_alg import nice_tree_decomp, conv_nx_nt_to_treenode, compute_treewidth_from_PACE
from dp_alg.tree_decomp_vacc import min_spec_rad_with_nt
from vaccination_strats.vaccinator import Vaccinator


# code for the optimal treewidth vaccinator that uses bottom-up dyanmic programming

class TWVaccinator(Vaccinator):
    def __init__(self, budget: int, num_rounds: int, num_nodes: int, learner_class, tw_lower=None, tw_upper=None):
        super().__init__(budget, num_rounds, num_nodes, learner_class)
        self.tw_lower = tw_lower
        self.tw_upper = tw_upper

    def get_vaccination_list(self, current_infected) -> List[int]:
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
