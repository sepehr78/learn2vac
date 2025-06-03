"""
Our Greedy algorithm:
Greedy vaccination strategy minimizing the spectral radius via heuristic removal.
"""
from typing import List

import networkx as nx

from greedy_alg.greedy_vac_opt import get_greedy_vaccinations_optimized
from vaccination_strats.vaccinator import Vaccinator


# this is the Greedy vaccinator (Algorithm 4)


class SpectralVaccinator(Vaccinator):
    """Vaccinator using a greedy heuristic to approximate spectral radius minimization.

    Implements Algorithm 4: at each round, remove nodes to maximally decrease the
    spectral radius of the inferred network.
    """
    def get_vaccination_list(self, current_infected) -> List[int]:
        """Select nodes to vaccinate by greedily minimizing post-removal spectral radius.

        Args:
            current_infected (Sequence[int]): Nodes currently infected.

        Returns:
            List[int]: Indices of nodes selected for vaccination.
        """
        if self.is_over_budget():
            return []

        # vaccinate the node such that after removal, the change in the spectral radius is minimized
        graph = self.learner.get_inferred_network()
        adj_mat = nx.to_numpy_array(graph)

        # Use the public greedy function
        vacc_nodes = get_greedy_vaccinations_optimized(adj_mat, self.budget)
        for node in vacc_nodes:
            self.vaccinated[node] = None
        return vacc_nodes
