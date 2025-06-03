from typing import List

import networkx as nx

from greedy_alg.greedy_vac_opt import get_greedy_vaccinations_optimized
from vaccination_strats.vaccinator import Vaccinator


# this is the Greedy vaccinator (Algorithm 4)


class SpectralVaccinator(Vaccinator):
    def get_vaccination_list(self, current_infected) -> List[int]:
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
