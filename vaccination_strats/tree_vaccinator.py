"""Vaccination strategy for trees using vertex separator to minimize spectral radius."""
from typing import List

from tree_vaccinations.vertex_separator import minimize_spectral_radius
from vaccination_strats.vaccinator import Vaccinator


class TreeVaccinator(Vaccinator):
    """Vaccinator for tree graphs using vertex separator heuristic."""
    def get_vaccination_list(self, current_infected) -> List[int]:
        """Select nodes to remove based on spectral radius minimization on trees.

        Args:
            current_infected (Sequence[int]): Nodes currently infected.

        Returns:
            List[int]: Nodes selected for vaccination.
        """
        if self.is_over_budget():
            return []

        # vaccinate the k nodes to reduce spectral radius the most (assuming tree)
        graph = self.learner.get_inferred_network()
        _, nodes_to_remove = minimize_spectral_radius(graph, self.budget)

        for node in nodes_to_remove:
            self.vaccinated[node] = None

        return nodes_to_remove
