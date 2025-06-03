"""
Implementation of the priority order approach of
"Suppressing Epidemics in Networks Using Priority Planning" (Scaman et al. 2016)
"""

from typing import List

from vaccination_strats.priority_order import compute_cutwidth_ilp
from vaccination_strats.vaccinator import Vaccinator


class POrderVaccinator(Vaccinator):
    """Vaccinator using precomputed priority order for node removal.

    Applies a given priority-order function to infer node importance
    and selects top-budget nodes for vaccination.
    """
    def __init__(self, budget: int, num_rounds: int, num_nodes: int,
                 learner_class, porder_finding_function=compute_cutwidth_ilp):
        """Initialize a POrderVaccinator.

        Args:
            budget (int): Total number of vaccinations allowed.
            num_rounds (int): Number of vaccination rounds.
            num_nodes (int): Number of nodes in the network.
            learner_class (Type): Learner class for inferring the network.
            porder_finding_function (callable): Function that returns a priority order of nodes.
        """
        super().__init__(budget, num_rounds, num_nodes, learner_class)
        self.compute_p_order = porder_finding_function


    def get_vaccination_list(self, current_infected) -> List[int]:
        """Select nodes to vaccinate based on a precomputed priority order.

        Args:
            current_infected (Sequence[int]): Nodes currently infected.

        Returns:
            List[int]: Top nodes from the computed priority order.
        """
        if self.is_over_budget():
            return []

        # compute the priority order
        graph = self.learner.get_inferred_network()
        p_order = self.compute_p_order(graph)

        # vaccinate first node in the priority order that is not already vaccinated
        vacc_nodes = p_order[:self.budget]
        for node in vacc_nodes:
            if node not in self.vaccinated:
            # if node in current_infected:
                self.vaccinated[node] = None

        return vacc_nodes

