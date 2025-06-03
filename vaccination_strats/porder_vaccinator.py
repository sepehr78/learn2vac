from typing import List

from vaccination_strats.vaccinator import Vaccinator


class POrderVaccinator(Vaccinator):
    def __init__(self, budget: int, num_rounds: int, num_nodes: int, learner_class, porder_finding_function):
        super().__init__(budget, num_rounds, num_nodes, learner_class)
        self.compute_p_order = porder_finding_function


    def get_vaccination_list(self, current_infected) -> List[int]:
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

