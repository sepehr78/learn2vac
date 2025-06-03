from typing import List

from vaccination_strats.vaccinator import Vaccinator


class DegreeVaccinator(Vaccinator):
    def get_vaccination_list(self, current_infected) -> List[int]:
        if self.is_over_budget():
            return []

        # vaccinate the node with the highest degree that is not already vaccinated
        graph = self.learner.get_inferred_network()
        degrees = dict(graph.degree())
        nodes = sorted(degrees, key=degrees.get, reverse=True)
        for node in nodes:
            if node not in self.vaccinated:
            # if node in current_infected:
                self.vaccinated[node] = None
        return nodes[:self.budget]
