"""Random vaccination strategy selecting nodes uniformly at random."""
from typing import List

import numpy as np

from vaccination_strats.vaccinator import Vaccinator


class RandomVaccinator(Vaccinator):
    """Vaccinator selecting random nodes for immunization."""
    def __init__(self, budget: int, num_rounds: int, num_nodes: int, learner_class):
        """Initialize a RandomVaccinator.

        Args:
            budget (int): Total number of vaccinations allowed.
            num_rounds (int): Number of vaccination rounds.
            num_nodes (int): Number of nodes in the network.
            learner_class (Type): Learner class for inferring the network.
        """
        super().__init__(budget, num_rounds, num_nodes, learner_class)


    def get_vaccination_list(self, current_infected) -> List[int]:
        """Select random unvaccinated nodes.

        Args:
            current_infected (Sequence[int]): Nodes currently infected.

        Returns:
            List[int]: Randomly selected nodes for vaccination.
        """
        if self.is_over_budget():
            return []

        # vaccinate a random node that is not already vaccinated
        unvaccinated = set(range(self.num_nodes)) - set(self.vaccinated)
        # unvaccinated = set(range(self.n))
        if len(unvaccinated) == 0:
            return []
        nodes = np.random.choice(list(unvaccinated), self.budget, replace=False)

        for node in nodes:
            self.vaccinated[node] = None
        return nodes

