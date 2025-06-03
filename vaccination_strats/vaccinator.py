"""Module defining the Vaccinator base class for SIS vaccination strategies."""
from typing import List


class Vaccinator:
    """Base interface for vaccination strategies against an SIS model.

    Attributes:
        budget (int): Total number of vaccinations allowed.
        num_rounds (int): Number of vaccination rounds.
        num_nodes (int): Number of nodes in the network.
        vaccinated (dict): Mapping of vaccinated node to round indexed by node.
        learner: Graph learner instance used to infer the network topology.
    """
    def __init__(self, budget: int, num_rounds: int, num_nodes: int, learner_class):
        """Initialize a Vaccinator.

        Args:
            budget (int): Total number of vaccinations allowed.
            num_rounds (int): Number of vaccination rounds.
            num_nodes (int): Number of nodes in the network.
            learner_class (Type): Learner class for inferring the network topology.
        """
        self.budget = budget
        self.num_rounds = num_rounds
        self.num_nodes = num_nodes
        self.vaccinated = dict()
        self.learner = learner_class(num_nodes)

    def observe(self, previous_infected, current_infected):
        """Forward infection observations to the graph learner.

        Args:
            previous_infected (Sequence[int]): Nodes infected in the previous round.
            current_infected (Sequence[int]): Nodes infected in the current round.
        """
        self.learner.observe(previous_infected, current_infected)


    def get_vaccination_list(self, current_infected) -> List[int]:
        """Select nodes to vaccinate in the current round.

        Args:
            current_infected (Sequence[int]): Nodes currently infected.

        Returns:
            List[int]: Indices of nodes to vaccinate this round.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError()


    def is_over_budget(self):
        """Return True if the vaccination budget has been reached."""
        return len(self.vaccinated) >= self.budget



