from typing import List


class Vaccinator:
    def __init__(self, budget: int, num_rounds: int, num_nodes: int, learner_class):
        """
        Initialize the Vaccinator.

        Parameters
        ----------
        budget : int
            The budget for the vaccination strategy.
        num_rounds : int
            The number of rounds to run the vaccination strategy.
        num_nodes : int
            The number of nodes in the network.
        """
        self.budget = budget
        self.num_rounds = num_rounds
        self.num_nodes = num_nodes
        self.vaccinated = dict()
        self.learner = learner_class(num_nodes)

    def observe(self, previous_infected, current_infected):
        self.learner.observe(previous_infected, current_infected)


    def get_vaccination_list(self, current_infected) -> List[int]:
        """
        Get the list of nodes to vaccinate in the current round.
        Parameters
        ----------
        """
    
        raise NotImplementedError()


    def is_over_budget(self):
        """
        Check if the vaccination strategy is over budget.
        """
        return len(self.vaccinated) >= self.budget



