"""Dummy learner that returns a pre-set graph without performing inference."""

class ExactFakeLearner:
    """Dummy learner that returns a given graph without inference.

    Args:
        n_nodes (int): Number of nodes in the network (unused).

    Attributes:
        graph: Pre-set graph to return on get_inferred_network.
    """

    def __init__(self, n_nodes):
        """Initialize the dummy learner.

        Args:
            n_nodes (int): Number of nodes in the network (unused).
        """
        self.graph = None

    def get_inferred_network(self):
        """Return the pre-set graph without modifications.

        Returns:
            The graph previously assigned to self.graph.
        """
        return self.graph

    def observe(self, previous_infected, current_infected):
        """No-op for dummy learner (does not record observations)."""
        pass

