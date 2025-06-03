class ExactFakeLearner:
    def __init__(self, n_nodes):
        self.graph = None

    def get_inferred_network(self):
        return self.graph

    def observe(self, previous_infected, current_infected):
        pass

