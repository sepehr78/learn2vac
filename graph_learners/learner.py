class Learner:
    def __init__(self, n_nodes, mhs_solver=None):
        self.n_nodes = n_nodes
        self.infection_history = {node: [] for node in range(n_nodes)}
        self.solver = mhs_solver

    def observe(self, previous_infected, current_infected):
        for node in current_infected:
            if node not in previous_infected:
                self.infection_history[node].append(previous_infected.copy())

    def get_inferred_network(self):
        raise NotImplementedError()

