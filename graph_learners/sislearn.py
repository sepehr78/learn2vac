import networkx as nx
import numpy as np

"""
Code for SISLearn, which learns the underlying SIS network from infection history.
"""


def compute_estimate(node, nodes_to_check, inf_hist, other_node_prev):
    # Vectorized computation of mu estimates
    # Compute current states: Y(node_u) == 0 & Y(node_i) == other_node_prev for all nodes_to_check

    if other_node_prev:
        current = (~inf_hist[:-1, node:node + 1] & inf_hist[:-1, nodes_to_check])
    else:
        current = (~inf_hist[:-1, node:node + 1] & ~inf_hist[:-1, nodes_to_check])

    # Compute next states: Y(node_u) in the next round
    next_round = inf_hist[1:, node:node + 1]

    # Compute mu estimates for all nodes_to_check
    num_cond = current.sum(axis=0)
    num_new_cond = (current & next_round).sum(axis=0)
    mu_estimates = np.divide(num_new_cond, num_cond, out=np.zeros_like(num_new_cond, dtype=float), where=num_cond != 0)
    return mu_estimates


def compute_nu_estimate(node_u, node_i, u_neighbor_superset_bool, inf_hist):
    u_neighbor_superset_bool = u_neighbor_superset_bool.copy()
    cond1 = (inf_hist[:, node_u] == 0) & (inf_hist[:, node_i] == 1)
    cond1[-1] = False  # since we cant observe the next round, we remove the last round
    num_cond1 = np.sum(cond1)

    cond2 = (inf_hist[:, node_u] == 0) & (inf_hist[:, node_i] == 0)
    cond2[-1] = False
    num_cond2 = np.sum(cond2)

    # determine unique Y_s
    # first exclude node_i from S
    u_neighbor_superset_bool[node_i] = False
    unique_ys_arr = np.unique(inf_hist[:, u_neighbor_superset_bool], axis=0)

    nu_est = 0
    for unique_ys in unique_ys_arr:
        cond_ys = np.all(inf_hist[:, u_neighbor_superset_bool] == unique_ys, axis=1)
        cond_ys[-1] = False # since we cant observe the next round, we remove the last round
        num_cond_ys = np.sum(cond_ys)

        cond1_ys = cond1 & cond_ys
        new_cond_1 = cond1_ys & np.roll(inf_hist[:, node_u], -1)
        num_new_cond1 = np.sum(new_cond_1)
        cond1_est = num_new_cond1 / num_cond1 if num_cond1 > 0 else 0

        cond2_ys = cond2 & cond_ys
        new_cond_2 = cond2_ys & np.roll(inf_hist[:, node_u], -1)
        num_new_cond2 = np.sum(new_cond_2)
        cond2_est = num_new_cond2 / num_cond2 if num_cond2 > 0 else 0

        nu_est_s = cond1_est - cond2_est
        ys_prob = num_cond_ys / len(cond_ys)
        nu_est += nu_est_s * ys_prob

    return nu_est



def compute_nu_estimate_shortcut(node_u, node_i, u_neighbor_superset_bool, inf_hist):
    # Exclude the last time step to avoid rolling
    effective_inf_hist = inf_hist[:-1]
    next_inf_hist = inf_hist[1:]

    # Compute cond1 and cond2
    cond1 = (~effective_inf_hist[:, node_u]) & effective_inf_hist[:, node_i]
    cond2 = (~effective_inf_hist[:, node_u]) & (~effective_inf_hist[:, node_i])

    # Compute the number of conditions
    num_cond1 = cond1.sum()
    num_cond2 = cond2.sum()

    # Compute the state of node_u in the next time step
    next_round = next_inf_hist[:, node_u]

    # Compute new conditions without using np.roll
    new_cond_1 = cond1 & next_round
    new_cond_2 = cond2 & next_round

    # Sum the new conditions
    num_new_cond1 = new_cond_1.sum()
    num_new_cond2 = new_cond_2.sum()

    # Calculate estimates, handling division by zero
    cond1_est = num_new_cond1 / num_cond1 if num_cond1 > 0 else 0.0
    cond2_est = num_new_cond2 / num_cond2 if num_cond2 > 0 else 0.0

    # Calculate and return the nu estimate
    nu_estimate = cond1_est - cond2_est

    return nu_estimate


def learn_neighbors(node, node_list, inf_hist, mu_thresh, nu_thresh, known_neighbors_bool=None, with_pruning=True):
    if known_neighbors_bool is None:
        known_neighbors_bool = np.zeros(len(node_list), dtype=bool)

    # compute superset of neighbors
    # Initialize the superset with known neighbors
    u_neighbor_superset_bool = known_neighbors_bool.copy()

    # Identify nodes to check (excluding known neighbors and the node itself)
    mask = ~known_neighbors_bool & (node_list != node)
    nodes_to_check = node_list[mask]

    mu_estimates = compute_estimate(node, nodes_to_check, inf_hist, other_node_prev=True)

    # Update the superset based on mu_thresh
    pass_mu = mu_estimates >= mu_thresh
    u_neighbor_superset_bool[nodes_to_check] = pass_mu


    if with_pruning:
        mu_estimates_2 = compute_estimate(node, nodes_to_check, inf_hist, other_node_prev=False)
        nu_estimates = mu_estimates - mu_estimates_2

        # Use a standard deviation multiplier to determine the threshold (works better in practice)
        nu_thresh = nu_estimates.mean() + 2 * nu_estimates.std()

        fail_nu = nu_estimates >= nu_thresh
        u_neighbor_superset_bool[nodes_to_check] = u_neighbor_superset_bool[nodes_to_check] & fail_nu

    return u_neighbor_superset_bool


class CILearner:
    def __init__(self, n_nodes, max_num_rounds, mu_eps, nu_eps, p_inf, max_deg, with_pruning=True, max_num_rounds_used=None):
        self.inf_hist = np.ascontiguousarray(np.zeros((max_num_rounds, n_nodes), dtype=bool))
        self.n_nodes = n_nodes
        self.t = 0
        self.mu_eps = mu_eps
        self.nu_eps = nu_eps
        self.p_inf = p_inf
        self.max_deg = max_deg
        self.with_pruning = with_pruning
        self.max_num_rounds_used = max_num_rounds_used if max_num_rounds_used is not None else max_num_rounds

        # ensure that eps_nu is not too large
        if 0 < p_inf < 1:
            assert 2 * nu_eps < p_inf * (1 - p_inf) ** (max_deg - 1), f"nu_eps is {nu_eps} but should be less than {p_inf * (1 - p_inf) ** (max_deg - 1) / 2}"



    def observe(self, previous_infected, current_infected):
        if self.t >= self.inf_hist.shape[0]:
            raise ValueError("Too many rounds")

        # if set convert to list
        if isinstance(previous_infected, set):
            previous_infected = list(previous_infected)
        if isinstance(current_infected, set):
            current_infected = list(current_infected)

        if self.t == 0:
            self.inf_hist[self.t, previous_infected] = True
            self.t += 1
        self.inf_hist[self.t, current_infected] = True
        self.t += 1


    def get_inferred_network(self):
        # construct the graph using Algorithm LearnNeighbors node by node
        g = nx.Graph()
        g.add_nodes_from(range(self.n_nodes))

        mu_thresh = self.p_inf - self.mu_eps
        if 0 < self.p_inf < 1:
            nu_thresh = self.p_inf * (1 - self.p_inf) ** (self.max_deg - 1) - self.nu_eps
        else:
            nu_thresh = -self.nu_eps

        node_list = np.arange(self.n_nodes)

        # ignore the first 20% of inf_hist
        inf_hist = self.inf_hist[:self.max_num_rounds_used]
        # inf_hist = self.inf_hist[int(0.1 * self.inf_hist.shape[0]):]

        for node in range(self.n_nodes):
            known_neighbors = g.neighbors(node)
            known_neighbors_bool = np.zeros(self.n_nodes, dtype=bool)
            known_neighbors_bool[list(known_neighbors)] = True

            learned_neighbors = learn_neighbors(node, node_list, inf_hist, mu_thresh, nu_thresh,
                                                known_neighbors_bool, self.with_pruning)

            for neighbor in node_list[learned_neighbors]:
                g.add_edge(node, neighbor)

        return g

