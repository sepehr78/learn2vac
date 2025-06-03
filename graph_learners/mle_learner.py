import networkx as nx
import numpy as np

# implementation of the MLE learner from Epidemiologic network inference (Barbillon et al. 2020)



class MLELearner:
    def __init__(self, n_nodes, max_num_rounds, p_inf, p_rec, thresh=0.5):
        self.inf_hist = np.zeros((max_num_rounds, n_nodes), dtype=bool)
        self.n_nodes = n_nodes
        self.t = 0
        self.p_inf = p_inf
        self.p_rec = p_rec
        self.thresh = thresh

    def compute_psi_ij(self, node_i, node_j, t):
        # assuming beta_ij (the prior) is 1
        Y_ti = self.inf_hist[t, node_i]
        Y_tj = self.inf_hist[t, node_j]
        Y_t1j = self.inf_hist[t + 1, node_j]

        e = self.p_rec
        c = self.p_inf


        if Y_tj == 1 and Y_ti == 1:
            if Y_t1j == 1:  # stays infected
                return 1-e
            else:  # recovers
                return e
        elif Y_tj == 1 and Y_ti == 0:
            if Y_t1j == 1:  # stays infected
                return 1-e
            else:  # recovers
                return e
        elif Y_tj == 0 and Y_ti == 1:
            if Y_t1j == 1:
                return c  # gets infected
            else:
                return 1-c  # stays healthy
        else:  # both are healthy
            if Y_t1j == 1:
                return 0  # gets infected (impossible)
            else:
                return 1  # stays healthy

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
    def compute_all_psi(self):
        """
        Vectorized computation of psi_ij,t for all i, j, t using nested np.where.

        Returns:
        - psi: NumPy array of shape (T-1, N, N)
        """
        T, N = self.inf_hist.shape
        e = self.p_rec
        c = self.p_inf

        # Extract Y_ti, Y_tj, Y_t1j
        Y_ti = self.inf_hist[:-1, :, np.newaxis]  # Shape: (T-1, N, 1)
        Y_tj = self.inf_hist[:-1, np.newaxis, :]  # Shape: (T-1, 1, N)
        Y_t1j = self.inf_hist[1:, np.newaxis, :]  # Shape: (T-1, 1, N)

        # Broadcast Y_ti and Y_tj to shape (T-1, N, N)
        Y_ti_broadcasted = Y_ti  # (T-1, N, 1) broadcasts to (T-1, N, N)
        Y_tj_broadcasted = Y_tj  # (T-1, 1, N) broadcasts to (T-1, N, N)
        Y_t1j_broadcasted = Y_t1j  # (T-1, 1, N) broadcasts to (T-1, N, N)

        # Compute psi using nested np.where to handle all cases
        psi = np.where(
            Y_tj_broadcasted == 1,
            np.where(
                Y_t1j_broadcasted == 1,
                1 - e,  # Stays infected
                e        # Recovers
            ),
            np.where(
                Y_ti_broadcasted == 1,
                np.where(
                    Y_t1j_broadcasted == 1,
                    c,      # Gets infected
                    1 - c   # Stays healthy
                ),
                np.where(
                    Y_t1j_broadcasted == 1,
                    0,      # Gets infected (impossible)
                    1       # Stays healthy
                )
            )
        )

        return psi  # Shape: (T-1, N, N)

    def get_inferred_network(self):
        """
        Infers the network based on infection history using vectorized operations.

        Returns:
        - g: A NetworkX graph with inferred edges.
        """
        g = nx.Graph()
        g.add_nodes_from(range(self.n_nodes))

        # Compute all psi values
        psi = self.compute_all_psi()  # Shape: (T-1, N, N)

        # Compute psi_jt = sum_k psi[k, j, t] over k for each j and t
        # Shape of psi_jt: (T-1, N)
        # Sum over axis=1 (node_i)
        psi_jt = psi.sum(axis=1)  # Shape: (T-1, N)

        # To prevent division by zero, replace zeros in psi_jt with ones temporarily
        # This ensures that psi / psi_jt will be zero where psi_jt was originally zero
        psi_jt_safe = np.where(psi_jt == 0, 1, psi_jt)  # Shape: (T-1, N)

        # Expand psi_jt_safe to (T-1, 1, N) for broadcasting
        psi_jt_safe_expanded = psi_jt_safe[:, np.newaxis, :]  # Shape: (T-1, 1, N)

        # Compute the ratio psi_ij,t / psi_jt for all i, j, t
        # Shape: (T-1, N, N)
        ratio = psi / psi_jt_safe_expanded

        # Compute (1 - ratio)
        one_minus_ratio = 1 - ratio

        # To compute the product over time, use logarithms to prevent numerical underflow
        # Handle cases where (1 - ratio) is zero by setting log(1 - ratio) to -inf
        with np.errstate(divide='ignore'):
            log_one_minus_ratio = np.log(one_minus_ratio)

        # Sum the log probabilities over time
        # Shape: (N, N)
        log_prob_not_edge = log_one_minus_ratio.sum(axis=0)

        # Convert back from log space
        prob_not_edge = np.exp(log_prob_not_edge)

        # Compute the probability of an edge existing
        prob_edge = 1 - prob_not_edge  # Shape: (N, N)

        # Extract upper triangular indices to avoid duplicate edges and self-loops
        rows, cols = np.triu_indices(self.n_nodes, k=1)

        # Apply the threshold to determine which edges to add
        mask = prob_edge[rows, cols] > self.thresh
        edges_to_add = zip(rows[mask], cols[mask])

        # Add the edges to the graph
        g.add_edges_from(edges_to_add)

        return g
