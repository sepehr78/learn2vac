import networkx as nx
import numpy as np
from scipy.linalg import eigvalsh


def spec_rad_from_adj_mat(A):
    """
    Computes the spectral radius of a symmetric matrix A.

    Parameters:
    A (np.ndarray): A symmetric NumPy array (adjacency matrix).

    Returns:
    float: The spectral radius (largest eigenvalue magnitude) of A.
           Returns 0.0 if the matrix is empty or has size 0x0.
    """
    if A.size == 0 or A.shape[0] == 0:
        return 0.0
    # eigvalsh computes eigenvalues of a Hermitian (or real symmetric) matrix
    eigenvalues = eigvalsh(A)

    if eigenvalues.size == 0:
        return 0.0
    # Max absolute eigenvalue is max(abs(min_eig), abs(max_eig))
    # Or more simply, max(abs(all_eigs))
    return np.max(np.abs(eigenvalues))


def spectral_radius(G):
    """
    Computes the spectral radius of an undirected NetworkX graph.

    Parameters:
    G (networkx.Graph): An undirected NetworkX graph.

    Returns:
    float: The spectral radius (largest eigenvalue) of the graph's adjacency matrix.
    """
    if not isinstance(G, nx.Graph):
        raise TypeError("Input must be a NetworkX Graph object.")

    if G.is_directed():
        raise ValueError("Graph must be undirected.")

    if len(G.nodes) <= 1:
        return 0  # assuming simple graph (no self-loops)

    # Get the adjacency matrix in sparse CSR format with float type
    A = nx.to_numpy_array(G)
    return spec_rad_from_adj_mat(A)
