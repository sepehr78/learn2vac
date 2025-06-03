import itertools
import random

import networkx as nx
import numpy as np
from tqdm import tqdm

import pandas as pd
from matplotlib import pyplot as plt
from numpy.linalg import eigvalsh
import time
from typing import List, Tuple, Union
from numpy.linalg import eigvalsh
from numba import njit
import seaborn as sns
from spectral_analysis import spec_rad_from_adj_mat


# --- Original Function ---
def get_greedy_vaccinations(adj_mat, budget, return_final_rad=False) -> List[int]:
    num_nodes = len(adj_mat)
    vacc_nodes = []
    for _ in range(budget):
        spectral_rad_arr = np.zeros(num_nodes)
        for i in range(num_nodes):
            # create a copy without the ith node
            adj_mat_copy = np.copy(adj_mat)
            adj_mat_copy[i, :] = adj_mat_copy[:, i] = 0
            spectral_rad_arr[i] = spec_rad_from_adj_mat(adj_mat_copy)
        mind_idx = np.argmin(spectral_rad_arr)
        adj_mat[mind_idx, :] = adj_mat[:, mind_idx] = 0
        vacc_nodes.append(mind_idx)

    if return_final_rad:
        return vacc_nodes, spec_rad_from_adj_mat(adj_mat)
    return vacc_nodes

# --- Optimized Function ---

def get_greedy_vaccinations_optimized(adj_mat_orig: np.ndarray, budget: int, return_final_rad: bool = False) -> Union[
    List[int], Tuple[List[int], float]]:
    """
    Optimized implementation. Finds nodes to vaccinate greedily.
    Does NOT modify the input adj_mat_orig.
    """
    num_nodes = adj_mat_orig.shape[0]

    budget = min(budget, num_nodes)
    if budget <= 0:
        if return_final_rad:
            # Calculate spec rad of the original if budget is 0
            return [], spec_rad_from_adj_mat(adj_mat_orig)
        else:
            return []

    # Keep track of original indices of nodes still in the graph
    current_nodes_idx = list(range(num_nodes))
    vacc_nodes = []

    for _ in range(budget):
        if not current_nodes_idx:
            break

        min_spec_rad_found = np.inf
        best_node_original_idx = -1
        index_to_remove_from_list = -1  # Index within current_nodes_idx

        # Extract the submatrix only once per budget iteration
        current_submatrix = adj_mat_orig[np.ix_(current_nodes_idx, current_nodes_idx)]

        # If only one node left, it must be the one to remove
        if len(current_nodes_idx) == 1:
            best_node_original_idx = current_nodes_idx[0]
            index_to_remove_from_list = 0
            # min_spec_rad_found remains inf, doesn't matter as choice is forced
        else:
            # Iterate through the nodes *currently* in the graph
            # 'i' is the index within the *current submatrix*
            for i in range(len(current_nodes_idx)):
                node_to_test_original_idx = current_nodes_idx[i]

                # Create the temporary submatrix *without* the i-th node/row/col
                # np.delete creates a copy, but of a *smaller* matrix now
                temp_sub_mat = np.delete(np.delete(current_submatrix, i, axis=0), i, axis=1)

                # Calculate spectral radius of the smaller temporary matrix
                current_rad = spec_rad_from_adj_mat(temp_sub_mat)

                # Use "<" for comparison. If ties occur, the first one found wins.
                # This matches np.argmin's behavior in the original logic (implicitly)
                if current_rad < min_spec_rad_found:
                    min_spec_rad_found = current_rad
                    best_node_original_idx = node_to_test_original_idx
                    index_to_remove_from_list = i

        # Vaccinate the chosen node
        if best_node_original_idx != -1:
            vacc_nodes.append(best_node_original_idx)
            # Remove the node from our list of active nodes
            del current_nodes_idx[index_to_remove_from_list]
        else:
            # Safeguard, see explanation in previous response
            print("Warning: Could not find a best node to remove in optimized.")
            break

    if return_final_rad:
        if not current_nodes_idx:
            final_rad = 0.0
        else:
            # Calculate final radius from the remaining nodes in the original matrix
            final_submatrix = adj_mat_orig[np.ix_(current_nodes_idx, current_nodes_idx)]
            final_rad = spec_rad_from_adj_mat(final_submatrix)
        return vacc_nodes, final_rad
    else:
        return vacc_nodes
