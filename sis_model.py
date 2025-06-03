import random

import networkx as nx
import numpy as np
from typing import Set, List

from spectral_analysis import spectral_radius


class SISNetwork:
    def __init__(self, graph, initial_seeds: List[int], infection_probability: float,
                 healing_probability: float):
        """Initialize the SIS Network model.

        Args:
            graph: The underlying directed graph representing the network.
            initial_seeds: List of nodes that are initially infected.
            infection_probability: Probability of infection spreading from an infected node to a susceptible neighbor.
            healing_probability: Probability of an infected node becoming susceptible (healing) in each time step.
        """
        self.graph = graph
        self.initial_seeds = initial_seeds
        self.infected: Set[int] = set(initial_seeds)
        self.infection_probability = infection_probability
        self.healing_probability = healing_probability
        self.time_step: int = 0
        self.num_nodes = len(graph)
        self.spectral_rad = None

        # use for vaccination
        self.heal_prob_scalar_dict = {}
        self.infect_prob_scalar_dict = {}

    def reset(self, sample_new_seed: bool = False):
        """Reset the SIS Network model to its initial state."""
        if sample_new_seed:
            self.infected = set(random.sample(list(self.graph.nodes()), len(self.initial_seeds)))
        else:
            self.infected = set(self.initial_seeds)
        self.time_step = 0
        self.heal_prob_scalar_dict = {}
        self.infect_prob_scalar_dict = {}


    def get_node_heal_prob(self, node: int) -> float:
        """Get the healing probability for a specific node.

        Args:
            node: The node to get the healing probability for.

        Returns:
            The healing probability for the node.
        """
        return np.clip(self.heal_prob_scalar_dict.get(node, 1.0) * self.healing_probability, 0, 1)

    def get_node_infect_prob(self, node: int) -> float:
        """Get the infection probability for a specific node.

        Args:
            node: The node to get the infection probability for.

        Returns:
            The infection probability for the node.
        """
        return np.clip(self.infect_prob_scalar_dict.get(node, 1.0) * self.infection_probability, 0, 1)


    def progress(self) -> None:
        """Progress the disease spread by one time step.

        In each time step:
            1. Attempt to heal each infected node.
            2. Attempt to infect susceptible nodes that have infected neighbors.
        """
        # Attempt to heal infected nodes
        healed = {node for node in self.infected if np.random.random() < self.get_node_heal_prob(node)}

        # Attempt to infect susceptible nodes
        new_infected = set()
        for node in self.graph.nodes():
            if node not in self.infected:
                if isinstance(self.graph, nx.DiGraph):
                    infected_neighbors = [n for n in self.graph.predecessors(node) if n in self.infected]
                else:
                    infected_neighbors = [n for n in self.graph.neighbors(node) if n in self.infected]
                if np.random.binomial(len(infected_neighbors), self.get_node_infect_prob(node)) > 0:
                    new_infected.add(node)

        self.infected -= healed
        self.infected.update(new_infected)
        self.time_step += 1

    def get_infected_count(self) -> int:
        """Get the current number of infected nodes.

        Returns:
            The number of infected nodes.
        """
        return len(self.infected)

    def get_susceptible_count(self) -> int:
        """Get the current number of susceptible nodes.

        Returns:
            The number of susceptible nodes.
        """
        return len(self.graph) - len(self.infected)

    def is_node_infected(self, node: int) -> bool:
        """Check if a specific node is infected.

        Args:
            node: The node to check.

        Returns:
            True if the node is infected, False otherwise.
        """
        return node in self.infected

    def get_infection_status(self) -> np.ndarray:
        """Get the infection status of all nodes in the network.

        Returns:
            numpy.ndarray: A binary array where 1 indicates that the node is infected and 0 indicates that the node is susceptible.
        """
        # return a list of 0s and 1s, where 1 indicates that the node is infected
        return np.asarray([1 if node in self.infected else 0 for node in self.graph.nodes()], dtype=bool)


    def vaccinate_node(self, node: int, healing_scalar=1, infection_scalar=1):
        """Vaccinate a specific node by scaling the healing and infection probabilities.

        Args:
            node: The node to vaccinate.
            healing_scalar: The scalar to multiply the healing probability by.
            infection_scalar: The scalar to multiply the infection probability by.
        """
        self.heal_prob_scalar_dict[node] = self.heal_prob_scalar_dict.get(node, 1.0) * healing_scalar
        self.infect_prob_scalar_dict[node] = self.infect_prob_scalar_dict.get(node, 1.0) * infection_scalar


    def vaccinate_nodes(self, nodes: List[int], healing_scalar=1, infection_scalar=1):
        """Vaccinate a list of nodes by scaling the healing and infection probabilities.

        Args:
            nodes: The list of nodes to vaccinate.
            healing_scalar: The scalar to multiply the healing probability by.
            infection_scalar: The scalar to multiply the infection probability by.
        """
        for node in nodes:
            self.vaccinate_node(node, healing_scalar, infection_scalar)


    def get_spectral_radius(self):
        """Get the spectral radius of the underlying graph.

        Returns:
            float: The spectral radius of the graph.
        """
        if self.spectral_rad is None:
            self.spectral_rad = spectral_radius(self.graph)
        return self.spectral_rad