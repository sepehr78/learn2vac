# Structure Learning and Vaccination for Epidemic Control

Python code for the paper "Learn to Vaccinate: Combining Structure Learning and Effective Vaccination for Epidemic and Outbreak Control," accepted at ICML 2025.

## Abstract

The Susceptible-Infected-Susceptible (SIS) model is a widely used model for the spread of information and infectious diseases, particularly non-immunizing ones, on a graph. Given a highly contagious disease, a natural question is how to best vaccinate individuals to minimize the disease’s extinction time. While previous works showed that the problem of optimal vaccination is closely linked to the NP-hard _Spectral Radius Minimization_ (SRM) problem, they assumed that the graph is known, which is often not the case in practice. In this work, we consider the problem of minimizing the extinction time of an outbreak modeled by an SIS model where the graph on which the disease spreads is unknown and only the infection states of the vertices are observed. To this end, we split the problem into two: learning the graph and determining effective vaccination strategies. We propose a novel inclusion-exclusion-based learning algorithm and, unlike previous approaches, establish its sample complexity for graph recovery. We then detail an optimal algorithm for the SRM problem and prove that its running time is polynomial in the number of vertices for graphs with bounded treewidth. This is complemented by an efficient and effective polynomial-time greedy heuristic for any graph. Finally, we present experiments on synthetic and real-world data that numerically validate our learning and vaccination algorithms.

## Repository Structure

- **sis_model.py**: Discrete SIS simulation engine.  
- **spectral_analysis.py**: Utilities for spectral radius computation and analysis.  
- **vaccination_strats/**: Vaccination strategy implementations. Includes:
  - **dp_vaccinator.py** (exact DP-based algorithm, bounded treewidth optimal SRM)
  - **greedy_vaccinator.py** (fast spectral‐radius heuristic)
  - additional baselines (random, degree, walks, etc.).
- **dp_alg/**: Code for the DP vaccination algorithm (SageMath required).
- **greedy_alg/**: Code for the greedy vaccination algorithm.
- **graph_learners/**: Graph-learning algorithms from infection observations.
  - **sislearn.py**: Proposed inclusion-exclusion learning algorithm: SISLearn.
- **tree_vaccinations/**: Code for tree-only vaccination strategies.

## Installation

Install the required Python packages:

```bash
pip install -r requirements.txt
```

(Ensure SageMath is installed if using the DP-based vaccination code in `dp_alg/`.)
