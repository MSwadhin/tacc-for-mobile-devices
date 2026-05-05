from __future__ import annotations

import networkx as nx
import numpy as np


def generate_zipf_demand(
    graph: nx.Graph,
    catalog_size: int,
    alpha: float,
    locality_strength: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create node-content request probabilities with Zipf skew and spatial locality."""
    rng = np.random.default_rng(seed)
    n = graph.number_of_nodes()
    ranks = np.arange(1, catalog_size + 1, dtype=np.float32)
    global_pop = ranks ** (-alpha)
    global_pop = global_pop / global_pop.sum()

    communities = max(2, min(6, n // 3))
    node_groups = np.arange(n) % communities
    content_groups = rng.integers(0, communities, size=catalog_size)
    demand = np.zeros((n, catalog_size), dtype=np.float32)

    for i in range(n):
        locality = np.where(content_groups == node_groups[i], locality_strength, 1.0)
        noise = rng.lognormal(mean=0.0, sigma=0.20, size=catalog_size)
        row = global_pop * locality * noise
        demand[i] = row / row.sum()

    return demand, global_pop.astype(np.float32)

