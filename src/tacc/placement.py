from __future__ import annotations

import networkx as nx
import numpy as np

from .evaluate import CostConfig, average_metrics, evaluate_placement
from .graph import perturb_graph


def empty_placement(n_nodes: int, catalog_size: int) -> np.ndarray:
    return np.zeros((n_nodes, catalog_size), dtype=bool)


def random_placement(n_nodes: int, catalog_size: int, capacity: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = empty_placement(n_nodes, catalog_size)
    for i in range(n_nodes):
        x[i, rng.choice(catalog_size, size=capacity, replace=False)] = True
    return x


def local_popularity_placement(demand: np.ndarray, capacity: int) -> np.ndarray:
    n_nodes, catalog_size = demand.shape
    x = empty_placement(n_nodes, catalog_size)
    for i in range(n_nodes):
        x[i, np.argsort(demand[i])[-capacity:]] = True
    return x


def global_popularity_placement(demand: np.ndarray, capacity: int) -> np.ndarray:
    n_nodes, catalog_size = demand.shape
    popular = np.argsort(demand.mean(axis=0))[-capacity:]
    x = empty_placement(n_nodes, catalog_size)
    x[:, popular] = True
    return x


def diversified_popularity_placement(demand: np.ndarray, capacity: int) -> np.ndarray:
    n_nodes, catalog_size = demand.shape
    ranked = np.argsort(demand.mean(axis=0))[::-1]
    x = empty_placement(n_nodes, catalog_size)
    for i in range(n_nodes):
        start = (i * capacity) % max(catalog_size - capacity + 1, 1)
        x[i, ranked[start : start + capacity]] = True
    return x


def _expected_objective(
    graph: nx.Graph,
    nodes: list[str],
    placement: np.ndarray,
    demand: np.ndarray,
    initial: np.ndarray,
    cost_cfg: CostConfig,
    perturb_rate: float,
    samples: int,
    rng: np.random.Generator,
) -> float:
    rows = []
    for _ in range(samples):
        g = perturb_graph(graph, rng, remove_node_rate=perturb_rate, remove_edge_rate=perturb_rate / 2.0)
        rows.append(evaluate_placement(g, nodes, placement, demand, initial, cost_cfg))
    return average_metrics(rows)["objective"]


def topology_greedy_placement(
    graph: nx.Graph,
    nodes: list[str],
    demand: np.ndarray,
    capacity: int,
    cost_cfg: CostConfig,
    perturb_rate: float,
    samples: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_nodes, catalog_size = demand.shape
    x = empty_placement(n_nodes, catalog_size)
    initial = x.copy()
    slots = n_nodes * capacity
    current = _expected_objective(graph, nodes, x, demand, initial, cost_cfg, perturb_rate, samples, rng)

    for _ in range(slots):
        best_gain = 0.0
        best_pair: tuple[int, int] | None = None
        for i in range(n_nodes):
            if int(x[i].sum()) >= capacity:
                continue
            for c in np.argsort(demand[i])[-min(catalog_size, 18) :]:
                if x[i, c]:
                    continue
                candidate = x.copy()
                candidate[i, c] = True
                score = _expected_objective(
                    graph, nodes, candidate, demand, initial, cost_cfg, perturb_rate, samples, rng
                )
                gain = current - score
                if gain > best_gain:
                    best_gain = gain
                    best_pair = (i, int(c))
        if best_pair is None:
            break
        x[best_pair] = True
        current -= best_gain
    return x

