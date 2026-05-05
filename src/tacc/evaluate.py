from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from .graph import shortest_latency_matrix


@dataclass(frozen=True)
class CostConfig:
    local_latency: float = 1.0
    origin_latency: float = 35.0
    replication_lambda: float = 0.02
    redundancy_beta: float = 18.0
    relocation_mu: float = 0.04


def align_placement(placement: np.ndarray, all_nodes: list[str], active_nodes: list[str]) -> np.ndarray:
    index = {node: i for i, node in enumerate(all_nodes)}
    return np.stack([placement[index[node]] for node in active_nodes], axis=0)


def evaluate_placement(
    graph: nx.Graph,
    all_nodes: list[str],
    placement: np.ndarray,
    demand: np.ndarray,
    initial: np.ndarray,
    cfg: CostConfig,
) -> dict[str, float]:
    active_nodes = [node for node in all_nodes if node in graph.nodes]
    if not active_nodes:
        return {
            "objective": cfg.origin_latency,
            "access_cost": cfg.origin_latency,
            "hit_ratio": 0.0,
            "local_hit_ratio": 0.0,
            "cooperative_hit_ratio": 0.0,
            "origin_miss_ratio": 1.0,
            "replication_cost": 0.0,
            "redundancy_cost": 0.0,
            "redundancy_ratio": 0.0,
            "relocation_cost": 0.0,
        }

    active_idx = [all_nodes.index(node) for node in active_nodes]
    x = placement[active_idx]
    x0 = initial[active_idx]
    p = demand[active_idx]
    latencies = shortest_latency_matrix(graph, active_nodes, cfg.origin_latency)

    access_cost = 0.0
    local_hit = 0.0
    cooperative_hit = 0.0
    origin_miss = 0.0
    for i in range(len(active_nodes)):
        for c in range(x.shape[1]):
            prob = float(p[i, c])
            if x[i, c]:
                access_cost += prob * cfg.local_latency
                local_hit += prob
                continue

            holders = np.flatnonzero(x[:, c])
            holders = holders[holders != i]
            best = cfg.origin_latency
            if holders.size:
                best = float(np.min(latencies[i, holders]))
            access_cost += prob * best
            if best < cfg.origin_latency:
                cooperative_hit += prob
            else:
                origin_miss += prob

    total_replicas = float(x.sum())
    replication_cost = total_replicas

    redundancy_cost = 0.0
    edge_weight = 0.0
    active_pos = {node: pos for pos, node in enumerate(active_nodes)}
    for u, v, data in graph.edges(data=True):
        if u not in active_pos or v not in active_pos:
            continue
        w = float(data.get("normalized_weight", data.get("weight", 1.0)))
        edge_weight += w
        redundancy_cost += w * float(np.logical_and(x[active_pos[u]], x[active_pos[v]]).sum())

    max_redundancy = max(edge_weight * x.shape[1], 1.0)
    redundancy_ratio = redundancy_cost / max_redundancy
    relocation_cost = float(np.logical_xor(x, x0).sum())

    objective = (
        access_cost
        + cfg.replication_lambda * replication_cost
        + cfg.redundancy_beta * redundancy_cost
        + cfg.relocation_mu * relocation_cost
    )
    total_requests = max(float(len(active_nodes)), 1.0)
    hit_ratio = min(max((local_hit + cooperative_hit) / total_requests, 0.0), 1.0)
    local_hit_ratio = min(max(local_hit / total_requests, 0.0), 1.0)
    cooperative_hit_ratio = min(max(cooperative_hit / total_requests, 0.0), 1.0)
    origin_miss_ratio = min(max(origin_miss / total_requests, 0.0), 1.0)
    return {
        "objective": objective / total_requests,
        "access_cost": access_cost / total_requests,
        "hit_ratio": hit_ratio,
        "local_hit_ratio": local_hit_ratio,
        "cooperative_hit_ratio": cooperative_hit_ratio,
        "origin_miss_ratio": origin_miss_ratio,
        "replication_cost": replication_cost / total_requests,
        "redundancy_cost": redundancy_cost / total_requests,
        "redundancy_ratio": redundancy_ratio,
        "relocation_cost": relocation_cost / total_requests,
    }


def average_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = rows[0].keys()
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}
