from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GraphBuildConfig:
    max_nodes: int = 24
    min_handoffs: int = 2


def build_handoff_graph(trace: pd.DataFrame, config: GraphBuildConfig) -> nx.Graph:
    """Build an undirected weighted AP graph from per-user AP association sequences."""
    events = trace[trace["ap"] != "OFF"].sort_values(["user_id", "timestamp"])
    activity = events["ap"].value_counts()
    selected = set(activity.head(config.max_nodes).index)

    handoffs: dict[tuple[str, str], int] = {}
    for _, group in events.groupby("user_id", sort=False):
        seq = [ap for ap in group["ap"].tolist() if ap in selected]
        for src, dst in zip(seq, seq[1:]):
            if src == dst:
                continue
            a, b = sorted((src, dst))
            handoffs[(a, b)] = handoffs.get((a, b), 0) + 1

    graph = nx.Graph()
    for ap in sorted(selected):
        graph.add_node(ap, activity=int(activity.get(ap, 0)))

    for (src, dst), count in handoffs.items():
        if count >= config.min_handoffs:
            graph.add_edge(src, dst, weight=float(count), latency_weight=1.0 / float(count))

    if not nx.is_connected(graph) and graph.number_of_nodes() > 1:
        # Keep the largest connected component to make cooperative retrieval meaningful.
        component = max(nx.connected_components(graph), key=len)
        graph = graph.subgraph(component).copy()

    total_weight = sum(data["weight"] for _, _, data in graph.edges(data=True)) or 1.0
    for _, _, data in graph.edges(data=True):
        data["normalized_weight"] = data["weight"] / total_weight

    return graph


def graph_features(graph: nx.Graph) -> np.ndarray:
    nodes = list(graph.nodes())
    degree = np.array([graph.degree(n) for n in nodes], dtype=np.float32)
    weighted_degree = np.array([graph.degree(n, weight="weight") for n in nodes], dtype=np.float32)
    centrality_map = nx.closeness_centrality(graph, distance="latency_weight") if graph.number_of_nodes() else {}
    centrality = np.array([centrality_map.get(n, 0.0) for n in nodes], dtype=np.float32)

    features = np.stack([degree, weighted_degree, centrality], axis=1)
    scale = np.maximum(features.max(axis=0, keepdims=True), 1.0)
    return features / scale


def perturb_graph(
    graph: nx.Graph,
    rng: np.random.Generator,
    remove_node_rate: float,
    remove_edge_rate: float,
) -> nx.Graph:
    g = graph.copy()
    nodes = list(g.nodes())
    if len(nodes) > 3 and remove_node_rate > 0:
        removable = [n for n in nodes if rng.random() < remove_node_rate]
        if len(removable) >= len(nodes) - 2:
            removable = removable[: max(0, len(nodes) - 3)]
        g.remove_nodes_from(removable)

    edges = list(g.edges())
    if remove_edge_rate > 0:
        g.remove_edges_from([edge for edge in edges if rng.random() < remove_edge_rate])

    if g.number_of_nodes() > 1 and not nx.is_connected(g):
        component = max(nx.connected_components(g), key=len)
        g = g.subgraph(component).copy()

    return g


def shortest_latency_matrix(graph: nx.Graph, nodes: list[str], origin_latency: float) -> np.ndarray:
    n = len(nodes)
    idx = {node: i for i, node in enumerate(nodes)}
    distances = np.full((n, n), origin_latency, dtype=np.float32)
    for i in range(n):
        distances[i, i] = 0.0

    paths = nx.all_pairs_dijkstra_path_length(graph, weight="latency_weight")
    for src, dist_map in paths:
        if src not in idx:
            continue
        for dst, distance in dist_map.items():
            if dst in idx:
                # Scale graph distance into a latency-like value below the origin path.
                distances[idx[src], idx[dst]] = min(origin_latency, 2.0 + 18.0 * float(distance))
    return distances

