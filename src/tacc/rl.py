from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random

import networkx as nx
import numpy as np
import torch
from torch import nn

from .evaluate import CostConfig, evaluate_placement
from .graph import graph_features, perturb_graph


@dataclass(frozen=True)
class DQNConfig:
    episodes: int = 90
    steps_per_episode: int = 18
    batch_size: int = 64
    gamma: float = 0.92
    learning_rate: float = 0.001
    epsilon_start: float = 0.35
    epsilon_end: float = 0.05
    replay_size: int = 5000
    hidden_dim: int = 192
    inference_samples: int = 6


class QNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


def _state(placement: np.ndarray, demand: np.ndarray, features: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            placement.astype(np.float32).ravel(),
            demand.astype(np.float32).ravel(),
            features.astype(np.float32).ravel(),
        ]
    )


def _apply_action(placement: np.ndarray, action: int, demand: np.ndarray, capacity: int) -> np.ndarray:
    n_nodes, catalog_size = placement.shape
    i = action // catalog_size
    c = action % catalog_size
    x = placement.copy()
    if x[i, c]:
        if int(x[i].sum()) > 1:
            x[i, c] = False
        return x
    if int(x[i].sum()) >= capacity:
        cached = np.flatnonzero(x[i])
        victim = cached[np.argmin(demand[i, cached])]
        x[i, victim] = False
    x[i, c] = True
    return x


def _expected_objective(
    graph: nx.Graph,
    nodes: list[str],
    placement: np.ndarray,
    demand: np.ndarray,
    reference: np.ndarray,
    cost_cfg: CostConfig,
    perturb_rate: float,
    samples: int,
    seed: int,
) -> float:
    values = []
    for sample in range(samples):
        rng = np.random.default_rng(seed + sample)
        g = perturb_graph(graph, rng, remove_node_rate=perturb_rate, remove_edge_rate=perturb_rate / 2.0)
        values.append(evaluate_placement(g, nodes, placement, demand, reference, cost_cfg)["objective"])
    return float(np.mean(values))


def train_and_refine(
    graph: nx.Graph,
    nodes: list[str],
    demand: np.ndarray,
    initial_placement: np.ndarray,
    capacity: int,
    cost_cfg: CostConfig,
    perturb_rate: float,
    seed: int,
    cfg: DQNConfig | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    cfg = cfg or DQNConfig()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    features = graph_features(graph)
    state_dim = initial_placement.size + demand.size + features.size
    action_dim = initial_placement.size
    q = QNet(state_dim, action_dim, cfg.hidden_dim)
    target = QNet(state_dim, action_dim, cfg.hidden_dim)
    target.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=cfg.learning_rate)
    replay: deque[tuple[np.ndarray, int, float, np.ndarray]] = deque(maxlen=cfg.replay_size)
    loss_value = 0.0

    for episode in range(cfg.episodes):
        epsilon = cfg.epsilon_end + (cfg.epsilon_start - cfg.epsilon_end) * (1.0 - episode / max(cfg.episodes, 1))
        g = perturb_graph(graph, rng, remove_node_rate=perturb_rate, remove_edge_rate=perturb_rate / 2.0)
        x = initial_placement.copy()
        baseline = evaluate_placement(g, nodes, x, demand, initial_placement, cost_cfg)["objective"]
        s = _state(x, demand, features)

        for _ in range(cfg.steps_per_episode):
            if rng.random() < epsilon:
                action = int(rng.integers(0, action_dim))
            else:
                with torch.no_grad():
                    action = int(torch.argmax(q(torch.from_numpy(s).float().unsqueeze(0))).item())
            x_next = _apply_action(x, action, demand, capacity)
            score = evaluate_placement(g, nodes, x_next, demand, initial_placement, cost_cfg)["objective"]
            reward = baseline - score
            s_next = _state(x_next, demand, features)
            replay.append((s, action, float(reward), s_next))
            x, s, baseline = x_next, s_next, score

            if len(replay) >= cfg.batch_size:
                batch = random.sample(replay, cfg.batch_size)
                states = torch.tensor(np.stack([b[0] for b in batch]), dtype=torch.float32)
                actions = torch.tensor([b[1] for b in batch], dtype=torch.int64)
                rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32)
                next_states = torch.tensor(np.stack([b[3] for b in batch]), dtype=torch.float32)
                q_values = q(states).gather(1, actions.view(-1, 1)).squeeze(1)
                with torch.no_grad():
                    targets = rewards + cfg.gamma * target(next_states).max(dim=1).values
                loss = nn.functional.smooth_l1_loss(q_values, targets)
                opt.zero_grad()
                loss.backward()
                opt.step()
                loss_value = float(loss.item())

        if episode % 10 == 0:
            target.load_state_dict(q.state_dict())

    refined = initial_placement.copy()
    best = _expected_objective(
        graph,
        nodes,
        refined,
        demand,
        initial_placement,
        cost_cfg,
        perturb_rate,
        cfg.inference_samples,
        seed + 100000,
    )
    for _ in range(cfg.steps_per_episode * 2):
        s = _state(refined, demand, features)
        with torch.no_grad():
            ranked = torch.argsort(q(torch.from_numpy(s).float().unsqueeze(0)).squeeze(0), descending=True)
        improved = False
        for action_tensor in ranked[: min(40, action_dim)]:
            candidate = _apply_action(refined, int(action_tensor.item()), demand, capacity)
            score = _expected_objective(
                graph,
                nodes,
                candidate,
                demand,
                initial_placement,
                cost_cfg,
                perturb_rate,
                cfg.inference_samples,
                seed + 100000,
            )
            if score < best:
                refined, best = candidate, score
                improved = True
                break
        if not improved:
            break

    return refined, {"training_loss": loss_value, "nominal_objective": best}
