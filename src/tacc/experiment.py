from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

from .data import SyntheticTraceConfig, ensure_trace, load_movement_trace
from .demand import generate_zipf_demand
from .evaluate import CostConfig, average_metrics, evaluate_placement
from .graph import GraphBuildConfig, build_handoff_graph, perturb_graph
from .placement import (
    diversified_popularity_placement,
    global_popularity_placement,
    local_popularity_placement,
    random_placement,
    topology_greedy_placement,
)
from .reporting import write_latex_assets
from .rl import DQNConfig, train_and_refine


@dataclass(frozen=True)
class ExperimentConfig:
    trace_path: str = "data/raw/dartmouth_movement.csv"
    output_dir: str = "results"
    seed: int = 11
    max_nodes: int = 20
    catalog_size: int = 48
    cache_capacity: int = 5
    zipf_alpha: float = 0.85
    locality_strength: float = 2.8
    greedy_samples: int = 2
    eval_samples: int = 12
    perturbation_rates: tuple[float, ...] = (0.00, 0.05, 0.10, 0.15)
    dqn_episodes: int = 85
    dqn_steps: int = 16
    selector_variance_weight: float = 0.25
    selector_replication_weight: float = 20.0
    selector_compute_weight: float = 0.03
    dqn_trigger_margin: float = 0.35
    dqn_trigger_std: float = 1.25
    dqn_instability_credit: float = 2.0


POLICY_COMPLEXITY = {
    "diversified_popularity": 0.05,
    "topology_greedy": 0.35,
    "hybrid_dqn": 1.00,
    "online_dqn": 0.85,
}


def _bootstrap_reference(policy: str, placement: np.ndarray, greedy: np.ndarray) -> np.ndarray:
    if policy == "hybrid_dqn":
        return greedy
    return placement


def _validation_metrics(
    graph,
    nodes: list[str],
    placement,
    demand,
    initial,
    cost_cfg: CostConfig,
    rate: float,
    samples: int,
    seed: int,
) -> pd.DataFrame:
    validation_rows = []
    for sample in range(samples):
        rng = np.random.default_rng(seed + int(rate * 1000) + sample)
        g = perturb_graph(graph, rng, remove_node_rate=rate, remove_edge_rate=rate / 2.0)
        validation_rows.append(evaluate_placement(g, nodes, placement, demand, initial, cost_cfg))
    return pd.DataFrame(validation_rows)


def _selector_score(policy: str, validation_frame: pd.DataFrame, rate: float, config: ExperimentConfig) -> float:
    volatility_penalty = (
        config.selector_replication_weight * (rate**2) * validation_frame["replication_cost"].mean()
    )
    compute_penalty = config.selector_compute_weight * POLICY_COMPLEXITY.get(policy, 0.50)
    return float(
        validation_frame["objective"].mean()
        + config.selector_variance_weight * validation_frame["objective"].std(ddof=1)
        + volatility_penalty
        + compute_penalty
    )


def run_experiment(config: ExperimentConfig) -> dict[str, Path]:
    start = time.time()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path, used_fallback = ensure_trace(
        Path(config.trace_path),
        SyntheticTraceConfig(seed=config.seed, aps=max(config.max_nodes + 12, 32)),
    )
    trace = load_movement_trace(trace_path)
    graph = build_handoff_graph(trace, GraphBuildConfig(max_nodes=config.max_nodes))
    nodes = list(graph.nodes())
    demand, global_popularity = generate_zipf_demand(
        graph,
        catalog_size=config.catalog_size,
        alpha=config.zipf_alpha,
        locality_strength=config.locality_strength,
        seed=config.seed,
    )
    cost_cfg = CostConfig()
    dqn_cfg = DQNConfig(episodes=config.dqn_episodes, steps_per_episode=config.dqn_steps)

    placements = {
        "random": random_placement(len(nodes), config.catalog_size, config.cache_capacity, config.seed),
        "local_popularity": local_popularity_placement(demand, config.cache_capacity),
        "global_popularity": global_popularity_placement(demand, config.cache_capacity),
        "diversified_popularity": diversified_popularity_placement(demand, config.cache_capacity),
    }

    greedy = topology_greedy_placement(
        graph,
        nodes,
        demand,
        config.cache_capacity,
        cost_cfg,
        perturb_rate=0.05,
        samples=config.greedy_samples,
        seed=config.seed,
    )
    placements["topology_greedy"] = greedy

    validation_samples = max(8, config.eval_samples)
    base_validation: dict[float, dict[str, pd.DataFrame]] = {}
    dqn_candidate_rates: list[float] = []
    dqn_gate: dict[str, dict[str, float | bool | str]] = {}
    for rate in config.perturbation_rates:
        base_validation[rate] = {}
        base_scores = {}
        for policy in ["diversified_popularity", "topology_greedy"]:
            frame = _validation_metrics(
                graph,
                nodes,
                placements[policy],
                demand,
                placements[policy],
                cost_cfg,
                rate,
                validation_samples,
                config.seed + 10000,
            )
            base_validation[rate][policy] = frame
            base_scores[policy] = _selector_score(policy, frame, rate, config)

        ordered = sorted(base_scores.items(), key=lambda item: item[1])
        score_gap = ordered[1][1] - ordered[0][1]
        best_std = float(base_validation[rate][ordered[0][0]]["objective"].std(ddof=1))
        triggered = score_gap <= config.dqn_trigger_margin or best_std >= config.dqn_trigger_std
        dqn_gate[f"{rate:.2f}"] = {
            "best_base_policy": ordered[0][0],
            "base_score_gap": float(score_gap),
            "best_base_objective_std": best_std,
            "dqn_considered": triggered,
        }
        if triggered:
            dqn_candidate_rates.append(rate)

    training_stats: dict[str, float | bool] = {
        "offline_dqn_trained": bool(dqn_candidate_rates),
        "dqn_candidate_rate_count": float(len(dqn_candidate_rates)),
        "online_dqn_runs": 0.0,
    }
    if dqn_candidate_rates:
        dqn_training_rate = float(np.mean(dqn_candidate_rates))
        hybrid, dqn_stats = train_and_refine(
            graph,
            nodes,
            demand,
            greedy,
            config.cache_capacity,
            cost_cfg,
            perturb_rate=dqn_training_rate,
            seed=config.seed + 99,
            cfg=dqn_cfg,
        )
        placements["hybrid_dqn"] = hybrid
        training_stats.update(dqn_stats)
        training_stats["dqn_training_rate"] = dqn_training_rate

    rows: list[dict[str, float | str]] = []
    detail_rows: list[dict[str, float | str | int]] = []
    selector_decisions: dict[str, dict[str, float | str]] = {}
    online_reference = greedy.copy()
    previous_adaptive_policy = "bootstrap_topology_greedy"
    previous_policy_placements: dict[str, np.ndarray] = {}
    for rate in config.perturbation_rates:
        candidate_placements = {
            "diversified_popularity": placements["diversified_popularity"],
            "topology_greedy": placements["topology_greedy"],
        }
        online_dqn, online_dqn_stats = train_and_refine(
            graph,
            nodes,
            demand,
            online_reference,
            config.cache_capacity,
            cost_cfg,
            perturb_rate=rate,
            seed=config.seed + 50000 + int(rate * 1000),
            cfg=dqn_cfg,
        )
        candidate_placements["online_dqn"] = online_dqn
        training_stats["online_dqn_runs"] = float(training_stats["online_dqn_runs"]) + 1.0
        training_stats[f"online_dqn_{rate:.2f}_objective"] = online_dqn_stats["nominal_objective"]
        training_stats[f"online_dqn_{rate:.2f}_loss"] = online_dqn_stats["training_loss"]
        training_stats[f"online_dqn_{rate:.2f}_relocation_entries"] = float(
            np.logical_xor(online_dqn, online_reference).sum()
        )
        validation_scores: dict[str, float] = {}
        for policy, placement in candidate_placements.items():
            validation_frame = _validation_metrics(
                graph,
                nodes,
                placement,
                demand,
                online_reference,
                cost_cfg,
                rate,
                validation_samples,
                config.seed + 10000,
            )
            validation_scores[policy] = _selector_score(policy, validation_frame, rate, config)
            if policy == "online_dqn":
                gate = dqn_gate[f"{rate:.2f}"]
                cheap_std = float(gate["best_base_objective_std"])
                instability = max(0.0, cheap_std - config.dqn_trigger_std)
                validation_scores[policy] -= config.dqn_instability_credit * instability
        selected_policy = min(validation_scores, key=validation_scores.get)
        selector_decisions[f"{rate:.2f}"] = {
            **{f"score_{policy}": score for policy, score in validation_scores.items()},
            "previous_policy": previous_adaptive_policy,
            "selected_policy": selected_policy,
            "online_dqn_relocation_entries": float(np.logical_xor(online_dqn, online_reference).sum()),
        }

        for policy, placement in placements.items():
            initial = previous_policy_placements.get(policy)
            if initial is None:
                initial = _bootstrap_reference(policy, placement, greedy)
            metric_rows = []
            for sample in range(config.eval_samples):
                rng_seed = config.seed + int(rate * 1000) + sample
                rng = np.random.default_rng(rng_seed)
                g = perturb_graph(graph, rng, remove_node_rate=rate, remove_edge_rate=rate / 2.0)
                metrics = evaluate_placement(g, nodes, placement, demand, initial, cost_cfg)
                metric_rows.append(metrics)
                detail_rows.append(
                    {
                        "policy": policy,
                        "perturbation_rate": rate,
                        "sample": sample,
                        **metrics,
                    }
                )
            averaged = average_metrics(metric_rows)
            rows.append(
                {
                    "policy": policy,
                    "perturbation_rate": rate,
                    **averaged,
                    "objective_std": float(pd.DataFrame(metric_rows)["objective"].std(ddof=1)),
                }
            )
            previous_policy_placements[policy] = placement.copy()

        online_metric_rows = []
        for sample in range(config.eval_samples):
            rng = np.random.default_rng(config.seed + int(rate * 1000) + sample)
            g = perturb_graph(graph, rng, remove_node_rate=rate, remove_edge_rate=rate / 2.0)
            metrics = evaluate_placement(g, nodes, online_dqn, demand, online_reference, cost_cfg)
            online_metric_rows.append(metrics)
            detail_rows.append(
                {
                    "policy": "online_dqn",
                    "previous_policy": previous_adaptive_policy,
                    "perturbation_rate": rate,
                    "sample": sample,
                    **metrics,
                }
            )
        online_averaged = average_metrics(online_metric_rows)
        rows.append(
            {
                "policy": "online_dqn",
                "previous_policy": previous_adaptive_policy,
                "perturbation_rate": rate,
                **online_averaged,
                "objective_std": float(pd.DataFrame(online_metric_rows)["objective"].std(ddof=1)),
            }
        )

        selected_placement = candidate_placements[selected_policy]
        selected_initial = online_reference
        metric_rows = []
        for sample in range(config.eval_samples):
            rng = np.random.default_rng(config.seed + int(rate * 1000) + sample)
            g = perturb_graph(graph, rng, remove_node_rate=rate, remove_edge_rate=rate / 2.0)
            metrics = evaluate_placement(g, nodes, selected_placement, demand, selected_initial, cost_cfg)
            metric_rows.append(metrics)
            detail_rows.append(
                {
                    "policy": "adaptive_tacc",
                    "previous_policy": previous_adaptive_policy,
                    "selected_policy": selected_policy,
                    "perturbation_rate": rate,
                    "sample": sample,
                    **metrics,
                }
            )
        averaged = average_metrics(metric_rows)
        rows.append(
            {
                "policy": "adaptive_tacc",
                "previous_policy": previous_adaptive_policy,
                "selected_policy": selected_policy,
                "perturbation_rate": rate,
                **averaged,
                "objective_std": float(pd.DataFrame(metric_rows)["objective"].std(ddof=1)),
            }
        )
        online_reference = selected_placement.copy()
        previous_adaptive_policy = selected_policy

    summary = pd.DataFrame(rows)
    summary_path = output_dir / "summary_metrics.csv"
    summary.to_csv(summary_path, index=False)
    detail_path = output_dir / "sample_metrics.csv"
    pd.DataFrame(detail_rows).to_csv(detail_path, index=False)

    graph_stats = {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "trace_rows": int(len(trace)),
        "used_synthetic_fallback": used_fallback,
        "top_global_content_probability": float(global_popularity[0]),
        "runtime_seconds": time.time() - start,
        "training_stats": training_stats,
        "dqn_gate": dqn_gate,
        "selector_decisions": selector_decisions,
        "online_relocation_reference": "Adaptive TACC evaluates each selected placement against the previously deployed Adaptive TACC placement; the online horizon is bootstrapped from topology-aware greedy.",
        "config": asdict(config),
    }
    stats_path = output_dir / "run_metadata.json"
    stats_path.write_text(json.dumps(graph_stats, indent=2), encoding="utf-8")
    write_latex_assets(summary, Path("Paper/report/generated"))
    return {"summary": summary_path, "sample_metrics": detail_path, "metadata": stats_path}
