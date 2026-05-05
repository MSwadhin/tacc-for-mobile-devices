#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tacc.experiment import ExperimentConfig, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run topology-aware cooperative caching experiments.")
    parser.add_argument("--trace", default="data/raw/dartmouth_movement.csv")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--max-nodes", type=int, default=20)
    parser.add_argument("--catalog-size", type=int, default=48)
    parser.add_argument("--cache-capacity", type=int, default=5)
    parser.add_argument("--dqn-episodes", type=int, default=85)
    parser.add_argument("--dqn-steps", type=int, default=16)
    args = parser.parse_args()

    config = ExperimentConfig(
        trace_path=args.trace,
        output_dir=args.output_dir,
        seed=args.seed,
        max_nodes=args.max_nodes,
        catalog_size=args.catalog_size,
        cache_capacity=args.cache_capacity,
        dqn_episodes=args.dqn_episodes,
        dqn_steps=args.dqn_steps,
    )
    outputs = run_experiment(config)
    print(f"Wrote metrics to {outputs['summary']}")
    print(f"Wrote sample metrics to {outputs['sample_metrics']}")
    print(f"Wrote metadata to {outputs['metadata']}")


if __name__ == "__main__":
    main()
