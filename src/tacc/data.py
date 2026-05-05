from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd


MOVEMENT_COLUMNS = ["user_id", "timestamp", "ap"]


@dataclass(frozen=True)
class SyntheticTraceConfig:
    users: int = 320
    aps: int = 36
    steps_per_user: int = 80
    communities: int = 6
    seed: int = 7


def generate_synthetic_movement(config: SyntheticTraceConfig) -> pd.DataFrame:
    """Generate a deterministic Dartmouth-like user/AP association trace."""
    rng = np.random.default_rng(config.seed)
    aps = np.array([f"AP{idx:03d}" for idx in range(config.aps)])
    community_ids = np.arange(config.communities)
    ap_communities = np.array_split(aps, config.communities)

    rows: list[tuple[str, int, str]] = []
    for user_idx in range(config.users):
        user = f"U{user_idx:04d}"
        home = int(rng.choice(community_ids))
        current_ap = str(rng.choice(ap_communities[home]))
        timestamp = int(rng.integers(0, 3600))

        for step in range(config.steps_per_user):
            rows.append((user, timestamp, current_ap))
            timestamp += int(rng.integers(90, 900))

            r = rng.random()
            if r < 0.08:
                current_ap = "OFF"
            elif current_ap == "OFF":
                current_ap = str(rng.choice(ap_communities[home]))
            elif r < 0.68:
                current_ap = str(rng.choice(ap_communities[home]))
            elif r < 0.90:
                neighbor = int(np.clip(home + rng.choice([-1, 1]), 0, config.communities - 1))
                current_ap = str(rng.choice(ap_communities[neighbor]))
            else:
                current_ap = str(rng.choice(aps))

            if step % 27 == 0 and rng.random() < 0.35:
                home = int(rng.choice(community_ids))

    return pd.DataFrame(rows, columns=MOVEMENT_COLUMNS)


def save_synthetic_movement(path: Path, config: SyntheticTraceConfig) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_movement(config)
    df.to_csv(path, index=False)
    metadata = {
        "synthetic_fallback": True,
        "reason": "Generated because the Dartmouth movement trace was not present locally.",
        "config": config.__dict__,
    }
    path.with_suffix(path.suffix + ".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path


def load_movement_trace(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(MOVEMENT_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Movement trace is missing required columns: {sorted(missing)}")
    df = df[MOVEMENT_COLUMNS].copy()
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["user_id", "timestamp", "ap"])
    df["timestamp"] = df["timestamp"].astype(int)
    return df


def ensure_trace(path: Path, synthetic_config: SyntheticTraceConfig | None = None) -> tuple[Path, bool]:
    """Return an existing movement trace, or create the deterministic fallback."""
    if path.exists():
        metadata_path = path.with_suffix(path.suffix + ".metadata.json")
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            return path, bool(metadata.get("synthetic_fallback", False))
        return path, False
    config = synthetic_config or SyntheticTraceConfig()
    return save_synthetic_movement(path, config), True
