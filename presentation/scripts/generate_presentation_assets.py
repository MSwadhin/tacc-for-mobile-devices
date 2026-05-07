from __future__ import annotations

from pathlib import Path
import json
import textwrap

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle
import networkx as nx
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "presentation"
FIG = OUT / "figures"
TAB = OUT / "tables"
SUMMARY = ROOT / "results" / "final" / "summary_metrics.csv"
META = ROOT / "results" / "final" / "run_metadata.json"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 18,
        "axes.titlesize": 24,
        "axes.labelsize": 19,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
        "figure.dpi": 180,
        "savefig.dpi": 240,
    }
)

COLORS = {
    "online_dqn": "#0072B2",
    "keep_current": "#009E73",
    "hybrid_dqn": "#D55E00",
    "topology_greedy": "#CC79A7",
    "diversified_popularity": "#E69F00",
    "random": "#7F7F7F",
    "local_popularity": "#56B4E9",
    "global_popularity": "#8C564B",
}

LABELS = {
    "online_dqn": "Online DQN",
    "keep_current": "Keep Current",
    "hybrid_dqn": "Fixed Greedy+DQN",
    "topology_greedy": "Topology Greedy",
    "diversified_popularity": "Diversified",
    "random": "Random",
    "local_popularity": "Local Pop.",
    "global_popularity": "Global Pop.",
}


def save(fig: plt.Figure, name: str) -> None:
    for ext in ("png", "svg"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def box(ax, xy, wh, text, color="#E8F1FA", edge="#2B5C7C", fontsize=20):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.025,rounding_size=0.035",
        linewidth=2.2,
        edgecolor=edge,
        facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, weight="bold", wrap=True)
    return patch


def arrow(ax, start, end, color="#333333", lw=2.6):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=20, linewidth=lw, color=color))


def flow_infographic() -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.03, 0.92, "Research Flow", fontsize=30, weight="bold")
    items = [
        ("Mobility\nhandoffs", "#DDEAF6"),
        ("Cooperation\ngraph", "#E3F4EA"),
        ("Stochastic\nobjective", "#FFF0D6"),
        ("Online DQN\nrefinement", "#E8E3F8"),
        ("Multi-metric\nevaluation", "#F9E2E2"),
    ]
    xs = np.linspace(0.06, 0.78, len(items))
    for i, (label, color) in enumerate(items):
        box(ax, (xs[i], 0.52), (0.16, 0.22), label, color=color, fontsize=19)
        if i < len(items) - 1:
            arrow(ax, (xs[i] + 0.16, 0.63), (xs[i + 1] - 0.01, 0.63))
    ax.text(
        0.06,
        0.26,
        "Key scientific idea: evaluate cache placement as a sequential control problem,\nnot as a one-shot popularity ranking.",
        fontsize=22,
        color="#222222",
    )
    save(fig, "fig_research_flow")


def challenge_map() -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.03, 0.92, "Challenges → Modeling Decisions", fontsize=30, weight="bold")
    rows = [
        ("Mobility changes\ncooperation", "Weighted AP\nhandoff graph"),
        ("Demand differs\nby AP", "Zipf + spatial\nlocality"),
        ("Capacity is\nsmall", "Binary placement\nwith K=5"),
        ("Duplication can\nwaste coverage", "Edge-weighted\nredundancy"),
        ("Topology is\nunstable", "Node/edge\nperturbation"),
        ("Updates are\ncostly", "Relocation vs.\nprevious cache"),
    ]
    y0 = 0.77
    for i, (left, right) in enumerate(rows):
        y = y0 - i * 0.12
        box(ax, (0.07, y), (0.33, 0.08), left, color="#F3F6FA", edge="#5B6B7A", fontsize=18)
        arrow(ax, (0.42, y + 0.04), (0.56, y + 0.04), color="#666666", lw=2.2)
        box(ax, (0.59, y), (0.33, 0.08), right, color="#EAF6EF", edge="#27734D", fontsize=18)
    save(fig, "fig_challenge_mapping")


def system_model() -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.03, 0.92, "System Model", fontsize=30, weight="bold")
    box(ax, (0.05, 0.62), (0.23, 0.18), "AP caches\n(nodes V)", color="#E8F1FA", fontsize=20)
    box(ax, (0.38, 0.62), (0.23, 0.18), "Handoff edges\n(weights W)", color="#E3F4EA", fontsize=20)
    box(ax, (0.71, 0.62), (0.23, 0.18), "Perturbed graph\n$\\tilde{G}$", color="#FFF0D6", fontsize=20)
    arrow(ax, (0.28, 0.71), (0.38, 0.71))
    arrow(ax, (0.61, 0.71), (0.71, 0.71))
    box(ax, (0.12, 0.23), (0.24, 0.18), "Placement matrix\n$x_{i,c}$", color="#F9E2E2", fontsize=20)
    box(ax, (0.42, 0.23), (0.24, 0.18), "Demand matrix\n$P_{i,c}$", color="#E8E3F8", fontsize=20)
    box(ax, (0.72, 0.23), (0.22, 0.18), "Objective\nlower is better", color="#F3F6FA", fontsize=20)
    arrow(ax, (0.36, 0.32), (0.42, 0.32))
    arrow(ax, (0.66, 0.32), (0.72, 0.32))
    ax.text(0.08, 0.08, "$J = C_{access} + \\lambda C_{rep} + \\beta C_{red} + \\mu C_{rel}$", fontsize=28, weight="bold")
    save(fig, "fig_system_model")


def pipeline() -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.03, 0.92, "Method Pipeline", fontsize=30, weight="bold")
    rows = [
        ("1", "Trace-compatible\nmobility events"),
        ("2", "Build weighted\nhandoff graph"),
        ("3", "Generate Zipf-local\ncontent demand"),
        ("4", "Create strong\ncandidate placements"),
        ("5", "Refine current cache\nwith Online DQN"),
        ("6", "Evaluate under\nperturbation samples"),
    ]
    xs = [0.05, 0.36, 0.67, 0.05, 0.36, 0.67]
    ys = [0.62, 0.62, 0.62, 0.28, 0.28, 0.28]
    for (num, label), x, y in zip(rows, xs, ys):
        box(ax, (x, y), (0.25, 0.17), label, color="#F5F7FB", edge="#3B5B7C", fontsize=18)
        ax.add_patch(Circle((x + 0.03, y + 0.14), 0.028, color="#0072B2"))
        ax.text(x + 0.03, y + 0.14, num, color="white", ha="center", va="center", fontsize=18, weight="bold")
    for s, e in [((0.30, 0.705), (0.36, 0.705)), ((0.61, 0.705), (0.67, 0.705)), ((0.79, 0.62), (0.18, 0.45)), ((0.30, 0.365), (0.36, 0.365)), ((0.61, 0.365), (0.67, 0.365))]:
        arrow(ax, s, e)
    save(fig, "fig_method_pipeline")


def dqn_loop() -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.03, 0.92, "Online DQN Refinement Loop", fontsize=30, weight="bold")
    positions = {
        "state": (0.08, 0.58),
        "q": (0.39, 0.58),
        "action": (0.70, 0.58),
        "reward": (0.39, 0.25),
        "accept": (0.70, 0.25),
        "reference": (0.08, 0.25),
    }
    box(ax, positions["state"], (0.22, 0.16), "State\nx, P, graph features", color="#E8F1FA", fontsize=18)
    box(ax, positions["q"], (0.22, 0.16), "Q-network\n2 hidden layers", color="#E8E3F8", fontsize=18)
    box(ax, positions["action"], (0.22, 0.16), "Action\nnode-content toggle", color="#FFF0D6", fontsize=18)
    box(ax, positions["reward"], (0.22, 0.16), "Reward\nobjective reduction", color="#E3F4EA", fontsize=18)
    box(ax, positions["accept"], (0.22, 0.16), "Accept only if\nJ improves", color="#F9E2E2", fontsize=18)
    box(ax, positions["reference"], (0.22, 0.16), "Reference\nprevious cache", color="#F3F6FA", fontsize=18)
    arrow(ax, (0.30, 0.66), (0.39, 0.66))
    arrow(ax, (0.61, 0.66), (0.70, 0.66))
    arrow(ax, (0.81, 0.58), (0.81, 0.41))
    arrow(ax, (0.70, 0.33), (0.61, 0.33))
    arrow(ax, (0.39, 0.33), (0.30, 0.33))
    arrow(ax, (0.19, 0.41), (0.19, 0.58))
    save(fig, "fig_dqn_loop")


def edge_caching_motivation() -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.03, 0.92, "Why Edge Caching?", fontsize=30, weight="bold")
    box(ax, (0.08, 0.54), (0.20, 0.16), "Users /\nIoT devices", color="#E8F1FA", fontsize=19)
    box(ax, (0.40, 0.54), (0.20, 0.16), "AP edge\ncache", color="#E3F4EA", fontsize=19)
    box(ax, (0.72, 0.54), (0.20, 0.16), "Remote\norigin", color="#F9E2E2", fontsize=19)
    arrow(ax, (0.28, 0.62), (0.40, 0.62), color="#009E73")
    arrow(ax, (0.60, 0.62), (0.72, 0.62), color="#D55E00")
    ax.text(0.18, 0.43, "local hit\nlow latency", ha="center", fontsize=19, color="#006B45")
    ax.text(0.66, 0.43, "origin fetch\nhigh latency", ha="center", fontsize=19, color="#9B2D00")
    ax.text(0.08, 0.18, "Caching decision: which objects should each AP store,\ngiven mobility, cooperation, capacity, and update cost?", fontsize=23)
    save(fig, "fig_edge_caching_motivation")


def research_gap() -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.03, 0.92, "Research Gap", fontsize=30, weight="bold")
    gaps = [
        ("Static topology", "misses mobility-driven\ncooperation changes"),
        ("Popularity only", "misses graph distance\nand redundancy"),
        ("No update cost", "hides cache rewrite\noverhead"),
    ]
    xs = [0.08, 0.39, 0.70]
    for x, (title, note) in zip(xs, gaps):
        box(ax, (x, 0.56), (0.22, 0.14), title, color="#F9E2E2", edge="#A64040", fontsize=20)
        ax.text(x + 0.11, 0.39, note, ha="center", va="center", fontsize=20)
    ax.text(0.08, 0.16, "This project closes the gap by evaluating cache placement as a\nperturbed, topology-aware, sequential online control problem.", fontsize=23)
    save(fig, "fig_research_gap")


def research_questions() -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.03, 0.92, "Research Questions", fontsize=30, weight="bold")
    rqs = [
        ("RQ1", "Mobility\n-> graph?"),
        ("RQ2", "Does topology\nhelp?"),
        ("RQ3", "Does learning\nadd value?"),
        ("RQ4", "When should\nwe update?"),
    ]
    xs = [0.07, 0.30, 0.53, 0.76]
    for x, (num, txt) in zip(xs, rqs):
        box(ax, (x, 0.45), (0.18, 0.24), f"{num}\n{txt}", color="#F5F7FB", edge="#3B5B7C", fontsize=20)
    ax.text(0.08, 0.20, "The deck answers these in order: formulation -> method -> measured results -> cautious interpretation.", fontsize=22)
    save(fig, "fig_research_questions")


def greedy_flow() -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.03, 0.92, "Topology-Aware Greedy", fontsize=30, weight="bold")
    steps = [
        ("Start with\nempty cache", "#F3F6FA"),
        ("Enumerate feasible\nnode-content pairs", "#E8F1FA"),
        ("Evaluate marginal\ngain under samples", "#FFF0D6"),
        ("Add best pair\nuntil capacity full", "#E3F4EA"),
    ]
    xs = [0.06, 0.30, 0.54, 0.78]
    for i, ((label, color), x) in enumerate(zip(steps, xs)):
        box(ax, (x, 0.52), (0.17, 0.18), label, color=color, edge="#444444", fontsize=18)
        if i < len(steps) - 1:
            arrow(ax, (x + 0.17, 0.61), (xs[i + 1] - 0.01, 0.61))
    ax.text(0.07, 0.24, "Purpose: create an interpretable, graph-aware warm start\nbefore learning searches local cache edits.", fontsize=23)
    save(fig, "fig_greedy_flow")


def experiment_protocol(meta: dict) -> None:
    cfg = meta["config"]
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.03, 0.92, "Experimental Protocol", fontsize=30, weight="bold")
    cards = [
        ("Topology", f"{meta['nodes']} AP nodes\n{meta['edges']} edges"),
        ("Catalog", f"{cfg['catalog_size']} objects\nK={cfg['cache_capacity']}"),
        ("Demand", f"Zipf alpha={cfg['zipf_alpha']}\nlocality={cfg['locality_strength']}"),
        ("Perturbation", "p=0.00..0.25\n12 samples/rate"),
        ("DQN", f"{cfg['dqn_episodes']} episodes\n{cfg['dqn_steps']} steps"),
        ("Weights", "lambda=0.02\nbeta=18, mu=0.04"),
    ]
    for idx, (title, text) in enumerate(cards):
        x = 0.07 + (idx % 3) * 0.30
        y = 0.58 if idx < 3 else 0.28
        box(ax, (x, y), (0.23, 0.16), f"{title}\n{text}", color="#F5F7FB", edge="#3B5B7C", fontsize=18)
    save(fig, "fig_experiment_protocol")


def rq_answers() -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.03, 0.92, "Research Question Answers", fontsize=30, weight="bold")
    answers = [
        ("RQ1", "20-node handoff\ngraph constructed"),
        ("RQ2", "Topology beats\nlocal/global popularity"),
        ("RQ3", "Online DQN beats\nfixed learned refinement"),
        ("RQ4", "p=0.25 requires\nno-update check"),
    ]
    xs = [0.06, 0.29, 0.52, 0.75]
    colors = ["#E3F4EA", "#E3F4EA", "#E3F4EA", "#FFF0D6"]
    for x, color, (rq, ans) in zip(xs, colors, answers):
        box(ax, (x, 0.43), (0.19, 0.25), f"{rq}\n{ans}", color=color, edge="#27734D" if color != "#FFF0D6" else "#AA7B00", fontsize=18)
    ax.text(0.08, 0.18, "Main interpretation: topology-aware online caching works best when rewriting is treated as a measured decision.", fontsize=22)
    save(fig, "fig_rq_answers")


def limitations() -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.03, 0.92, "Limitations", fontsize=30, weight="bold")
    items = [
        "Trace-compatible\nmobility",
        "Synthetic\ncontent demand",
        "20 AP / 48 object\nscale",
        "Acceptance under\nhigh perturbation",
        "Explicit graph\nfeatures, not GNN",
    ]
    xs = np.linspace(0.06, 0.78, 5)
    for x, item in zip(xs, items):
        box(ax, (x, 0.48), (0.16, 0.20), item, color="#F9E2E2", edge="#A64040", fontsize=18)
    ax.text(0.08, 0.20, "These limit deployment magnitude claims, not the reproducible method pipeline.", fontsize=23)
    save(fig, "fig_limitations")


def future_work() -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.03, 0.92, "Future Work Roadmap", fontsize=30, weight="bold")
    steps = [
        ("Raw traces", "real AP\nassociations"),
        ("Temporal split", "train past\nevaluate future"),
        ("Acceptance tests", "confidence and\nno-update thresholds"),
        ("Scale up", "larger graphs\ncandidate pruning"),
        ("GNN policies", "permutation-aware\nencoders"),
    ]
    xs = np.linspace(0.08, 0.82, 5)
    for i, (title, note) in enumerate(steps):
        ax.add_patch(Circle((xs[i], 0.58), 0.055, color="#0072B2"))
        ax.text(xs[i], 0.58, str(i + 1), color="white", ha="center", va="center", fontsize=22, weight="bold")
        ax.text(xs[i], 0.39, f"{title}\n{note}", ha="center", va="center", fontsize=18)
        if i < len(steps) - 1:
            arrow(ax, (xs[i] + 0.06, 0.58), (xs[i + 1] - 0.06, 0.58))
    save(fig, "fig_future_work")


def takeaway() -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.03, 0.92, "Final Takeaway", fontsize=30, weight="bold")
    items = [
        ("Topology", "Mobility-derived graphs\nshould shape placement."),
        ("Learning", "DQN should refine the\ncurrent deployed cache."),
        ("Control", "Rewriting must beat\nthe no-update option."),
    ]
    xs = [0.08, 0.39, 0.70]
    colors = ["#E3F4EA", "#E8E3F8", "#FFF0D6"]
    for x, color, (title, note) in zip(xs, colors, items):
        box(ax, (x, 0.47), (0.22, 0.20), f"{title}\n{note}", color=color, edge="#444444", fontsize=18)
    ax.text(0.08, 0.18, "Cooperative edge caching is a sequential control problem,\nnot a single static popularity rule.", fontsize=25, weight="bold")
    save(fig, "fig_takeaway")


def objective_trends(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    policies = ["keep_current", "diversified_popularity", "topology_greedy", "hybrid_dqn", "online_dqn"]
    for policy in policies:
        g = df[df.policy == policy].sort_values("perturbation_rate")
        ax.plot(
            g["perturbation_rate"],
            g["objective"],
            marker="o",
            linewidth=3,
            markersize=8,
            color=COLORS[policy],
            label=LABELS[policy],
        )
    ax.set_title("Objective Across Perturbation Rates")
    ax.set_xlabel("Perturbation rate p")
    ax.set_ylabel("Objective (lower is better)")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, loc="upper left", frameon=False)
    ax.annotate(
        "Keep Current slightly wins\nat p=0.25",
        xy=(0.25, 4.342),
        xytext=(0.145, 2.65),
        arrowprops=dict(arrowstyle="->", lw=2, color="#222222"),
        fontsize=18,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#555555"),
    )
    save(fig, "fig_objective_trends")


def nominal_bars(df: pd.DataFrame) -> None:
    g = df[df.perturbation_rate == 0.0].copy()
    order = ["online_dqn", "topology_greedy", "diversified_popularity", "hybrid_dqn", "random", "local_popularity", "global_popularity"]
    g = g.set_index("policy").loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(16, 9))
    bars = ax.barh([LABELS[p] for p in g.policy], g.objective, color=[COLORS[p] for p in g.policy])
    ax.invert_yaxis()
    ax.set_title("Nominal Graph Objective at p=0.00")
    ax.set_xlabel("Objective (lower is better)")
    ax.grid(axis="x", alpha=0.25)
    for bar, val in zip(bars, g.objective):
        ax.text(val + 0.25, bar.get_y() + bar.get_height() / 2, f"{val:.2f}", va="center", fontsize=18)
    save(fig, "fig_nominal_baseline_bars")


def p025_closeup(df: pd.DataFrame) -> None:
    policies = ["keep_current", "online_dqn", "hybrid_dqn", "topology_greedy", "diversified_popularity"]
    g = df[(df.perturbation_rate == 0.25) & (df.policy.isin(policies))].set_index("policy").loc[policies].reset_index()
    fig, ax = plt.subplots(figsize=(16, 9))
    bars = ax.bar([LABELS[p] for p in g.policy], g.objective, color=[COLORS[p] for p in g.policy])
    ax.set_title("High Perturbation Check: p=0.25")
    ax.set_ylabel("Objective (lower is better)")
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(0, max(g.objective) * 1.2)
    for bar, val in zip(bars, g.objective):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.12, f"{val:.3f}", ha="center", fontsize=18)
    ax.text(
        0.08,
        0.82,
        "No-update beats Online DQN\nby 0.073 objective",
        transform=ax.transAxes,
        fontsize=20,
        bbox=dict(boxstyle="round,pad=0.4", fc="#FFF4CC", ec="#AA7B00"),
    )
    plt.setp(ax.get_xticklabels(), rotation=12, ha="right")
    save(fig, "fig_p025_closeup")


def hit_breakdown(df: pd.DataFrame) -> None:
    policies = ["online_dqn", "keep_current", "diversified_popularity", "topology_greedy", "hybrid_dqn"]
    g = df[(df.policy.isin(policies)) & (df.perturbation_rate.isin([0.0, 0.25]))].copy()
    g["label"] = g["policy"].map(LABELS) + "\np=" + g["perturbation_rate"].map(lambda x: f"{x:.2f}")
    fig, ax = plt.subplots(figsize=(16, 9))
    x = np.arange(len(g))
    ax.bar(x, g.local_hit_ratio, color="#0072B2", label="Local hit")
    ax.bar(x, g.cooperative_hit_ratio, bottom=g.local_hit_ratio, color="#009E73", label="Coop hit")
    ax.bar(
        x,
        g.origin_miss_ratio,
        bottom=g.local_hit_ratio + g.cooperative_hit_ratio,
        color="#D55E00",
        label="Origin miss",
    )
    ax.set_title("Hit-Ratio Decomposition")
    ax.set_ylabel("Probability mass")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(g.label, rotation=35, ha="right")
    ax.legend(ncol=3, frameon=False, loc="upper center")
    ax.grid(axis="y", alpha=0.2)
    save(fig, "fig_hit_breakdown")


def cost_components(df: pd.DataFrame) -> None:
    g = df[df.policy == "online_dqn"].sort_values("perturbation_rate").copy()
    rates = g.perturbation_rate.astype(str)
    repl = 0.02 * g.replication_cost
    red = 18.0 * g.redundancy_cost
    reloc = 0.04 * g.relocation_cost
    fig, ax = plt.subplots(figsize=(16, 9))
    x = np.arange(len(g))
    ax.bar(x, repl, label="Weighted replication", color="#56B4E9")
    ax.bar(x, red, bottom=repl, label="Weighted redundancy", color="#E69F00")
    ax.bar(x, reloc, bottom=repl + red, label="Weighted relocation", color="#CC79A7")
    ax.set_title("Online DQN Non-Access Overheads")
    ax.set_ylabel("Weighted objective contribution")
    ax.set_xticks(x)
    ax.set_xticklabels([f"p={r:.2f}" for r in g.perturbation_rate])
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    save(fig, "fig_weighted_cost_components")


def metric_heatmap(df: pd.DataFrame) -> None:
    policies = ["online_dqn", "keep_current", "diversified_popularity", "topology_greedy", "hybrid_dqn"]
    pivot = df[df.policy.isin(policies)].pivot(index="policy", columns="perturbation_rate", values="objective")
    pivot = pivot.loc[policies]
    fig, ax = plt.subplots(figsize=(16, 8))
    im = ax.imshow(pivot.values, cmap="YlGnBu_r", aspect="auto")
    ax.set_title("Objective Heatmap: Strong Policies")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{c:.2f}" for c in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([LABELS[p] for p in pivot.index])
    ax.set_xlabel("Perturbation rate p")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:.2f}", ha="center", va="center", fontsize=18)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Objective")
    save(fig, "fig_objective_heatmap")


def baseline_taxonomy() -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.03, 0.92, "Baseline Taxonomy", fontsize=30, weight="bold")
    columns = [
        ("Weak / sanity", ["Random", "Local popularity", "Global popularity"], "#F9E2E2"),
        ("Strong static", ["Diversified", "Topology-aware greedy"], "#FFF0D6"),
        ("Learning", ["Fixed Greedy+DQN", "Online DQN"], "#E8E3F8"),
        ("Operational check", ["Keep Current"], "#E3F4EA"),
    ]
    for i, (title, items, color) in enumerate(columns):
        x = 0.05 + i * 0.235
        box(ax, (x, 0.62), (0.20, 0.12), title, color=color, edge="#444444", fontsize=18)
        ax.text(x + 0.10, 0.47, "\n".join(items), ha="center", va="center", fontsize=19)
    ax.text(0.06, 0.16, "Why this matters: the method is compared against simple rules, strong non-neural diversity,\nlearned fixed refinement, and a no-update deployment baseline.", fontsize=21)
    save(fig, "fig_baseline_taxonomy")


def build_tables(df: pd.DataFrame) -> None:
    def write_markdown_table(frame: pd.DataFrame, path: Path) -> None:
        cols = list(frame.columns)
        lines = ["| " + " | ".join(str(c) for c in cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, row in frame.iterrows():
            values = []
            for value in row:
                if isinstance(value, float):
                    values.append(f"{value:.3f}")
                else:
                    values.append(str(value))
            lines.append("| " + " | ".join(values) + " |")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    nominal_cols = ["policy", "objective", "access_cost", "hit_ratio", "replication_cost", "redundancy_cost", "relocation_cost"]
    nominal = df[df.perturbation_rate == 0.0][nominal_cols].copy()
    nominal["policy"] = nominal["policy"].map(LABELS)
    nominal = nominal.sort_values("objective")
    nominal.to_csv(TAB / "nominal_metrics.csv", index=False)
    write_markdown_table(nominal, TAB / "nominal_metrics.md")

    policies = ["keep_current", "diversified_popularity", "topology_greedy", "hybrid_dqn", "online_dqn"]
    perturb = df[df.policy.isin(policies)].pivot(index="policy", columns="perturbation_rate", values="objective")
    perturb = perturb.loc[policies]
    perturb.index = [LABELS[p] for p in perturb.index]
    perturb.to_csv(TAB / "perturbation_objectives.csv")
    perturb_md = perturb.reset_index().rename(columns={"index": "policy"})
    write_markdown_table(perturb_md, TAB / "perturbation_objectives.md")

    online = df[df.policy == "online_dqn"][
        ["perturbation_rate", "objective", "access_cost", "hit_ratio", "replication_cost", "redundancy_cost", "relocation_cost"]
    ]
    online.to_csv(TAB / "online_dqn_metrics.csv", index=False)
    write_markdown_table(online, TAB / "online_dqn_metrics.md")


def network_illustration() -> None:
    rng = np.random.default_rng(11)
    g = nx.connected_watts_strogatz_graph(20, 5, 0.22, seed=11)
    pos = nx.spring_layout(g, seed=11)
    weights = rng.uniform(0.3, 2.0, size=g.number_of_edges())
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_title("Mobility-Derived AP Cooperation Graph")
    nx.draw_networkx_edges(g, pos, ax=ax, width=weights, edge_color="#9AA7B2", alpha=0.8)
    nx.draw_networkx_nodes(g, pos, ax=ax, node_size=780, node_color="#0072B2", edgecolors="white", linewidths=2)
    nx.draw_networkx_labels(g, pos, labels={i: str(i + 1) for i in range(20)}, font_size=18, font_color="white", ax=ax)
    ax.text(
        0.02,
        0.02,
        "Edges represent frequent handoffs; weights influence latency and redundancy.",
        transform=ax.transAxes,
        fontsize=19,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#777777"),
    )
    ax.axis("off")
    save(fig, "fig_mobility_graph_concept")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SUMMARY)
    meta = json.loads(META.read_text())

    flow_infographic()
    edge_caching_motivation()
    research_gap()
    research_questions()
    challenge_map()
    system_model()
    pipeline()
    dqn_loop()
    greedy_flow()
    baseline_taxonomy()
    experiment_protocol(meta)
    network_illustration()
    objective_trends(df)
    nominal_bars(df)
    p025_closeup(df)
    hit_breakdown(df)
    cost_components(df)
    metric_heatmap(df)
    rq_answers()
    limitations()
    future_work()
    takeaway()
    build_tables(df)

    asset_lines = [
        "# Generated Presentation Assets",
        "",
        f"Source metrics: `{SUMMARY.relative_to(ROOT)}`",
        f"Experiment: {meta['nodes']} nodes, {meta['edges']} edges, {meta['config']['catalog_size']} objects, K={meta['config']['cache_capacity']}",
        "",
        "## Figures",
    ]
    for path in sorted(FIG.glob("*.png")):
        asset_lines.append(f"- `{path.relative_to(ROOT)}`")
    asset_lines.append("")
    asset_lines.append("## Tables")
    for path in sorted(TAB.glob("*")):
        asset_lines.append(f"- `{path.relative_to(ROOT)}`")
    (OUT / "GENERATED_ASSETS.md").write_text("\n".join(asset_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
