#!/usr/bin/env python3
from __future__ import annotations

from math import cos, pi, sin
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tacc.data import load_movement_trace
from tacc.graph import GraphBuildConfig, build_handoff_graph


OUT = ROOT / "Paper" / "final_paper" / "generated"
SUMMARY = ROOT / "results" / "final" / "summary_metrics.csv"
TRACE = ROOT / "data" / "raw" / "dartmouth_movement.csv"

DISPLAY = {
    "random": "Random",
    "local_popularity": "Local Popularity",
    "global_popularity": "Global Popularity",
    "diversified_popularity": "Diversified",
    "topology_greedy": "Topo-Greedy",
    "hybrid_dqn": "Greedy+DQN",
    "adaptive_tacc": "Adaptive TACC",
}

COLORS = {
    "random": "gray",
    "local_popularity": "blue",
    "global_popularity": "brown",
    "diversified_popularity": "orange",
    "topology_greedy": "teal",
    "hybrid_dqn": "purple",
    "adaptive_tacc": "red",
}


def fmt(value: float) -> str:
    return f"{value:.3f}"


def write(path: str, lines: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_summary() -> pd.DataFrame:
    return pd.read_csv(SUMMARY)


def objective_chart(summary: pd.DataFrame) -> None:
    policies = ["diversified_popularity", "topology_greedy", "hybrid_dqn", "adaptive_tacc"]
    rates = sorted(summary["perturbation_rate"].unique())
    values = summary[summary["policy"].isin(policies)]["objective"]
    ymin = float(values.min()) * 0.92
    ymax = float(values.max()) * 1.06
    span = max(ymax - ymin, 1e-6)
    x0, xstep = 1.0, 2.0
    y0, height = 0.70, 2.55

    lines = [
        "\\begin{figure}[t]",
        "\\centering",
        "\\begin{tikzpicture}[x=1cm,y=1cm]",
        "\\draw[->] (0.7,0.55) -- (7.65,0.55) node[right] {Perturbation rate};",
        "\\draw[->] (0.7,0.55) -- (0.7,3.45) node[above] {Objective};",
    ]
    for i, rate in enumerate(rates):
        x = x0 + i * xstep
        lines.append(f"\\node[below] at ({x:.2f},0.55) {{{rate:.2f}}};")
        lines.append(f"\\draw[gray!25] ({x:.2f},0.58) -- ({x:.2f},3.25);")
    for tick in [ymin, (ymin + ymax) / 2.0, ymax]:
        y = y0 + height * (tick - ymin) / span
        lines.append(f"\\draw[gray!25] (0.72,{y:.2f}) -- (7.25,{y:.2f});")
        lines.append(f"\\node[left] at (0.68,{y:.2f}) {{{tick:.1f}}};")

    for pidx, policy in enumerate(policies):
        rows = summary[summary["policy"] == policy].sort_values("perturbation_rate")
        coords = []
        for _, row in rows.iterrows():
            x = x0 + rates.index(row["perturbation_rate"]) * xstep
            y = y0 + height * (row["objective"] - ymin) / span
            coords.append((x, y))
        coord_text = " -- ".join(f"({x:.2f},{y:.2f})" for x, y in coords)
        color = COLORS[policy]
        style = "very thick" if policy == "adaptive_tacc" else "thick"
        lines.append(f"\\draw[{color}, {style}] {coord_text};")
        for x, y in coords:
            lines.append(f"\\fill[{color}] ({x:.2f},{y:.2f}) circle (1.8pt);")
        ly = 3.15 - pidx * 0.32
        lines.append(f"\\draw[{color}, {style}] (7.75,{ly:.2f}) -- (8.20,{ly:.2f});")
        lines.append(f"\\node[right] at (8.25,{ly:.2f}) {{{DISPLAY[policy]}}};")
    lines += [
        "\\end{tikzpicture}",
        "\\caption{Objective trends for the strongest placement policies across perturbation rates. Adaptive TACC follows a cost-aware regime-specific policy rather than forcing one static placement rule or always invoking DQN refinement.}",
        "\\label{fig:generated_objective_trends}",
        "\\end{figure}",
    ]
    write("fig_objective_trends.tex", lines)


def hit_breakdown(summary: pd.DataFrame) -> None:
    rows = summary[summary["policy"] == "adaptive_tacc"].sort_values("perturbation_rate")
    lines = [
        "\\begin{figure}[t]",
        "\\centering",
        "\\begin{tikzpicture}[x=1cm,y=1cm]",
        "\\draw[->] (0.55,0.45) -- (8.30,0.45) node[right] {Perturbation rate};",
        "\\draw[->] (0.55,0.45) -- (0.55,3.30) node[above] {Request share};",
        "\\node[left] at (0.50,0.45) {0};",
        "\\node[left] at (0.50,3.15) {1};",
    ]
    bar_w = 0.65
    for idx, row in rows.iterrows():
        x = 1.15 + list(rows.index).index(idx) * 1.7
        y = 0.45
        local = row["local_hit_ratio"] * 2.7
        coop = row["cooperative_hit_ratio"] * 2.7
        miss = row["origin_miss_ratio"] * 2.7
        lines.append(f"\\fill[teal!70] ({x:.2f},{y:.2f}) rectangle ({x+bar_w:.2f},{y+local:.2f});")
        y += local
        lines.append(f"\\fill[blue!45] ({x:.2f},{y:.2f}) rectangle ({x+bar_w:.2f},{y+coop:.2f});")
        y += coop
        lines.append(f"\\fill[red!45] ({x:.2f},{y:.2f}) rectangle ({x+bar_w:.2f},{y+miss:.2f});")
        lines.append(f"\\draw ({x:.2f},0.45) rectangle ({x+bar_w:.2f},3.15);")
        lines.append(f"\\node[below] at ({x+bar_w/2:.2f},0.45) {{{row['perturbation_rate']:.2f}}};")
    lines += [
        "\\fill[teal!70] (7.25,2.75) rectangle (7.55,3.00); \\node[right] at (7.60,2.88) {Local hit};",
        "\\fill[blue!45] (7.25,2.35) rectangle (7.55,2.60); \\node[right] at (7.60,2.48) {Cooperative hit};",
        "\\fill[red!45] (7.25,1.95) rectangle (7.55,2.20); \\node[right] at (7.60,2.08) {Origin miss};",
        "\\end{tikzpicture}",
        "\\caption{Hit-ratio decomposition for Adaptive TACC. Most served demand is cooperative rather than purely local, showing why graph-aware cache diversity matters.}",
        "\\label{fig:generated_hit_breakdown}",
        "\\end{figure}",
    ]
    write("fig_hit_breakdown.tex", lines)


def selector_pie(summary: pd.DataFrame) -> None:
    rows = summary[summary["policy"] == "adaptive_tacc"].sort_values("perturbation_rate")
    counts = rows["selected_policy"].value_counts().to_dict()
    total = sum(counts.values())
    start = 0.0
    lines = [
        "\\begin{figure}[t]",
        "\\centering",
        "\\begin{tikzpicture}[x=1cm,y=1cm]",
    ]
    label_map = {
        "hybrid_dqn": "Greedy+DQN",
        "diversified_popularity": "Diversified",
        "topology_greedy": "Topo-Greedy",
    }
    colors = {"hybrid_dqn": "purple", "diversified_popularity": "orange", "topology_greedy": "teal"}
    for policy, count in counts.items():
        frac = count / total
        end = start + frac * 360.0
        mid = (start + end) / 2.0
        x = 1.15 * cos(mid * pi / 180.0)
        y = 1.15 * sin(mid * pi / 180.0)
        large = "1" if end - start > 180 else "0"
        # TikZ arc handles angles directly; the large flag is not needed but keeping arcs explicit is clearer.
        lines.append(
            f"\\fill[{colors[policy]}!65] (0,0) -- ({start:.2f}:1.45) arc ({start:.2f}:{end:.2f}:1.45) -- cycle;"
        )
        lines.append(f"\\node at ({x:.2f},{y:.2f}) {{{count}/{total}}};")
        start = end
    lines += [
        "\\draw (0,0) circle (1.45);",
    ]
    for idx, policy in enumerate(sorted(counts, key=lambda name: label_map[name])):
        y = 0.75 - idx * 0.45
        lines.append(
            f"\\fill[{colors[policy]}!65] (2.25,{y-0.15:.2f}) rectangle (2.55,{y+0.15:.2f}); "
            f"\\node[right] at (2.60,{y:.2f}) {{{label_map[policy]}}};"
        )
    lines += [
        "\\end{tikzpicture}",
        "\\caption{Adaptive TACC policy-selection frequency across the four perturbation regimes. The cost-aware selector uses topology-aware greedy when DQN's marginal gain is too small to justify added computation, diversified placement in moderate perturbation, and DQN refinement when high-churn instability makes refinement worthwhile.}",
        "\\label{fig:generated_selector_pie}",
        "\\end{figure}",
    ]
    write("fig_selector_pie.tex", lines)


def cost_components(summary: pd.DataFrame) -> None:
    rate = 0.0
    policies = ["diversified_popularity", "topology_greedy", "hybrid_dqn", "adaptive_tacc"]
    rows = summary[(summary["perturbation_rate"] == rate) & (summary["policy"].isin(policies))].copy()
    rows["rep_penalty"] = 0.02 * rows["replication_cost"]
    rows["red_penalty"] = 18.0 * rows["redundancy_cost"]
    rows["rel_penalty"] = 0.04 * rows["relocation_cost"]
    max_total = float(rows["objective"].max()) * 1.08
    scale = 2.7 / max_total
    lines = [
        "\\begin{figure}[t]",
        "\\centering",
        "\\begin{tikzpicture}[x=1cm,y=1cm]",
        "\\draw[->] (0.55,0.45) -- (8.70,0.45) node[right] {Policy};",
        "\\draw[->] (0.55,0.45) -- (0.55,3.35) node[above] {Objective components};",
    ]
    components = [
        ("access_cost", "blue!45", "Access"),
        ("rep_penalty", "teal!55", "Replication"),
        ("red_penalty", "orange!60", "Redundancy"),
        ("rel_penalty", "red!45", "Relocation"),
    ]
    for ridx, (_, row) in enumerate(rows.iterrows()):
        x = 1.05 + ridx * 1.45
        y = 0.45
        for col, color, _ in components:
            h = float(row[col]) * scale
            lines.append(f"\\fill[{color}] ({x:.2f},{y:.2f}) rectangle ({x+0.60:.2f},{y+h:.2f});")
            y += h
        lines.append(f"\\draw ({x:.2f},0.45) rectangle ({x+0.60:.2f},{y:.2f});")
        lines.append(f"\\node[rotate=35, anchor=east] at ({x+0.55:.2f},0.28) {{{DISPLAY[row['policy']]}}};")
    for lidx, (_, color, label) in enumerate(components):
        y = 3.05 - lidx * 0.30
        lines.append(f"\\fill[{color}] (6.65,{y:.2f}) rectangle (6.95,{y+0.18:.2f});")
        lines.append(f"\\node[right] at (7.00,{y+0.09:.2f}) {{{label}}};")
    lines += [
        "\\end{tikzpicture}",
        "\\caption{Objective component breakdown at nominal perturbation. Access cost dominates the objective, while redundancy and relocation distinguish the strongest policies.}",
        "\\label{fig:generated_cost_components}",
        "\\end{figure}",
    ]
    write("fig_cost_components.tex", lines)


def metrics_table(summary: pd.DataFrame) -> None:
    policies = ["diversified_popularity", "topology_greedy", "hybrid_dqn", "adaptive_tacc"]
    rates = [0.00, 0.05, 0.10, 0.15]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Paper-ready summary of objective and hit ratio for the strongest policies.}",
        "\\label{tab:generated_compact_metrics}",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Policy & Metric & $p=0.00$ & $p=0.05$ & $p=0.10$ & $p=0.15$ \\\\",
        "\\midrule",
    ]
    for policy in policies:
        rows = summary[summary["policy"] == policy].set_index("perturbation_rate")
        obj = " & ".join(fmt(rows.loc[r, "objective"]) for r in rates)
        hit = " & ".join(fmt(rows.loc[r, "hit_ratio"]) for r in rates)
        lines.append(f"{DISPLAY[policy]} & Objective & {obj} \\\\")
        lines.append(f" & Hit ratio & {hit} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("table_compact_metrics.tex", lines)


def mobility_graph() -> None:
    trace = load_movement_trace(TRACE)
    graph = build_handoff_graph(trace, GraphBuildConfig(max_nodes=12, min_handoffs=2))
    nodes = list(graph.nodes())[:12]
    n = len(nodes)
    positions = {}
    for idx, node in enumerate(nodes):
        angle = 2 * pi * idx / n
        positions[node] = (2.2 * cos(angle), 2.2 * sin(angle))
    edges = sorted(
        [(u, v, d.get("weight", 1.0)) for u, v, d in graph.edges(data=True) if u in positions and v in positions],
        key=lambda item: item[2],
        reverse=True,
    )[:24]
    max_w = max([w for _, _, w in edges] or [1.0])
    lines = [
        "\\begin{figure}[t]",
        "\\centering",
        "\\begin{tikzpicture}[x=1cm,y=1cm, every node/.style={font=\\scriptsize}]",
    ]
    for u, v, w in edges:
        x1, y1 = positions[u]
        x2, y2 = positions[v]
        width = 0.25 + 1.25 * w / max_w
        lines.append(f"\\draw[gray!45, line width={width:.2f}pt] ({x1:.2f},{y1:.2f}) -- ({x2:.2f},{y2:.2f});")
    for node, (x, y) in positions.items():
        label = node.replace("_", "\\_")
        lines.append(f"\\fill[teal!70] ({x:.2f},{y:.2f}) circle (4pt);")
        lines.append(f"\\node[above right] at ({x:.2f},{y:.2f}) {{{label}}};")
    lines += [
        "\\end{tikzpicture}",
        "\\caption{Mobility-derived AP handoff graph sample generated from the trace-compatible movement data. Edge thickness reflects relative handoff count among the strongest displayed links.}",
        "\\label{fig:generated_mobility_graph}",
        "\\end{figure}",
    ]
    write("fig_mobility_graph.tex", lines)


def main() -> None:
    summary = load_summary()
    objective_chart(summary)
    hit_breakdown(summary)
    selector_pie(summary)
    cost_components(summary)
    metrics_table(summary)
    mobility_graph()
    print(f"Wrote generated visuals to {OUT}")


if __name__ == "__main__":
    main()
