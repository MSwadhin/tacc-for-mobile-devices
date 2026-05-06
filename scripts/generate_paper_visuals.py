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
    "keep_current": "Keep Current",
    "local_popularity": "Local Popularity",
    "global_popularity": "Global Popularity",
    "diversified_popularity": "Diversified",
    "topology_greedy": "Topo-Greedy",
    "hybrid_dqn": "Greedy+DQN",
    "online_dqn": "Online DQN",
}

COLORS = {
    "random": "gray",
    "keep_current": "black",
    "local_popularity": "blue",
    "global_popularity": "brown",
    "diversified_popularity": "orange",
    "topology_greedy": "teal",
    "hybrid_dqn": "purple",
    "online_dqn": "green",
}


def fmt(value: float) -> str:
    return f"{value:.3f}"


def write(path: str, lines: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_summary() -> pd.DataFrame:
    return pd.read_csv(SUMMARY)


def objective_chart(summary: pd.DataFrame) -> None:
    policies = ["keep_current", "diversified_popularity", "topology_greedy", "hybrid_dqn", "online_dqn"]
    rates = sorted(summary["perturbation_rate"].unique())
    values = summary[summary["policy"].isin(policies)]["objective"]
    ymin = float(values.min()) * 0.92
    ymax = float(values.max()) * 1.06
    span = max(ymax - ymin, 1e-6)
    x0, xstep = 1.0, 1.25
    y0, height = 0.70, 2.55

    lines = [
        "\\begin{figure}[t]",
        "\\centering",
        "\\begin{tikzpicture}[x=1cm,y=1cm]",
        "\\draw[->] (0.7,0.55) -- (7.85,0.55) node[right] {Perturbation rate};",
        "\\draw[->] (0.7,0.55) -- (0.7,3.45) node[above] {Objective};",
    ]
    for i, rate in enumerate(rates):
        x = x0 + i * xstep
        lines.append(f"\\node[below] at ({x:.2f},0.55) {{{rate:.2f}}};")
        lines.append(f"\\draw[gray!25] ({x:.2f},0.58) -- ({x:.2f},3.25);")
    for tick in [ymin, (ymin + ymax) / 2.0, ymax]:
        y = y0 + height * (tick - ymin) / span
        lines.append(f"\\draw[gray!25] (0.72,{y:.2f}) -- (7.15,{y:.2f});")
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
        style = "very thick" if policy == "online_dqn" else "thick"
        lines.append(f"\\draw[{color}, {style}] {coord_text};")
        for x, y in coords:
            lines.append(f"\\fill[{color}] ({x:.2f},{y:.2f}) circle (1.8pt);")
        ly = 3.20 - pidx * 0.27
        lines.append(f"\\draw[{color}, {style}] (7.95,{ly:.2f}) -- (8.40,{ly:.2f});")
        lines.append(f"\\node[right] at (8.45,{ly:.2f}) {{{DISPLAY[policy]}}};")
    lines += [
        "\\end{tikzpicture}",
        "\\caption{Objective trends for the strongest placement policies across perturbation rates. Online DQN is the transition-aware online controller; Keep Current is the no-update baseline that carries the previous online cache forward.}",
        "\\label{fig:generated_objective_trends}",
        "\\end{figure}",
    ]
    write("fig_objective_trends.tex", lines)


def hit_breakdown(summary: pd.DataFrame) -> None:
    rows = summary[summary["policy"] == "online_dqn"].sort_values("perturbation_rate")
    lines = [
        "\\begin{figure}[t]",
        "\\centering",
        "\\begin{tikzpicture}[x=1cm,y=1cm]",
        "\\draw[->] (0.55,0.45) -- (8.65,0.45) node[right] {Perturbation rate};",
        "\\draw[->] (0.55,0.45) -- (0.55,3.30) node[above] {Request share};",
        "\\node[left] at (0.50,0.45) {0};",
        "\\node[left] at (0.50,3.15) {1};",
    ]
    bar_w = 0.65
    for idx, row in rows.iterrows():
        x = 1.05 + list(rows.index).index(idx) * 1.05
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
        "\\caption{Hit-ratio decomposition for Online DQN. Most served demand is cooperative rather than purely local, showing why graph-aware cache diversity matters.}",
        "\\label{fig:generated_hit_breakdown}",
        "\\end{figure}",
    ]
    write("fig_hit_breakdown.tex", lines)


def cost_components(summary: pd.DataFrame) -> None:
    rows = summary[summary["policy"] == "online_dqn"].sort_values("perturbation_rate").copy()
    rows["rep_penalty"] = 0.02 * rows["replication_cost"]
    rows["red_penalty"] = 18.0 * rows["redundancy_cost"]
    rows["rel_penalty"] = 0.04 * rows["relocation_cost"]
    rows["overhead"] = rows["rep_penalty"] + rows["red_penalty"] + rows["rel_penalty"]
    max_total = float(rows["overhead"].max()) * 1.18
    scale = 2.55 / max(max_total, 1e-6)
    lines = [
        "\\begin{figure}[t]",
        "\\centering",
        "\\begin{tikzpicture}[x=1cm,y=1cm]",
        "\\draw[->] (0.55,0.45) -- (8.65,0.45) node[right] {Perturbation rate};",
        "\\draw[->] (0.55,0.45) -- (0.55,3.35) node[above] {Weighted overhead};",
        "\\node[left] at (0.50,0.45) {0};",
        f"\\node[left] at (0.50,3.00) {{{max_total:.2f}}};",
    ]
    components = [
        ("rep_penalty", "teal!55", "Replication"),
        ("red_penalty", "orange!60", "Redundancy"),
        ("rel_penalty", "red!45", "Relocation"),
    ]
    for ridx, (_, row) in enumerate(rows.iterrows()):
        x = 1.00 + ridx * 1.05
        y = 0.45
        for col, color, _ in components:
            h = float(row[col]) * scale
            lines.append(f"\\fill[{color}] ({x:.2f},{y:.2f}) rectangle ({x+0.60:.2f},{y+h:.2f});")
            y += h
        lines.append(f"\\draw ({x:.2f},0.45) rectangle ({x+0.60:.2f},{y:.2f});")
        lines.append(f"\\node[below] at ({x+0.30:.2f},0.45) {{{row['perturbation_rate']:.2f}}};")
    for lidx, (_, color, label) in enumerate(components):
        y = 3.05 - lidx * 0.30
        lines.append(f"\\fill[{color}] (7.15,{y:.2f}) rectangle (7.45,{y+0.18:.2f});")
        lines.append(f"\\node[right] at (7.50,{y+0.09:.2f}) {{{label}}};")
    lines += [
        "\\end{tikzpicture}",
        "\\caption{Weighted non-access overhead for Online DQN across perturbation windows. Replication, redundancy, and relocation are intentionally small relative to access cost under the configured objective weights, but they grow with perturbation and make cache churn visible.}",
        "\\label{fig:generated_cost_components}",
        "\\end{figure}",
    ]
    write("fig_cost_components.tex", lines)


def metrics_table(summary: pd.DataFrame) -> None:
    policies = ["keep_current", "diversified_popularity", "topology_greedy", "hybrid_dqn", "online_dqn"]
    rates = sorted(summary["perturbation_rate"].unique())
    columns = " & ".join(f"$p={rate:.2f}$" for rate in rates)
    alignment = "ll" + ("r" * len(rates))
    metrics = [
        ("objective", "Obj."),
        ("access_cost", "Access"),
        ("hit_ratio", "Hit"),
        ("replication_cost", "Repl."),
        ("redundancy_cost", "Redun."),
        ("relocation_cost", "Reloc."),
    ]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Full metric summary for the strongest non-duplicate policies across perturbation rates. Replication, redundancy, and relocation are raw normalized cost components before objective weights are applied.}",
        "\\label{tab:generated_compact_metrics}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{1.6pt}",
        f"\\begin{{tabular}}{{{alignment}}}",
        "\\toprule",
        f"Policy & Metric & {columns} \\\\",
        "\\midrule",
    ]
    for policy in policies:
        rows = summary[summary["policy"] == policy].set_index("perturbation_rate")
        for midx, (column, label) in enumerate(metrics):
            values = " & ".join(fmt(rows.loc[r, column]) for r in rates)
            policy_label = DISPLAY[policy] if midx == 0 else ""
            lines.append(f"{policy_label} & {label} & {values} \\\\")
        lines.append("\\addlinespace[1pt]")
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
    cost_components(summary)
    metrics_table(summary)
    mobility_graph()
    print(f"Wrote generated visuals to {OUT}")


if __name__ == "__main__":
    main()
