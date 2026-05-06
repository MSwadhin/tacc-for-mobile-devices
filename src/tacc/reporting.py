from __future__ import annotations

from pathlib import Path

import pandas as pd


DISPLAY_NAMES = {
    "random": "Random",
    "keep_current": "Keep Current",
    "local_popularity": "Local Popularity",
    "global_popularity": "Global Popularity",
    "diversified_popularity": "Diversified Popularity",
    "topology_greedy": "Topology-Aware Greedy",
    "hybrid_dqn": "Greedy + DQN Refinement",
    "online_dqn": "Online DQN Refinement",
}


def write_latex_assets(summary: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    best_rate = summary["perturbation_rate"].min()
    table_df = summary[summary["perturbation_rate"] == best_rate].copy()
    table_df = table_df.sort_values("objective")

    table_lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Measured performance at the lowest perturbation setting. Lower objective, access cost, replication, redundancy, and relocation are better; higher hit ratio is better.}",
        "\\label{tab:measured_results}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2pt}",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Policy & Obj. & Access & Hit & Repl. & Redun. & Reloc. \\\\",
        "\\midrule",
    ]
    for _, row in table_df.iterrows():
        name = DISPLAY_NAMES.get(row["policy"], row["policy"])
        objective = f"{row['objective']:.3f}"
        if "objective_std" in row and not pd.isna(row["objective_std"]):
            objective = f"{row['objective']:.3f}$\\pm${row['objective_std']:.3f}"
        table_lines.append(
            f"{name} & {objective} & {row['access_cost']:.3f} & "
            f"{row['hit_ratio']:.3f} & {row['replication_cost']:.3f} & "
            f"{row['redundancy_cost']:.3f} & {row['relocation_cost']:.3f} \\\\"
        )
    table_lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    (output_dir / "result_table.tex").write_text("\n".join(table_lines), encoding="utf-8")

    compact_policies = ["keep_current", "diversified_popularity", "topology_greedy", "hybrid_dqn", "online_dqn"]
    compact = summary[summary["policy"].isin(compact_policies)].copy()
    rates = sorted(compact["perturbation_rate"].unique())
    columns = " & ".join(f"$p={rate:.2f}$" for rate in rates)
    alignment = "l" + ("r" * len(rates))
    compact_lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Objective values across topology perturbation rates for the strongest policies. Lower is better.}",
        "\\label{tab:perturbation_results}",
        f"\\begin{{tabular}}{{{alignment}}}",
        "\\toprule",
        f"Policy & {columns} \\\\",
        "\\midrule",
    ]
    for policy in compact_policies:
        row = compact[compact["policy"] == policy].set_index("perturbation_rate")
        values = []
        for rate in rates:
            values.append(f"{row.loc[rate, 'objective']:.3f}")
        compact_lines.append(f"{DISPLAY_NAMES[policy]} & {' & '.join(values)} \\\\")
    compact_lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    (output_dir / "perturbation_table.tex").write_text("\n".join(compact_lines), encoding="utf-8")

    policies = [
        "random",
        "local_popularity",
        "diversified_popularity",
        "topology_greedy",
        "online_dqn",
    ]
    filtered = summary[summary["policy"].isin(policies)].copy()
    min_obj = filtered["objective"].min()
    max_obj = filtered["objective"].max()
    span = max(max_obj - min_obj, 1e-6)
    colors = {
        "random": "gray",
        "local_popularity": "blue",
        "diversified_popularity": "orange",
        "topology_greedy": "teal",
        "online_dqn": "green",
    }
    y_positions = {policy: 0.8 + idx * 0.45 for idx, policy in enumerate(policies)}
    rates = sorted(filtered["perturbation_rate"].unique())
    x_positions = {rate: 1.0 + idx * 1.25 for idx, rate in enumerate(rates)}

    lines = [
        "\\begin{figure}[t]",
        "\\centering",
        "\\begin{tikzpicture}[x=1cm,y=1cm]",
        "\\draw[->] (0.7,0.55) -- (7.85,0.55) node[right] {Perturbation rate};",
        "\\draw[->] (0.7,0.55) -- (0.7,3.1) node[above] {Objective};",
    ]
    for rate, x in x_positions.items():
        lines.append(f"\\node[below] at ({x:.2f},0.55) {{{rate:.2f}}};")
    for policy in policies:
        rows = filtered[filtered["policy"] == policy].sort_values("perturbation_rate")
        coords = []
        for _, row in rows.iterrows():
            x = x_positions[row["perturbation_rate"]]
            y = 0.75 + 2.1 * (row["objective"] - min_obj) / span
            coords.append(f"({x:.2f},{y:.2f})")
        color = colors[policy]
        lines.append(f"\\draw[{color}, thick] {' -- '.join(coords)};")
        for coord in coords:
            lines.append(f"\\fill[{color}] {coord} circle (1.6pt);")
        legend_y = y_positions[policy]
        lines.append(f"\\draw[{color}, thick] (7.95,{legend_y:.2f}) -- (8.40,{legend_y:.2f});")
        lines.append(f"\\node[right] at (8.45,{legend_y:.2f}) {{{DISPLAY_NAMES[policy]}}};")
    lines.extend(
        [
            "\\end{tikzpicture}",
            "\\caption{Objective value under topology perturbation. Online DQN is evaluated as a sequential cache-update policy against no-update, diversity-aware placement, topology-aware greedy, and fixed Greedy+DQN baselines.}",
            "\\label{fig:objective_perturbation}",
            "\\end{figure}",
            "",
        ]
    )
    (output_dir / "result_figure.tex").write_text("\n".join(lines), encoding="utf-8")
