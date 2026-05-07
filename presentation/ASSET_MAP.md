# Asset Map

All generated visuals are available as both PNG and SVG in `presentation/figures/`. Use SVG for maximum quality when the presentation tool supports it.

## Figure Mapping

| Asset | Primary Slides | Purpose | Source |
| --- | --- | --- | --- |
| `fig_research_flow` | 2 | Introduce the complete research pipeline at a high level. | Conceptual summary from paper sections 1, 7, 9 |
| `fig_edge_caching_motivation` | 3 | Explain local edge hit versus remote origin fetch. | Background and motivation |
| `fig_challenge_mapping` | 4 | Map caching challenges to modeling decisions. | Paper Table I / Background |
| `fig_research_gap` | 5 | Identify the gap: static topology, popularity-only caching, missing update cost. | Introduction and related work |
| `fig_research_questions` | 6 | Present RQ1-RQ4 visually. | Research questions |
| `fig_system_model` | 7, 9 | Show graph, demand, placement, and objective terms. | System model and problem formulation |
| `fig_mobility_graph_concept` | 1, 8, 10 | Visualize AP cooperation graph idea. | Conceptual rendering based on 20-node graph scale |
| `fig_method_pipeline` | 11 | Explain the executable pipeline from mobility trace to evaluation. | Methodology and implementation |
| `fig_baseline_taxonomy` | 12 | Explain why the baseline set is strong. | Evaluation protocol |
| `fig_greedy_flow` | 13 | Explain topology-aware greedy marginal-gain search. | Algorithm 1 and placement implementation |
| `fig_dqn_loop` | 14 | Explain state, action, reward, accepted-action inference. | DQN methodology |
| `fig_experiment_protocol` | 15 | Summarize final run settings and objective weights. | Evaluation parameters and run metadata |
| `fig_nominal_baseline_bars` | 16 | Show nominal `p=0.00` objective ranking. | `results/final/summary_metrics.csv` |
| `fig_objective_trends` | 17 | Show objective across perturbation rates. | `results/final/summary_metrics.csv` |
| `fig_p025_closeup` | 18 | Highlight that Keep Current beats Online DQN at `p=0.25`. | `results/final/summary_metrics.csv` |
| `fig_hit_breakdown` | 19 | Show local hit, cooperative hit, and origin miss decomposition. | `results/final/summary_metrics.csv` |
| `fig_weighted_cost_components` | 20 | Show weighted replication, redundancy, and relocation overheads. | `results/final/summary_metrics.csv` |
| `fig_objective_heatmap` | 21 | Compare strong policies across perturbation rates compactly. | `results/final/summary_metrics.csv` |
| `fig_rq_answers` | 22 | Summarize answers to all research questions. | Evaluation and discussion |
| `fig_limitations` | 23 | Present threats to validity without crowding the slide. | Limitations |
| `fig_future_work` | 24 | Show future work as a roadmap. | Future Work |
| `fig_takeaway` | 25 | Close with the three core takeaways. | Conclusion |

## Table Mapping

| Table | Primary Slides | Use |
| --- | --- | --- |
| `nominal_metrics.md` / `.csv` | 16 | Full nominal comparison across all policies. |
| `perturbation_objectives.md` / `.csv` | 17, 18, 21 | Objective values by perturbation rate for strong policies. |
| `online_dqn_metrics.md` / `.csv` | 20, 22 | Online DQN cost and relocation values across perturbation rates. |

## Original Paper Assets

The paper also contains LaTeX/TikZ assets in `Paper/final_paper/generated/`:

- `fig_objective_trends.tex`
- `fig_hit_breakdown.tex`
- `fig_cost_components.tex`
- `fig_mobility_graph.tex`
- `table_compact_metrics.tex`

These are useful if the final presentation is built in LaTeX/Beamer. The `presentation/figures/` images are easier for PowerPoint or Google Slides.

## Slide Build Notes

- Put each figure on a mostly full-width slide.
- Use a short caption or single takeaway line below the figure.
- Keep detailed numeric interpretation in speaker notes.
- Do not show both `fig_objective_trends` and `fig_objective_heatmap` without explaining the different purposes: trend line for story, heatmap for compact comparison.
