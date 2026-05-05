# Project Log: Topology-Aware Cooperative Caching for IoT Edge Networks

## Goal

Complete the project described in `goal.txt`: analyze the existing report and PDF, identify methodology and implementation gaps, fill them with research-backed choices, derive a complete pipeline, acquire or prepare the dataset, implement and run experiments, improve the results, and expand the LaTeX report into a complete paper with figures, tables, metrics, discussion, conclusion, and BibTeX references.

## Current Baseline

- Existing paper sources live in `Paper/report/`.
- The compiled reference PDF is `Paper/current_version.pdf`.
- The active report entry point is `Paper/report/main.tex`.
- The current report describes a topology-aware cooperative caching idea, but the implementation, dataset processing, actual measured results, reproducible scripts, and final results tables/figures are missing.
- The repository currently has no implementation code outside the paper sources.

## Initial Gaps

### Methodology Gaps

- The current methodology names a greedy plus DRL pipeline but does not specify concrete state vectors, action indexing, neural architecture, training hyperparameters, convergence criteria, or reproducibility settings.
- Graph construction from the Dartmouth mobility trace is described conceptually, but the parsing format, filtering rules, handoff extraction, node selection, graph weighting, and normalization choices are not yet operational.
- Demand generation uses Zipf popularity but needs an explicit spatial-locality model, temporal train/test split, and controlled random seeds.
- The perturbation model mentions node additions/removals but needs measurable rates, edge removal handling, and evaluation protocol across repeated samples.
- The objective function needs a direct mapping from symbols to implementation parameters and reported metrics.

### Implementation and Evaluation Gaps

- No downloader, parser, simulator, baselines, RL agent, experiment runner, result files, or plots currently exist.
- Evaluation text currently claims results without measured outputs.
- Required baselines should include at least random placement, local popularity, global popularity/greedy, topology-aware greedy, and hybrid greedy plus learning refinement.
- Metrics should include average access cost, cache hit ratio, local hit ratio, cooperative hit ratio, origin miss ratio, redundancy ratio, relocation overhead, and runtime.
- Experiments need repeated seeds and perturbation-rate sweeps so the final numbers are defensible.

### Writing and Report Gaps

- The report needs a stronger dataset section, an implementation section, reproducibility details, measured tables/figures, limitations, and conclusion.
- Some BibTeX entries are incomplete or inconsistent, including duplicated keys and entries whose venue/type metadata should be corrected.
- Several section files are short placeholders and need expansion into a coherent 30-page paper.

## Solid Plan

1. Inspect and summarize current paper/PDF context.
2. Build a persistent implementation structure:
   - `src/tacc/` for reusable pipeline code.
   - `scripts/` for dataset acquisition and experiment execution.
   - `data/` for downloaded or fallback generated traces.
   - `results/` for metrics, plots, and generated LaTeX snippets.
3. Implement dataset acquisition:
   - Try to download the public Dartmouth/CRAWDAD movement trace from official sources.
   - If the dataset is gated or unavailable without account login, document that constraint and provide a deterministic synthetic Dartmouth-like mobility fallback so experiments remain reproducible.
4. Implement the full pipeline:
   - Parse mobility traces into user AP sequences.
   - Construct a weighted handoff graph.
   - Generate Zipf plus spatial-locality demand.
   - Generate topology perturbations.
   - Evaluate placements with the stated objective.
   - Implement baselines and topology-aware greedy.
   - Implement a compact DQN-style refinement agent in PyTorch.
5. Run experiments:
   - Use small enough defaults for reliable local execution.
   - Sweep perturbation rates and compare baselines.
   - Save CSV/JSON summaries and report-ready tables/figures.
   - Improve hyperparameters if the hybrid result is not competitive.
6. Expand the LaTeX report:
   - Preserve the currently used section structure.
   - Add detailed methodology, implementation, experimental setup, measured results, ablation, discussion, limitations, and conclusion.
   - Add figures/tables from the experiment outputs.
   - Keep original references and add needed BibTeX entries.
7. Verify:
   - Run the pipeline end to end.
   - Attempt LaTeX compilation if a TeX toolchain is available.
   - Record any environment limitations clearly.

## Execution Notes

- 2026-05-05: Read `goal.txt` and active LaTeX sources. Confirmed that the paper exists but implementation and measured results are missing.
- 2026-05-05: Checked local tools. `torch`, `networkx`, `numpy`, `pandas`, `scipy`, `sklearn`, and `plotly` are available. `pdftotext`, `pdfinfo`, and TeX compilers are not available.
- 2026-05-05: Checked official Dartmouth/CRAWDAD/IEEE DataPort dataset access. Direct old CRAWDAD download URLs return 404. The official Dartmouth page points to IEEE DataPort, so the repo records landing pages and an access note rather than redistributing raw data.
- 2026-05-05: Implemented `src/tacc/` package and scripts for dataset access, synthetic fallback generation, graph construction, Zipf-locality demand, perturbation sampling, objective evaluation, baselines, topology-aware greedy, DQN refinement, and LaTeX result generation.
- 2026-05-05: Smoke test found and fixed a boolean relocation-cost bug.
- 2026-05-05: Final experiment completed in 145.2 seconds. Final run used 20 AP nodes, 190 mobility edges, 48 content objects, cache capacity 5, and 12 samples per perturbation rate.
- 2026-05-05: Final measured results: hybrid DQN objective 2.414 at perturbation 0.00 versus topology-aware greedy 2.633 and diversified popularity 2.739. Hybrid remained better than greedy at all tested perturbation rates.
- 2026-05-05: Expanded report sections with abstract, dataset access/preprocessing, implementation details, DQN specifics, measured evaluation tables/figures, updated discussion, and revised conclusion.
- 2026-05-05: Verification completed: Python files compile, active LaTeX citations resolve against `reference.bib`, and no duplicate BibTeX keys remain. A local TeX compiler is not installed, so PDF compilation could not be performed in this environment.
- 2026-05-05: Created a clean final paper in `Paper/final_paper/` with `main.tex`, copied `reference.bib`, and separate section files under `Paper/final_paper/sections/`. The final paper includes research questions, gap closure, related-work comparison, system model, dataset and demand construction, methodology, implementation, evaluation, discussion, limitations, and conclusion.
- 2026-05-05: Verified the final paper structure: all `\input{}` targets exist, all active citations resolve against the copied bibliography, and the final paper contains 6,368 words across 13 section files plus `main.tex`. PDF compilation remains blocked by missing local TeX tools.
- 2026-05-05: Added `scripts/generate_paper_visuals.py` to generate paper-ready TikZ/LaTeX visuals from `results/final/summary_metrics.csv` and the trace-compatible movement graph. Generated visuals include objective trend graph, hit-breakdown stacked bars, cost-component stacked bars, adaptive-selector pie chart, mobility graph, and compact metrics table. Included these generated assets in `Paper/final_paper/sections/06_dataset.tex` and `Paper/final_paper/sections/09_evaluation.tex`.

## Progress Ledger

- [x] Read project goal.
- [x] Inspect current report sources.
- [x] Create persistent project log.
- [x] Finish dataset access decision.
- [x] Implement reproducible pipeline.
- [x] Run and improve experiments.
- [x] Generate report assets.
- [x] Expand report.
- [x] Verify final artifacts.
